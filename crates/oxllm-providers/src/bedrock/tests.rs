use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn bedrock_deployment(model: &str) -> Deployment {
    Deployment {
        model_name: "bedrock-claude".to_string(),
        oxllm_params: OxllmParams {
            model: model.to_string(),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn current_gpt_families_and_regional_aliases_use_converse() {
    for model in [
        "bedrock/openai.gpt-6.1-sol",
        "bedrock/us.openai.gpt-6.1-sol",
        "bedrock/global.openai.gpt-6.1-sol",
    ] {
        let deployment = bedrock_deployment(model);
        assert!(
            BedrockProvider
                .endpoint_url(&deployment)
                .ends_with("/converse")
        );
        let body = BedrockProvider
            .transform_request(
                &json!({
                    "messages":[{"role":"user","content":"hi"}],"max_tokens":100,
                    "temperature":0.3,"top_p":0.8
                }),
                &deployment,
            )
            .unwrap();
        assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
        assert_eq!(body["inferenceConfig"]["maxTokens"], 100);
        assert!(body["inferenceConfig"].get("temperature").is_none());
        assert!(body["inferenceConfig"].get("topP").is_none());
        assert!(body.get("anthropic_version").is_none());
    }
}

#[test]
fn invoke_messages_preserve_per_message_output_configuration_and_merge_beta() {
    let beta = crate::anthropic::BETA_MID_CONVERSATION_OUTPUT_CONFIG;
    let deployment = bedrock_deployment("bedrock/anthropic.claude-sonnet-5-5");
    let request = json!({
        "messages":[{"role":"user","content":"hi","output_config":{"effort":"high"}}],
        "anthropic_beta":["existing-feature", beta],
        "max_tokens":100
    });
    let body = BedrockProvider
        .transform_request(&request, &deployment)
        .unwrap();
    assert_eq!(
        body["messages"][0]["output_config"],
        request["messages"][0]["output_config"]
    );
    let betas = body["anthropic_beta"].as_array().unwrap();
    assert_eq!(
        betas
            .iter()
            .filter(|value| value.as_str() == Some(beta))
            .count(),
        1
    );
    assert!(
        betas
            .iter()
            .any(|value| value.as_str() == Some("existing-feature"))
    );
    assert!(body.get("model").is_none());
}

#[test]
fn endpoint_url_strips_bedrock_prefix() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet-20241022-v2:0/invoke"
    );
}

#[test]
fn endpoint_url_without_bedrock_prefix() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("anthropic.claude-3-5-sonnet-20241022-v2:0");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet-20241022-v2:0/invoke"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = BedrockProvider;
    let mut dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    dep.oxllm_params.api_base = Some("https://bedrock-runtime.eu-west-1.amazonaws.com".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/invoke"
    );
}

#[test]
fn endpoint_url_uses_aws_region_when_api_base_absent() {
    let p = BedrockProvider;
    let mut dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    dep.oxllm_params
        .extra
        .insert("aws_region".to_string(), json!("eu-west-1"));
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/invoke"
    );
}

#[test]
fn endpoint_url_percent_encodes_arn_with_slash() {
    let p = BedrockProvider;
    let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123";
    let dep = bedrock_deployment(&format!("bedrock/{arn}"));
    let url = p.endpoint_url(&dep);
    assert!(
        url.ends_with("/converse") || url.contains("%2F"),
        "ARN with `/` must be percent-encoded as one path segment, got: {url}"
    );
    assert!(
        url.contains("application-inference-profile%2Fabc123"),
        "expected encoded profile segment, got: {url}"
    );
    // Terminal path segment must remain /invoke or /converse (not split).
    assert!(
        url.ends_with("/invoke") || url.ends_with("/converse"),
        "URL must end with /invoke or /converse, got: {url}"
    );
}

#[test]
fn endpoint_url_trims_trailing_slash_in_base() {
    let p = BedrockProvider;
    let mut dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    dep.oxllm_params.api_base =
        Some("https://bedrock-runtime.eu-west-1.amazonaws.com/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/invoke"
    );
}

#[test]
fn transform_request_inlines_anthropic_version_for_anthropic_models() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0");
    let req = json!({
    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    });
    let out = p.transform_request(&req, &dep).expect("transform ok");
    assert_eq!(out["anthropic_version"], ANTHROPIC_VERSION);
    assert!(out.get("stream").is_none(), "stream must be stripped");
    // Bedrock infers the model from the URL path — the body must not carry
    // a `model` field for Anthropic Messages calls.
    assert!(
        out.get("model").is_none(),
        "model must be dropped from the Anthropic body"
    );
    let messages = out["messages"].as_array().expect("messages is array");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}

#[test]
fn transform_request_removes_model_from_body_for_anthropic() {
    // The Anthropic Messages adapter on Bedrock infers the model from the
    // URL path; a `model` body field is rejected ("Extra inputs are not
    // permitted").
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    let req = json!({
    "model": "bedrock/anthropic.claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    });
    let out = p.transform_request(&req, &dep).expect("transform ok");
    assert!(
        out.get("model").is_none(),
        "model must be dropped from the Anthropic body"
    );
}

#[test]
fn transform_request_handles_wildcard_model() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/*");
    let req = json!({
    "model": "bedrock/anthropic.claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    });
    let out = p.transform_request(&req, &dep).expect("transform ok");
    assert_eq!(out["anthropic_version"], ANTHROPIC_VERSION);
    assert!(out.get("model").is_none());
    assert!(
        p.endpoint_url_for_request(&dep, &req)
            .ends_with("/model/anthropic.claude-3-5-sonnet/invoke")
    );
}

#[test]
fn transform_request_passes_non_anthropic_through() {
    // Unknown vendors fall through to a passthrough that forwards the
    // OpenAI-shaped body verbatim with the resolved model id.
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/custom.unknown-model-v1:0");
    let req = json!({
    "model": "custom.unknown-model-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    "temperature": 0.7,
    });
    let out = p.transform_request(&req, &dep).expect("transform ok");
    assert!(out.get("anthropic_version").is_none());
    assert_eq!(out["model"], "custom.unknown-model-v1:0");
    assert_eq!(out["temperature"], 0.7);
}

#[test]
fn transform_request_non_anthropic_drops_client_model_for_configured() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/custom.unknown-model-v1:0");
    let req = json!({
    "model": "anything-else",
    "messages": [],
    });
    let out = p.transform_request(&req, &dep).expect("transform ok");
    assert_eq!(out["model"], "custom.unknown-model-v1:0");
}

#[test]
fn transform_response_uses_anthropic_shape_for_anthropic_models() {
    let p = BedrockProvider;
    let body = json!({
    "id": "msg_abc",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022-v2:0",
    "content": [{"type": "text", "text": "Bonjour!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 3},
    });
    let out = p
        .transform_response(&body, "anthropic.claude-3-5-sonnet-20241022-v2:0")
        .expect("ok");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(out["id"], "msg_abc");
    assert_eq!(out["choices"][0]["message"]["content"], "Bonjour!");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
    assert_eq!(out["usage"]["prompt_tokens"], 10);
    assert_eq!(out["usage"]["completion_tokens"], 3);
}

#[test]
fn transform_response_passthrough_for_non_anthropic_models() {
    let p = BedrockProvider;
    let body = json!({"some": "shape"});
    let out = p
        .transform_response(&body, "meta.llama3-1-70b")
        .expect("ok");
    assert_eq!(out, body);
}

#[test]
fn transform_response_meta_llama_maps_generation_to_openai_shape() {
    let p = BedrockProvider;
    let body = json!({
    "generation": "Hello from Llama",
    "prompt_token_count": 7,
    "generation_token_count": 4,
    });
    let out = p
        .transform_response(&body, "meta.llama3-1-70b-instruct-v1:0")
        .expect("ok");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(out["model"], "meta.llama3-1-70b-instruct-v1:0");
    assert!(out.get("created").is_some());
    assert_eq!(out["choices"][0]["message"]["role"], "assistant");
    assert_eq!(out["choices"][0]["message"]["content"], "Hello from Llama");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
    assert_eq!(out["usage"]["prompt_tokens"], 7);
    assert_eq!(out["usage"]["completion_tokens"], 4);
    assert_eq!(out["usage"]["total_tokens"], 11);
}

#[test]
fn transform_response_mistral_maps_outputs_array_to_openai_shape() {
    let p = BedrockProvider;
    let body = json!({
    "outputs": [{
    "text": "Bonjour de Mistral",
    "stop_reason": "length",
    }],
    });
    let out = p
        .transform_response(&body, "mistral.mistral-7b-instruct-v0:2")
        .expect("ok");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(
        out["choices"][0]["message"]["content"],
        "Bonjour de Mistral"
    );
    // Mistral `length` maps onto OpenAI `length`.
    assert_eq!(out["choices"][0]["finish_reason"], "length");
}

#[test]
fn transform_response_titan_maps_results_to_openai_shape() {
    let p = BedrockProvider;
    let body = json!({
    "results": [{
    "outputText": "Hello from Titan",
    "completionReason": "FINISHED",
    }],
    });
    let out = p
        .transform_response(&body, "amazon.titan-text-express-v1")
        .expect("ok");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(out["choices"][0]["message"]["content"], "Hello from Titan");
    // Unknown completion reasons default to "stop".
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
}

#[test]
fn transform_stream_chunk_meta_llama_outputs_array_yields_openai_delta() {
    let p = BedrockProvider;
    let chunk = r#"{"outputs":[{"text":"partial"}]}"#;
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["object"], "chat.completion.chunk");
    assert_eq!(out["choices"][0]["delta"]["content"], "partial");
}

#[test]
fn transform_stream_chunk_unknown_vendor_text_field_yields_delta() {
    let p = BedrockProvider;
    let chunk = r#"{"generation":"some token"}"#;
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "some token");
}

#[test]
fn transform_stream_chunk_empty_text_yields_none() {
    let p = BedrockProvider;
    let chunk = r#"{"outputs":[{"text":""}]}"#;
    assert!(p.transform_stream_chunk(chunk).expect("ok").is_none());
}

#[test]
fn transform_stream_chunk_passes_anthropic_events_through_anthropic_transform() {
    let p = BedrockProvider;
    // Bedrock eventstream frames arrive as bare JSON objects with `type`.
    let chunk =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
    // Guard against the dispatch regression: a snake_case Anthropic `type`
    // must NOT be claimed by the Converse path (both paths emit identical
    // `{content: "Hi"}` for text deltas, so checking only the output would
    // mask a misroute). Asserting the dispatch decision directly.
    let value: Value = serde_json::from_str(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
    )
    .unwrap();
    assert!(
        !converse::looks_like_converse_event(&value),
        "Anthropic snake_case event must not be classified as Converse"
    );
    let out = p.transform_stream_chunk(chunk).expect("ok").expect("some");
    assert_eq!(out["object"], "chat.completion.chunk");
    assert_eq!(out["choices"][0]["delta"]["content"], "Hi");
}

/// Regression: an Anthropic `message_delta` event carries both a `delta`
/// and a `usage` field, which the old Converse heuristic keyed on — so it
/// was misclassified as a Converse `metadata` event. That silently dropped
/// the finish_reason and zeroed usage. After the snake_case-`type` guard,
/// the event must route through the Anthropic path, preserving
/// finish_reason and surfacing usage.
#[test]
fn transform_stream_chunk_anthropic_message_delta_preserves_finish_reason_and_usage() {
    let p = BedrockProvider;
    let chunk = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":20}}"#;
    // Dispatch guard: this event must NOT be claimed as Converse.
    let value: Value = serde_json::from_str(chunk).unwrap();
    assert!(
        !converse::looks_like_converse_event(&value),
        "Anthropic message_delta must not be classified as Converse metadata"
    );
    let out = p.transform_stream_chunk(chunk).expect("ok").expect("some");
    // finish_reason must be present and mapped (end_turn → stop), NOT null
    // or absent (which is what the Converse metadata path would produce).
    let finish = &out["choices"][0]["finish_reason"];
    assert!(
        finish.is_string(),
        "finish_reason must be preserved (got {finish})"
    );
    assert_eq!(finish, "stop");
    // usage must be surfaced — Anthropic maps output_tokens → completion_tokens.
    assert_eq!(out["usage"]["completion_tokens"], 20);
}

#[test]
fn transform_stream_chunk_extracts_data_line_when_present() {
    let p = BedrockProvider;
    let chunk = "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}";
    let out = p.transform_stream_chunk(chunk).expect("ok").expect("some");
    assert_eq!(out["choices"][0]["delta"]["content"], "Hi");
}

#[test]
fn transform_stream_chunk_skips_empty_and_done() {
    let p = BedrockProvider;
    assert!(p.transform_stream_chunk("").expect("ok").is_none());
    assert!(p.transform_stream_chunk("[DONE]").expect("ok").is_none());
}

#[test]
fn transform_stream_chunk_skips_unparseable_text() {
    let p = BedrockProvider;
    assert!(p.transform_stream_chunk("not json").expect("ok").is_none());
}

#[test]
fn apply_auth_is_a_no_op() {
    // SigV4 signing happens in the host after the body is serialized. The
    // provider `apply_auth` must leave the header map untouched so the host's
    // signing pass owns every canonical header.
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    let creds = ResolvedCredentials::AwsSigV4 {
        access_key: "AK".to_string(),
        secret_key: "sk".to_string(),
        session_token: None,
        region: "us-east-1".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    headers.insert("x-other".to_string(), "v".to_string());
    p.apply_auth(&dep, &creds, &mut headers);
    assert_eq!(headers.len(), 1);
    assert_eq!(headers.get("x-other").map(String::as_str), Some("v"));
}

#[test]
fn provider_id_and_supports() {
    let p = BedrockProvider;
    assert_eq!(p.id(), "bedrock");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.embeddings().is_some());
}

#[test]
fn wildcard_resolution_reads_model_from_request() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/*");
    let req = json!({
    "model": "bedrock/anthropic.claude-3-opus",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["anthropic_version"], ANTHROPIC_VERSION);
    assert!(out.get("model").is_none());
    assert!(
        p.endpoint_url_for_request(&dep, &req)
            .ends_with("/model/anthropic.claude-3-opus/invoke")
    );
}

#[test]
fn wildcard_converse_url_and_body_use_the_same_resolved_model() {
    let dep = bedrock_deployment("bedrock/*");
    let req = json!({"model":"bedrock/openai.gpt-6.1-sol","messages":[{"role":"user","content":"hi"}],"max_tokens":16});
    let body = BedrockProvider.transform_request(&req, &dep).unwrap();
    assert!(body.get("model").is_none());
    assert!(body.get("inferenceConfig").is_some());
    assert!(
        BedrockProvider
            .endpoint_url_for_request(&dep, &req)
            .ends_with("/model/openai.gpt-6.1-sol/converse")
    );
}

#[test]
fn converse_model_specific_parameters_and_tool_names_follow_the_wire_schema() {
    let first = "a".repeat(65);
    let second = format!("{}b", "a".repeat(64));
    let req = json!({"messages":[{"role":"user","content":"hi"}],"top_k":20,"tools":[
        {"type":"function","function":{"name":first,"parameters":{"type":"object"}}},
        {"type":"function","function":{"name":second,"parameters":{"type":"object"}}}
    ]});
    let body = converse::transform_request_converse(
        &req,
        "anthropic.claude-sonnet-4-6",
        &serde_json::Map::new(),
    )
    .unwrap();
    assert!(body["inferenceConfig"].get("topK").is_none());
    assert_eq!(body["additionalModelRequestFields"]["top_k"], 20);
    let names = body["toolConfig"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["toolSpec"]["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(names.iter().all(|name| name.len() <= 64));
    assert_ne!(names[0], names[1]);
}

#[test]
fn bare_wildcard_also_works() {
    // A bare `*` wildcard resolves the model from the request body. Using
    // an unknown-vendor model keeps it on the passthrough fallback so the
    // `model` body field is preserved.
    let p = BedrockProvider;
    let dep = bedrock_deployment("*");
    let req = json!({
    "model": "bedrock/custom.unknown-v1:0",
    "messages": [],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "custom.unknown-v1:0");
    assert!(out.get("anthropic_version").is_none());
}

fn make_eventstream_frame(headers: &[u8], payload: &[u8]) -> Vec<u8> {
    let total_len = 12 + headers.len() + payload.len() + 4;
    let mut buf = Vec::with_capacity(total_len);
    buf.extend_from_slice(&(total_len as u32).to_be_bytes());
    buf.extend_from_slice(&(headers.len() as u32).to_be_bytes());
    let prelude_crc = super::crc32(&buf);
    buf.extend_from_slice(&prelude_crc.to_be_bytes());
    buf.extend_from_slice(headers);
    buf.extend_from_slice(payload);
    // Write a valid CRC32 over total_len..=payload (mirrors the encoder) so
    // the decoder's integrity check accepts the frame.
    let crc = super::crc32(&buf);
    buf.extend_from_slice(&crc.to_be_bytes());
    buf
}

#[test]
fn decode_eventstream_frame_extracts_payload() {
    let payload = br#"{"type":"message_start"}"#;
    let frame = make_eventstream_frame(&[], payload);
    let (consumed, extracted) =
        decode_eventstream_frame(&frame).expect("frame should be decodable");
    assert_eq!(consumed, frame.len());
    assert_eq!(extracted, payload);
}

#[test]
fn decode_eventstream_frame_returns_none_for_partial_buffer() {
    let payload = br#"{"type":"message_start"}"#;
    let frame = make_eventstream_frame(&[], payload);
    let (consumed, _) = decode_eventstream_frame(&frame).expect("full frame");
    assert_eq!(consumed, frame.len());
    assert!(decode_eventstream_frame(&frame[..frame.len() - 1]).is_none());
    assert!(decode_eventstream_frame(&[]).is_none());
}

#[test]
fn decode_eventstream_frame_skips_headers_to_reach_payload() {
    let header_bytes: Vec<u8> = {
        let name = b":content-type";
        let value = b"application/json";
        let mut h = Vec::new();
        h.push(name.len() as u8);
        h.extend_from_slice(name);
        h.push(7); // eventstream string header
        h.extend_from_slice(&(value.len() as u16).to_be_bytes());
        h.extend_from_slice(value);
        h
    };
    let payload = br#"{"ok":true}"#;
    let frame = make_eventstream_frame(&header_bytes, payload);
    let (consumed, extracted) =
        decode_eventstream_frame(&frame).expect("frame with headers should decode");
    assert_eq!(consumed, frame.len());
    assert_eq!(extracted, payload);
}

#[test]
fn decode_eventstream_frame_only_consumes_first_frame() {
    let payload1 = br#"{"a":1}"#;
    let payload2 = br#"{"b":2}"#;
    let frame1 = make_eventstream_frame(&[], payload1);
    let frame2 = make_eventstream_frame(&[], payload2);
    let mut combined = frame1.clone();
    combined.extend_from_slice(&frame2);
    let (consumed, extracted) =
        decode_eventstream_frame(&combined).expect("first frame should decode");
    assert_eq!(consumed, frame1.len());
    assert_eq!(extracted, payload1);
    let (consumed2, extracted2) =
        decode_eventstream_frame(&combined[consumed..]).expect("second frame should decode");
    assert_eq!(consumed2, frame2.len());
    assert_eq!(extracted2, payload2);
}

// =============================================================================
// encode_eventstream_frame — the inverse of decode_eventstream_frame.
// =============================================================================

#[test]
fn encode_eventstream_frame_round_trips_through_decode() {
    let payload = br#"{"event":{"sessionStart":{"inferenceConfiguration":{"maxTokens":1024}}}}"#;
    let frame = encode_eventstream_frame(payload);
    let (consumed, extracted) =
        decode_eventstream_frame(&frame).expect("encoded frame must decode");
    assert_eq!(consumed, frame.len());
    assert_eq!(extracted, payload);
}

#[test]
fn encode_eventstream_frame_empty_payload_round_trips() {
    let frame = encode_eventstream_frame(&[]);
    let (consumed, extracted) = decode_eventstream_frame(&frame).expect("decode");
    assert_eq!(consumed, frame.len());
    assert!(extracted.is_empty());
}

#[test]
fn encode_eventstream_frame_has_correct_length_fields() {
    let payload = b"hello";
    let frame = encode_eventstream_frame(payload);
    // total_len = 12 (prelude including CRC) + 5 (payload) + 4 (message CRC).
    assert_eq!(frame.len(), 21);
    let total_len = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]);
    assert_eq!(total_len as usize, frame.len());
    let headers_len = u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]);
    assert_eq!(headers_len, 0, "realtime frames carry no headers");
    assert_eq!(&frame[8..12], &0xcd22c79bu32.to_be_bytes());
    assert_eq!(&frame[12..frame.len() - 4], payload);
}

#[test]
fn encode_eventstream_frame_crc_matches_independent_computation() {
    // CRC32 of the prefix+payload region must equal the trailing 4 bytes.
    // Reference value computed with Python's zlib.crc32 over
    // total_len(21) ++ headers_len(0) ++ prelude_crc ++ "hello".
    let payload = b"hello";
    let frame = encode_eventstream_frame(payload);
    let expected: u32 = 0x570c3496;
    let got = u32::from_be_bytes([
        frame[frame.len() - 4],
        frame[frame.len() - 3],
        frame[frame.len() - 2],
        frame[frame.len() - 1],
    ]);
    assert_eq!(
        got, expected,
        "CRC32 must match zlib over total_len..=payload"
    );
}

#[test]
fn encode_eventstream_frame_multiple_frames_decode_sequentially() {
    let payloads: &[&[u8]] = &[b"{\"a\":1}", b"{\"b\":2}", b"{\"c\":3}"];
    let mut buf = Vec::new();
    for p in payloads {
        buf.extend_from_slice(&encode_eventstream_frame(p));
    }
    let mut offset = 0;
    for expected in payloads {
        let (consumed, extracted) =
            decode_eventstream_frame(&buf[offset..]).expect("each frame decodes");
        assert_eq!(extracted, *expected);
        offset += consumed;
    }
    assert_eq!(offset, buf.len(), "all bytes consumed");
}

// =============================================================================
// decode_eventstream_frame — CRC32 integrity validation (MEDIUM #5).
// =============================================================================

#[test]
fn decode_eventstream_frame_rejects_corrupted_payload() {
    // Flip one payload byte of a validly-encoded frame; the trailing CRC32
    // no longer matches. A complete corrupt frame must NOT be handed to the
    // transform layer (no corrupt payload), and must NOT stall the stream
    // by returning None forever — the decoder advances past it, returning an
    // empty payload so the caller drops the frame and continues.
    let payload = b"{\"event\":\"hello\"}";
    let mut frame = encode_eventstream_frame(payload);
    // Flip a byte inside the payload region (offset 12 is the first payload
    // byte here since headers_len == 0).
    frame[12] ^= 0xFF;
    let (consumed, extracted) = decode_eventstream_frame(&frame)
        .expect("a complete corrupt frame must advance the cursor, not stall");
    assert_eq!(
        consumed,
        frame.len(),
        "must consume the whole corrupt frame"
    );
    assert!(
        extracted.is_empty(),
        "must NOT hand a corrupt payload to the transform layer"
    );
}

#[test]
fn decode_eventstream_frame_rejects_corrupted_crc() {
    let payload = b"payload-with-crc-check";
    let mut frame = encode_eventstream_frame(payload);
    // Corrupt the trailing CRC byte itself.
    let last = frame.len() - 1;
    frame[last] ^= 0xFF;
    let (consumed, extracted) = decode_eventstream_frame(&frame)
        .expect("a complete corrupt frame must advance the cursor, not stall");
    assert_eq!(consumed, frame.len());
    assert!(
        extracted.is_empty(),
        "corrupt frame must yield an empty payload"
    );
}

/// The 12-byte prelude's last 4 bytes are a CRC32 over the first 8 bytes
/// (`total_len` + `headers_len`). Decode must validate that checksum *before*
/// trusting the length fields: a frame whose prelude CRC is wrong has no
/// reliable boundaries, so the decoder fails closed with `None` (caller keeps
/// buffering) rather than decoding its payload. The trailing message CRC is
/// recomputed so the *only* invalid bytes are the prelude CRC — if the prelude
/// check were dropped, this frame would decode as valid, which is exactly what
/// this test discriminates.
#[test]
fn decode_eventstream_frame_rejects_corrupted_prelude_crc() {
    let payload = b"{\"event\":\"prelude-crc\"}";
    let mut frame = encode_eventstream_frame(payload);
    frame[8] ^= 0xFF; // first byte of the prelude CRC
    let end = frame.len() - 4;
    let crc = super::crc32(&frame[..end]);
    frame[end..].copy_from_slice(&crc.to_be_bytes());
    assert!(
        decode_eventstream_frame(&frame).is_none(),
        "a corrupt prelude CRC must fail closed before total_len is trusted"
    );
}

/// A corrupt complete frame followed by a valid frame must decode the second
/// frame correctly — the decoder advances past the bad frame instead of
/// retrying it forever (which would stall the stream).
#[test]
fn decode_eventstream_frame_advances_past_corrupt_to_next_frame() {
    let bad_payload = b"corrupt";
    let good_payload = b"{\"event\":\"ok\"}";
    let mut bad = encode_eventstream_frame(bad_payload);
    bad[12] ^= 0xFF; // corrupt the payload region (headers_len == 0)
    let good = encode_eventstream_frame(good_payload);
    let mut buf = bad.clone();
    buf.extend_from_slice(&good);

    // First call: skips the corrupt frame (empty payload, advances cursor).
    let (consumed1, extracted1) =
        decode_eventstream_frame(&buf).expect("must advance past the corrupt frame");
    assert_eq!(consumed1, bad.len());
    assert!(extracted1.is_empty());

    // Second call on the remainder: decodes the good frame.
    let (consumed2, extracted2) =
        decode_eventstream_frame(&buf[consumed1..]).expect("good frame must decode");
    assert_eq!(consumed2, good.len());
    assert_eq!(extracted2, good_payload);
}

#[test]
fn decode_eventstream_frame_accepts_intact_frame_after_crc_check() {
    // Sanity: the CRC check must not reject a well-formed encoded frame.
    let payload = b"crc-round-trip";
    let frame = encode_eventstream_frame(payload);
    let (consumed, extracted) = decode_eventstream_frame(&frame).expect("intact frame must decode");
    assert_eq!(consumed, frame.len());
    assert_eq!(extracted, payload);
}

// --- Ported twin: model override for anthropic models ---
//
// Ported from the source `twin_bedrock_provider_success` — the provider
// must use the deployment's configured model in the upstream URL. The
// Anthropic Messages body itself carries no `model` field (Bedrock infers
// it from the URL path).

#[test]
fn transform_request_overrides_request_model_with_deployment_model() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0");
    let req = json!({
        "model": "caller-supplied-ignored",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert!(
        out.get("model").is_none(),
        "model must be dropped from the Anthropic body"
    );
    assert_eq!(out["anthropic_version"], ANTHROPIC_VERSION);
}

// --- Ported twin: non-anthropic model override ---
//
// For non-Anthropic Bedrock models (meta, mistral, etc.), the provider
// must still override the request model with the deployment's configured
// model.

#[test]
fn endpoint_url_with_cross_region_prefix() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke"
    );
    let dep = bedrock_deployment("bedrock/apac.anthropic.claude-haiku-4-5-20251001-v1:0");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/apac.anthropic.claude-haiku-4-5-20251001-v1:0/invoke"
    );
}

#[test]
fn geo_specific_cris_prefix_is_stripped_for_anthropic_detect() {
    assert_eq!(
        super::strip_cross_region_prefix("us-east-1.anthropic.claude-3-5-sonnet"),
        "anthropic.claude-3-5-sonnet"
    );
    assert_eq!(
        super::strip_cross_region_prefix("global.anthropic.claude-sonnet-4"),
        "anthropic.claude-sonnet-4"
    );
    assert_eq!(
        super::strip_cross_region_prefix("us.anthropic.claude-haiku"),
        "anthropic.claude-haiku"
    );
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/us-east-1.anthropic.claude-3-5-sonnet");
    let req = json!({
        "model": "us-east-1.anthropic.claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 16,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(
        out["anthropic_version"], ANTHROPIC_VERSION,
        "geo-specific CRIS Anthropic ids must take the native Anthropic path"
    );
}

#[test]
fn transform_request_for_anthropic_strips_unknown_params() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    let req = json!({
    "model": "anthropic.claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    "temperature": 0.5,
    "unknown_param": "should_be_stripped",
    });
    let out = p.transform_request(&req, &dep).expect("transform ok");
    assert_eq!(out["anthropic_version"], ANTHROPIC_VERSION);
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_tokens"], 100);
    assert!(
        out.get("unknown_param").is_none(),
        "unknown params must be stripped for anthropic models"
    );
}

#[test]
fn transform_request_non_anthropic_passthrough_preserves_guardrail_config() {
    // Unknown-vendor passthrough preserves caller-supplied body fields like
    // guardrailConfig verbatim. (Known vendors build a fresh body and do
    // not echo client guardrail config — Bedrock reads it from URL params.)
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/custom.unknown-model-v1:0");
    let req = json!({
    "model": "custom.unknown-model-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    "guardrailConfig": {
    "guardrailIdentifier": "ff6ujrregl1q",
    "guardrailVersion": "DRAFT",
    "trace": "enabled",
    },
    });
    let out = p.transform_request(&req, &dep).expect("transform ok");
    assert_eq!(
        out["guardrailConfig"]["guardrailIdentifier"],
        "ff6ujrregl1q"
    );
    assert_eq!(out["guardrailConfig"]["guardrailVersion"], "DRAFT");
    assert_eq!(out["guardrailConfig"]["trace"], "enabled");
    assert!(out.get("anthropic_version").is_none());
}

#[test]
fn transform_response_with_citations_content_blocks() {
    let p = BedrockProvider;
    let body = json!({
    "id": "msg_abc",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet",
    "content": [
    {"type": "text", "text": "According to sources, "},
    {"type": "text", "text": "the answer is 42.", "citations": [
    {"start_index": 0, "end_index": 10, "cited_text": "sources"},
    ]},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
    });
    let out = p
        .transform_response(&body, "anthropic.claude-3-5-sonnet")
        .expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["content"],
        "According to sources, the answer is 42."
    );
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
    assert_eq!(out["usage"]["prompt_tokens"], 10);
    assert_eq!(out["usage"]["completion_tokens"], 5);
}

#[test]
fn vendor_finish_reason_mappings() {
    let p = BedrockProvider;
    for (vendor_reason, expected) in [
        ("end_turn", "stop"),
        ("stop", "stop"),
        ("STOP", "stop"),
        ("max_tokens", "length"),
        ("length", "length"),
        ("MAX_TOKENS", "length"),
        ("tool_calls", "tool_calls"),
        ("TOOL_CALLS", "tool_calls"),
        ("content_filter", "content_filter"),
        ("FILTERED", "content_filter"),
        ("unknown_reason", "stop"),
    ] {
        let body = json!({
        "generation": "Hello",
        "stop_reason": vendor_reason,
        });
        let out = p
            .transform_response(&body, "meta.llama3-1-70b")
            .expect("ok");
        assert_eq!(
            out["choices"][0]["finish_reason"], expected,
            "for vendor_reason={vendor_reason}"
        );
    }
}

#[test]
fn extract_vendor_text_recognises_output_text_field() {
    let p = BedrockProvider;
    let body = json!({"outputText": "Hello from a custom vendor"});
    let out = p.transform_response(&body, "custom-model").expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["content"],
        "Hello from a custom vendor"
    );
}

#[test]
fn transform_request_overrides_model_for_non_anthropic_models() {
    // Unknown-vendor passthrough still overrides the request model with the
    // deployment's configured model (with `bedrock/` stripped).
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/custom.unknown-model-v1:0");
    let req = json!({
        "model": "caller-supplied-ignored",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "custom.unknown-model-v1:0");
    assert!(out.get("anthropic_version").is_none());
}

// =============================================================================
// Ported from the reference `tests/llm_translation/test_bedrock_nova_embedding.py`.
//
// oxllm's Bedrock provider does not yet implement the Amazon Nova multimodal
// embedding transform (`AmazonNovaEmbeddingConfig` in the reference). The genuinely
// portable parts of that suite are the *provider detection* tests
// (`TestNovaProviderDetection`): model names beginning with `amazon.` (incl.
// Nova) are non-Anthropic, so they route through the Bedrock passthrough path
// rather than the Anthropic Messages transform. The request/response/E2E tests
// that exercise the Nova-specific config have no oxllm equivalent yet and are
// omitted.
// =============================================================================

/// Ported from `test_nova_provider_detection` / `test_nova_in_model_name`:
/// an `amazon.nova-*` model id is non-Anthropic AND auto-routes to Converse
/// (Nova's InvokeModel body is its own messages-v1 schema, not OpenAI-shaped)
/// so the URL keeps the full model id on the Converse path.
#[test]
fn endpoint_url_for_amazon_nova_model() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/amazon.nova-2-multimodal-embeddings-v1:0");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-2-multimodal-embeddings-v1:0/converse"
    );
}

/// Ported from `test_nova_in_model_name`: cross-region Nova prefixes
/// (`us.amazon.nova-*`) also resolve to a well-formed Converse URL.
#[test]
fn endpoint_url_for_cross_region_nova_model() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/us.amazon.nova-2-multimodal-embeddings-v1:0");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.amazon.nova-2-multimodal-embeddings-v1:0/converse"
    );
}

/// Ported from `test_nova_provider_detection`: Nova models are detected as
/// non-Anthropic and auto-route to the Converse transform — the body is the
/// unified Converse shape (no `anthropic_version`, params under
/// `inferenceConfig`), never the Anthropic Messages transform.
#[test]
fn transform_request_nova_model_is_non_anthropic_passthrough() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/amazon.nova-2-multimodal-embeddings-v1:0");
    let req = json!({
    "model": "amazon.nova-2-multimodal-embeddings-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.5,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert!(
        out.get("anthropic_version").is_none(),
        "Nova is non-Anthropic; anthropic_version must not be inlined"
    );
    // Converse body: OpenAI params map into inferenceConfig.
    assert_eq!(out["inferenceConfig"]["temperature"], 0.5);
    assert!(out.get("temperature").is_none());
}

/// A Nova model id routes through the Converse transform (no legacy
/// top-level model passthrough to strip).
#[test]
fn transform_request_strips_bedrock_prefix_for_nova_model() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/amazon.nova-2-multimodal-embeddings-v1:0");
    let req = json!({
    "model": "bedrock/amazon.nova-2-multimodal-embeddings-v1:0",
    "messages": [],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_ne!(
        out.get("model").and_then(|v| v.as_str()),
        Some("bedrock/amazon.nova-2-multimodal-embeddings-v1:0"),
        "the bedrock/ routing prefix must not leak into the Converse body"
    );
}

/// A wildcard deployment (`bedrock/*`) resolves the actual model from the
/// request body, including Nova model ids, and routes them through the
/// Converse transform as non-Anthropic.
#[test]
fn nova_model_resolves_through_wildcard_deployment() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/*");
    let req = json!({
    "model": "bedrock/amazon.nova-lite-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert!(out.get("anthropic_version").is_none());
    assert!(
        out.get("messages").is_some(),
        "Converse body carries messages"
    );
}

/// Ported from `test_nova_provider_detection`: a Nova model's response uses
/// the vendor transform path. An unrecognised vendor shape (Nova embedding
/// bodies are `{embeddings: [...]}`) passes through unchanged rather than
/// being mangled into a chat completion.
#[test]
fn nova_response_passthrough_for_unrecognised_vendor_shape() {
    let p = BedrockProvider;
    // Nova embedding-style body — no text field the vendor transform recognises.
    let body = json!({
    "embeddingType": "TEXT",
    "embeddings": [{"embedding": [0.1, 0.2, 0.3]}],
    });
    let out = p
        .transform_response(&body, "amazon.nova-2-multimodal-embeddings-v1:0")
        .expect("ok");
    assert_eq!(
        out, body,
        "unrecognised vendor shape passes through verbatim"
    );
}

/// Cohere and Titan embed ids are translated. Other Bedrock models, including
/// Nova, are rejected in the embed transform so they are not forwarded as an
/// OpenAI body.
#[test]
fn bedrock_embeddings_reject_non_embed_models() {
    let p = BedrockProvider;
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.embeddings().is_some());
    let dep = bedrock_deployment("bedrock/amazon.nova-micro-v1:0");
    let err = oxllm_core::EmbeddingProvider::transform_request(
        p.embeddings().expect("embeddings"),
        &json!({"input": "hello"}),
        &dep,
    )
    .expect_err("nova is not an embed model");
    assert!(err.to_string().contains("does not support embeddings"));
}

// =============================================================================
// Feature: thinking blocks for Anthropic-on-Bedrock.
//
// Bedrock-hosted Claude reasoning models return the standard Anthropic
// Messages response shape, including `thinking` content blocks. The base
// Anthropic transform discards them; the Bedrock layer re-extracts them as
// `reasoning_content` (concatenated text) + `thinking_blocks` (raw blocks).
// =============================================================================

#[test]
fn transform_response_extracts_thinking_blocks_for_anthropic_on_bedrock() {
    let p = BedrockProvider;
    let body = json!({
    "id": "msg_think",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-7-sonnet",
    "content": [
    {"type": "thinking", "thinking": "Considering the question.", "signature": "sig_001"},
    {"type": "thinking", "thinking": " Paris is the capital.", "signature": "sig_002"},
    {"type": "text", "text": "The capital of France is Paris."},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 20},
    });
    let out = p
        .transform_response(&body, "anthropic.claude-3-7-sonnet")
        .expect("ok");
    let message = &out["choices"][0]["message"];
    assert_eq!(message["content"], "The capital of France is Paris.");
    assert_eq!(
        message["reasoning_content"],
        "Considering the question. Paris is the capital."
    );
    let blocks = message["thinking_blocks"].as_array().expect("blocks");
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["signature"], "sig_001");
    assert_eq!(blocks[1]["signature"], "sig_002");
}

#[test]
fn transform_response_redacted_thinking_block_preserved_without_reasoning_text() {
    let p = BedrockProvider;
    let body = json!({
    "id": "msg_rt",
    "content": [
    {"type": "redacted_thinking", "data": "opaque-bytes"},
    {"type": "text", "text": "Answer."},
    ],
    "stop_reason": "end_turn",
    });
    let out = p
        .transform_response(&body, "anthropic.claude-3-7-sonnet")
        .expect("ok");
    let message = &out["choices"][0]["message"];
    // No visible thinking text — reasoning_content must be absent.
    assert!(message.get("reasoning_content").is_none());
    let blocks = message["thinking_blocks"].as_array().expect("blocks");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["type"], "redacted_thinking");
}

#[test]
fn transform_response_without_thinking_blocks_has_no_reasoning_fields() {
    let p = BedrockProvider;
    let body = json!({
    "id": "msg_plain",
    "content": [{"type": "text", "text": "Bonjour!"}],
    "stop_reason": "end_turn",
    });
    let out = p
        .transform_response(&body, "anthropic.claude-3-5-sonnet")
        .expect("ok");
    let message = &out["choices"][0]["message"];
    assert!(message.get("reasoning_content").is_none());
    assert!(message.get("thinking_blocks").is_none());
}

// =============================================================================
// Feature: guardrail config passthrough.
//
// AWS Bedrock applies inline guardrails via URL query parameters on the
// InvokeModel endpoint (`?guardrailIdentifier=...&guardrailVersion=...`).
// Configured via `extra.guardrail_id`/`guardrail_version` (snake_case) or
// the AWS-native camelCase variants.
// =============================================================================

fn bedrock_deployment_with_extra(model: &str, extra: &[(&str, serde_json::Value)]) -> Deployment {
    let mut dep = bedrock_deployment(model);
    for (k, v) in extra {
        dep.oxllm_params.extra.insert((*k).to_string(), v.clone());
    }
    dep
}

#[test]
fn endpoint_url_appends_guardrail_query_params_snake_case() {
    let p = BedrockProvider;
    let dep = bedrock_deployment_with_extra(
        "bedrock/anthropic.claude-3-5-sonnet",
        &[
            ("guardrail_id", json!("ff6ujrregl1q")),
            ("guardrail_version", json!("DRAFT")),
        ],
    );
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/invoke\
         ?guardrailIdentifier=ff6ujrregl1q&guardrailVersion=DRAFT"
    );
}

#[test]
fn endpoint_url_appends_guardrail_query_params_camel_case() {
    let p = BedrockProvider;
    let dep = bedrock_deployment_with_extra(
        "bedrock/anthropic.claude-3-5-sonnet",
        &[
            ("guardrailIdentifier", json!("gr-123")),
            ("guardrailVersion", json!("1")),
        ],
    );
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/invoke\
         ?guardrailIdentifier=gr-123&guardrailVersion=1"
    );
}

#[test]
fn endpoint_url_includes_trace_when_enabled() {
    let p = BedrockProvider;
    let dep = bedrock_deployment_with_extra(
        "bedrock/anthropic.claude-3-5-sonnet",
        &[
            ("guardrail_id", json!("ff6ujrregl1q")),
            ("guardrail_version", json!("DRAFT")),
            ("trace", json!(true)),
        ],
    );
    let url = p.endpoint_url(&dep);
    assert!(
        url.contains("&trace=enabled"),
        "trace flag must produce &trace=enabled query param, got: {url}"
    );
}

#[test]
fn endpoint_url_omits_trace_when_disabled_or_absent() {
    let p = BedrockProvider;
    let dep = bedrock_deployment_with_extra(
        "bedrock/anthropic.claude-3-5-sonnet",
        &[
            ("guardrail_id", json!("ff6ujrregl1q")),
            ("guardrail_version", json!("DRAFT")),
            ("trace", json!(false)),
        ],
    );
    let url = p.endpoint_url(&dep);
    assert!(!url.contains("trace="), "no trace param when false: {url}");
}

#[test]
fn endpoint_url_clean_when_guardrail_unconfigured() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/invoke"
    );
}

#[test]
fn endpoint_url_no_query_when_only_one_guardrail_field_set() {
    let p = BedrockProvider;
    let dep = bedrock_deployment_with_extra(
        "bedrock/anthropic.claude-3-5-sonnet",
        &[("guardrail_id", json!("orphan"))],
    );
    let url = p.endpoint_url(&dep);
    assert!(
        !url.contains('?'),
        "guardrail requires both id and version; got: {url}"
    );
}

/// Converse deployments put guardrails in the body (`guardrailConfig`),
/// NOT in URL query params. Appending query params would break the host's
/// `/converse` → `/converseStream` streaming rewrite AND be rejected by
/// Bedrock.
#[test]
fn endpoint_url_converse_omits_guardrail_query_params() {
    let p = BedrockProvider;
    let dep = bedrock_deployment_with_extra(
        "bedrock/anthropic.claude-3-5-sonnet",
        &[
            ("use_converse_api", json!(true)),
            ("guardrail_id", json!("ff6ujrregl1q")),
            ("guardrail_version", json!("DRAFT")),
            ("trace", json!(true)),
        ],
    );
    let url = p.endpoint_url(&dep);
    assert!(
        !url.contains('?'),
        "Converse must not append guardrail query params (they go in the body): {url}"
    );
    assert!(url.ends_with("/converse"));
}

// =============================================================================
// Feature: multimodal image support for Anthropic-on-Bedrock.
//
// Bedrock-hosted Claude expects the native Anthropic image shape
// (`{"type":"image","source":{"type":"base64","media_type":...,"data":...}}`).
// OpenAI clients send `image_url` blocks with `data:` URLs; the Bedrock
// transform rewrites those before the body goes upstream.
// =============================================================================

#[test]
fn transform_request_converts_data_url_image_to_anthropic_source_block() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    let req = json!({
    "model": "anthropic.claude-3-5-sonnet",
    "messages": [{
    "role": "user",
    "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ],
    }],
    "max_tokens": 100,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    let content = &out["messages"][0]["content"];
    let blocks = content.as_array().expect("content array");
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[1]["type"], "image");
    assert_eq!(blocks[1]["source"]["type"], "base64");
    assert_eq!(blocks[1]["source"]["media_type"], "image/png");
    assert_eq!(blocks[1]["source"]["data"], "iVBORw0KGgo=");
    // Original OpenAI shape is fully replaced.
    assert!(blocks[1].get("image_url").is_none());
}

#[test]
fn transform_request_leaves_http_image_url_untouched() {
    // Bedrock cannot fetch remote URLs; we leave them in place for the host
    // pipeline to inline (or reject) rather than silently mangling them.
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    let req = json!({
    "model": "anthropic.claude-3-5-sonnet",
    "messages": [{
    "role": "user",
    "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ],
    }],
    "max_tokens": 100,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    let block = &out["messages"][0]["content"][0];
    assert_eq!(block["type"], "image_url");
    assert_eq!(block["image_url"]["url"], "https://example.com/cat.png");
}

#[test]
fn transform_request_text_only_body_is_unaffected_by_image_conversion() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
    let req = json!({
    "model": "anthropic.claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "plain text"}],
    "max_tokens": 100,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["messages"][0]["content"], "plain text");
}

// =============================================================================
// Feature: performance configuration passthrough.
//
// Bedrock InvokeModel accepts a `performanceConfig: {latency: "optimized"}`
// body hint on supported model families. Configured via
// `extra.performance_config` or `extra.performanceConfig` and injected into
// the body for non-Anthropic models (Anthropic-on-Bedrock uses its own
// body shape where this field is not applicable).
// =============================================================================

#[test]
fn transform_request_injects_performance_config_snake_case_for_vendor_models() {
    let p = BedrockProvider;
    let dep = bedrock_deployment_with_extra(
        "bedrock/meta.llama3-1-70b-instruct-v1:0",
        &[("performance_config", json!({"latency": "optimized"}))],
    );
    let req = json!({
    "model": "meta.llama3-1-70b-instruct-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["performanceConfig"]["latency"], "optimized");
}

#[test]
fn transform_request_injects_performance_config_camel_case_for_vendor_models() {
    let p = BedrockProvider;
    let dep = bedrock_deployment_with_extra(
        "bedrock/meta.llama3-1-70b-instruct-v1:0",
        &[("performanceConfig", json!({"latency": "standard"}))],
    );
    let req = json!({
    "model": "meta.llama3-1-70b-instruct-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["performanceConfig"]["latency"], "standard");
}

#[test]
fn transform_request_omits_performance_config_when_unconfigured() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/meta.llama3-1-70b-instruct-v1:0");
    let req = json!({
    "model": "meta.llama3-1-70b-instruct-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert!(out.get("performanceConfig").is_none());
}

// =============================================================================
// Feature: Llama-3 chat template for InvokeModel.
//
// Meta Llama on Bedrock takes a single `prompt` field that must follow the
// official Llama-3 special-token layout. Plain concatenation (the old
// behaviour) made the model echo its input; `messages_to_llama3_prompt`
// builds the proper `<|begin_of_text|>...<|eot_id|>` envelope. These tests
// pin the template shape end-to-end through `transform_request`.
// =============================================================================

/// The Llama-3 prompt must open with `<|begin_of_text|>` — without it the
/// model has no BOS anchor and may misbehave on the first turn.
#[test]
fn llama3_prompt_starts_with_begin_of_text() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/meta.llama3-1-70b-instruct-v1:0");
    let req = json!({
    "model": "meta.llama3-1-70b-instruct-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    let prompt = out["prompt"].as_str().expect("prompt present");
    assert!(
        prompt.starts_with("<|begin_of_text|>"),
        "Llama-3 prompt must start with <|begin_of_text|>, got: {prompt}"
    );
}

/// Every message is wrapped in `<|start_header_id|>{role}<|end_header_id|>`
/// so the model can distinguish turns.
#[test]
fn llama3_prompt_wraps_each_message_in_header_tokens() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/meta.llama3-8b-instruct-v1:0");
    let req = json!({
    "model": "meta.llama3-8b-instruct-v1:0",
    "messages": [{"role": "user", "content": "Hello"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    let prompt = out["prompt"].as_str().expect("prompt present");
    assert!(
        prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"),
        "user message must be header-wrapped, got: {prompt}"
    );
}

/// The prompt must terminate with an empty assistant header so the model
/// continues the conversation rather than waiting for more input.
#[test]
fn llama3_prompt_ends_with_empty_assistant_header() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/meta.llama3-1-70b-instruct-v1:0");
    let req = json!({
    "model": "meta.llama3-1-70b-instruct-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    let prompt = out["prompt"].as_str().expect("prompt present");
    assert!(
        prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        "Llama-3 prompt must end with empty assistant header, got: {prompt}"
    );
}

/// A system message is rendered with the `system` header, before the user
/// turn — matching the official Llama-3 template.
#[test]
fn llama3_prompt_formats_system_message_with_system_header() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/meta.llama3-1-70b-instruct-v1:0");
    let req = json!({
    "model": "meta.llama3-1-70b-instruct-v1:0",
    "messages": [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "Hi"},
    ],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    let prompt = out["prompt"].as_str().expect("prompt present");
    let expected = "<|begin_of_text|>\
<|start_header_id|>system<|end_header_id|>\n\nBe terse.<|eot_id|>\
<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n";
    assert_eq!(prompt, expected);
}

/// A multi-turn (user/assistant/user) conversation renders each turn in
/// order with the right header, ending on the assistant prompt.
#[test]
fn llama3_prompt_renders_multi_turn_conversation() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/meta.llama3-1-70b-instruct-v1:0");
    let req = json!({
    "model": "meta.llama3-1-70b-instruct-v1:0",
    "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
    ],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    let prompt = out["prompt"].as_str().expect("prompt present");
    let expected = "<|begin_of_text|>\
<|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n4<|eot_id|>\
<|start_header_id|>user<|end_header_id|>\n\nAnd 3+3?<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n";
    assert_eq!(prompt, expected);
}

/// Regression guard: the Llama path must NOT fall back to plain
/// concatenation. The old shape was `"Hi\n"`; the new shape wraps every
/// message in Llama-3 special tokens.
#[test]
fn llama3_prompt_does_not_use_plain_concatenation() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/meta.llama3-1-70b-instruct-v1:0");
    let req = json!({
    "model": "meta.llama3-1-70b-instruct-v1:0",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    let prompt = out["prompt"].as_str().expect("prompt present");
    assert_ne!(prompt, "Hi\n", "must not fall back to plain concatenation");
    assert!(
        !prompt.starts_with("Hi"),
        "prompt must begin with the BOS token, not the user content"
    );
}

/// Titan still uses plain-concatenation `inputText` — only Llama switched
/// to a vendor-specific template. Guards against an over-eager refactor
/// rewriting Titan's body too.
#[test]
fn titan_prompt_still_uses_plain_concatenation() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/amazon.titan-text-express-v1");
    let req = json!({
    "model": "amazon.titan-text-express-v1",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["inputText"], "Hi\n");
    assert!(
        out.get("prompt").is_none(),
        "Titan uses inputText, not prompt"
    );
}

#[test]
fn transform_request_does_not_inject_performance_config_for_anthropic_models() {
    // The Anthropic delegation path produces an Anthropic-shaped body where
    // `performanceConfig` is not a valid field. Even if the deployment
    // configures it, we must not forward it for Anthropic models.
    let p = BedrockProvider;
    let dep = bedrock_deployment_with_extra(
        "bedrock/anthropic.claude-3-5-sonnet",
        &[("performance_config", json!({"latency": "optimized"}))],
    );
    let req = json!({
    "model": "anthropic.claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert!(
        out.get("performanceConfig").is_none(),
        "Anthropic-on-Bedrock must not get a performanceConfig body field"
    );
}

// =============================================================================
// Feature: vendor streaming tool calls.
//
// Meta Llama and Mistral on Bedrock can stream tool-call data inside
// InvokeModelWithResponseStream chunks. The base vendor stream transform only
// surfaced text; we now emit OpenAI-shaped `delta.tool_calls` deltas when the
// chunk carries tool-call data.
// =============================================================================

#[test]
fn transform_stream_chunk_meta_llama_tool_calls_emit_openai_delta() {
    let p = BedrockProvider;
    // Meta Llama 3.1+ streaming chunk carrying tool calls at the top level.
    let chunk = r#"{"tool_calls":[{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}]}"#;
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("should emit a tool_calls delta");
    assert_eq!(out["object"], "chat.completion.chunk");
    let tool_calls = &out["choices"][0]["delta"]["tool_calls"];
    let arr = tool_calls.as_array().expect("tool_calls array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["index"], 0);
    assert_eq!(arr[0]["id"], "call_bedrock");
    assert_eq!(arr[0]["type"], "function");
    assert_eq!(arr[0]["function"]["name"], "get_weather");
    assert_eq!(arr[0]["function"]["arguments"], r#"{"city":"Paris"}"#);
    // No text content delta when emitting tool_calls.
    assert!(out["choices"][0]["delta"].get("content").is_none());
}

#[test]
fn transform_stream_chunk_mistral_outputs_tool_calls_emit_openai_delta() {
    let p = BedrockProvider;
    // Mistral streaming shape: tool_calls nested under outputs[].tool_calls.
    let chunk = r#"{"outputs":[{"text":"","tool_calls":[{"id":"call_abc","name":"search","arguments":"{\"q\":\"rust\"}"}]}]}"#;
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("should emit a tool_calls delta");
    let arr = out["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["id"], "call_abc");
    assert_eq!(arr[0]["function"]["name"], "search");
    assert_eq!(arr[0]["function"]["arguments"], r#"{"q":"rust"}"#);
}

#[test]
fn transform_stream_chunk_tool_calls_with_structured_input_are_serialised() {
    // Some vendors ship tool-call arguments as structured JSON rather than a
    // pre-serialised string. The delta must always carry a JSON string.
    let value: Value = serde_json::from_str(
        r#"{"tool_calls":[{"name":"get_weather","input":{"city":"Paris","unit":"C"}}]}"#,
    )
    .unwrap();
    let out = transform_vendor_stream_chunk(&value).expect("some chunk");
    let args = &out["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"];
    let args_str = args.as_str().expect("arguments string");
    let parsed: Value = serde_json::from_str(args_str).expect("arguments parseable json");
    assert_eq!(parsed["city"], "Paris");
    assert_eq!(parsed["unit"], "C");
}

#[test]
fn transform_stream_chunk_text_only_chunk_still_yields_content_delta() {
    let p = BedrockProvider;
    let chunk = r#"{"outputs":[{"text":"hello"}]}"#;
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("text chunk");
    assert_eq!(out["choices"][0]["delta"]["content"], "hello");
    assert!(out["choices"][0]["delta"].get("tool_calls").is_none());
}

#[test]
fn transform_stream_chunk_empty_tool_calls_array_falls_back_to_text() {
    // An empty tool_calls array should not suppress an accompanying text
    // delta — surface the text.
    let p = BedrockProvider;
    let chunk = r#"{"outputs":[{"text":"hi","tool_calls":[]}]}"#;
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("fallback text");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

// =============================================================================
// Converse API transform (Batch 1).
//
// `converse.rs` implements a pure OpenAI ⇄ Converse transform. These tests
// exercise the request, response, and streaming paths directly via the
// public functions, plus the dual-path wiring in `BedrockProvider`
// (`use_converse_api` flag → endpoint URL + request/response dispatch).
// =============================================================================

// =============================================================================
// Model routing: inference profiles, throughput/context suffixes, vendor
// detection, and the must-use-Converse list.
// =============================================================================

/// An application inference profile ARN is auto-routed to Converse even
/// when the deployment does not set `use_converse_api`. InvokeModel cannot
/// resolve a provider from the opaque ARN, but Converse is provider-agnostic.
#[test]
fn inference_profile_arn_routes_to_converse() {
    assert!(should_use_converse(
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123",
        false,
    ));
}

/// A regular model with no converse flag stays on the Invoke path.
#[test]
fn regular_model_without_flag_stays_on_invoke() {
    assert!(!should_use_converse(
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        false,
    ));
}

/// The explicit `use_converse_api` flag always routes to Converse.
#[test]
fn explicit_converse_flag_routes_to_converse() {
    assert!(should_use_converse("anthropic.claude-3-5-sonnet", true));
}

/// Nova, DeepSeek, and new-generation Mistral auto-route to Converse:
/// their InvokeModel bodies are not OpenAI-shaped (Nova) or require a
/// `messages` array the legacy `prompt` transform does not send — verified
/// live 2026-08 (nova-micro InvokeModel 400s on `max_tokens`; deepseek.v3.2
/// and ministral-3-8b fail with `missing field 'messages'`; all three
/// Converse fine). Legacy Mistral keeps the working `prompt` Invoke path.
#[test]
fn nova_deepseek_and_new_mistral_auto_route_to_converse() {
    for id in [
        "amazon.nova-micro-v1:0",
        "amazon.nova-lite-v1:0:300k",
        "us.amazon.nova-pro-v1:0",
        "deepseek.v3.2",
        "deepseek.r1-v1:0",
        "mistral.ministral-3-8b-instruct",
        "mistral.mistral-large-3-675b-instruct",
        "mistral.devstral-2-123b",
        "mistral.magistral-small-2509",
    ] {
        assert!(
            should_use_converse(id, false),
            "{id} must auto-route to Converse"
        );
    }
    // Legacy Mistral ids keep the InvokeModel prompt path (still accepted
    // upstream — verified live on mixtral-8x7b).
    for id in [
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mistral-small-2402-v1:0",
    ] {
        assert!(
            !should_use_converse(id, false),
            "legacy {id} must keep the InvokeModel path"
        );
    }
}

/// Throughput suffix `:51k` is stripped, leaving the version tag intact.
#[test]
fn strip_throughput_suffix_51k() {
    assert_eq!(
        strip_model_suffixes("anthropic.claude-3-5-sonnet-20241022-v2:0:51k"),
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    );
}

/// Throughput suffix `:18k` is stripped from a cross-region-prefixed model.
#[test]
fn strip_throughput_suffix_18k() {
    assert_eq!(
        strip_model_suffixes("us.meta.llama3-2-11b-instruct-v1:0:18k"),
        "us.meta.llama3-2-11b-instruct-v1:0",
    );
}

/// Context-window bracket `[1m]` is stripped.
#[test]
fn strip_context_window_1m() {
    assert_eq!(
        strip_model_suffixes("us.anthropic.claude-opus-4-6-v1[1m]"),
        "us.anthropic.claude-opus-4-6-v1",
    );
}

/// Context-window bracket `[200k]` is stripped.
#[test]
fn strip_context_window_200k() {
    assert_eq!(
        strip_model_suffixes("anthropic.claude-3-5-sonnet[200k]"),
        "anthropic.claude-3-5-sonnet",
    );
}

/// A model with no throughput/context-window suffix is returned unchanged
/// (version tags like `:0` are preserved — they are not throughput suffixes).
#[test]
fn strip_suffixes_noop_without_suffix() {
    assert_eq!(
        strip_model_suffixes("anthropic.claude-3-5-sonnet-20241022-v2:0"),
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    );
}

/// `anthropic.*` models are detected as Anthropic, including through the
/// `bedrock/` routing and cross-region prefixes.
#[test]
fn detect_anthropic_vendor() {
    assert_eq!(
        detect_bedrock_provider("anthropic.claude-3-5-sonnet"),
        BedrockVendor::Anthropic,
    );
    assert_eq!(
        detect_bedrock_provider("bedrock/us.anthropic.claude-opus-4"),
        BedrockVendor::Anthropic,
    );
}

/// `amazon.nova*` models are detected as Nova (not the generic Amazon).
#[test]
fn detect_nova_vendor() {
    assert_eq!(
        detect_bedrock_provider("amazon.nova-pro-v1:0"),
        BedrockVendor::Nova,
    );
    assert_eq!(
        detect_bedrock_provider("us.amazon.nova-lite-v1:0"),
        BedrockVendor::Nova,
    );
}

/// `meta.*` models are detected as Llama.
#[test]
fn detect_llama_vendor() {
    assert_eq!(
        detect_bedrock_provider("meta.llama3-1-70b-instruct-v1:0"),
        BedrockVendor::Llama,
    );
}

/// Unrecognised vendor prefixes yield `Unknown`.
#[test]
fn detect_unknown_vendor() {
    assert_eq!(
        detect_bedrock_provider("unknown.xyz-v1:0"),
        BedrockVendor::Unknown,
    );
}

/// `deepseek.` is in the must-use-Converse list and auto-routes to Converse
/// without the explicit flag: Bedrock's DeepSeek bodies require an
/// OpenAI-style `messages` array, so the legacy `prompt` InvokeModel shape
/// fails with `missing field 'messages'` (verified live 2026-08 on
/// deepseek.v3.2; Converse works). Versioned variants also match.
#[test]
fn deepseek_auto_routes_to_converse() {
    assert!(should_use_converse("deepseek.r1", false));
    assert!(should_use_converse("bedrock/deepseek.v3.2", false));
    assert!(should_use_converse("bedrock/deepseek.r1-v1:0", false));
}

/// The endpoint URL reflects suffix stripping: a model with a throughput
/// suffix produces a clean `/model/<base>/invoke` URL.
#[test]
fn endpoint_url_strips_throughput_suffix() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0:51k");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet-20241022-v2:0/invoke",
    );
}

/// The endpoint URL flips to `/converse` for an inference profile ARN
/// deployment even without `use_converse_api`.
#[test]
fn endpoint_url_converse_for_inference_profile_arn() {
    let p = BedrockProvider;
    let dep = bedrock_deployment(
        "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123",
    );
    let url = p.endpoint_url(&dep);
    assert!(
        url.ends_with("/converse"),
        "inference profile must auto-route to Converse: {url}",
    );
}

/// `transform_stream_chunk_stateful` must not panic when the host-owned
/// `state` slot holds a foreign `Any` type. Instead it falls back to the
/// stateless parser, which still emits valid OpenAI-shaped chunks (just
/// without tool-name restoration).
#[test]
fn transform_stream_chunk_stateful_falls_back_when_state_is_foreign_type() {
    let p = BedrockProvider;
    let mut state: Option<Box<dyn std::any::Any + Send + Sync>> = Some(Box::new(0_u32));
    let chunk = r#"{"metadata":{"usage":{"inputTokens":1,"outputTokens":2,"totalTokens":3}}}"#;
    let req = json!({"messages": []});
    let result = p.transform_stream_chunk_stateful(chunk, &mut state, &req);
    assert!(result.is_ok(), "must not panic on foreign state type");
}

/// Anthropic-on-Bedrock (InvokeModel) stream events must flow through the
/// STATEFUL Anthropic transform, not the stateless `thread_local!`-backed
/// one. Each test runs on a fresh thread (empty thread-local cells), so if
/// the stateless path were used the follow-up chunk would carry no
/// `id`/`model`; with the stateful path the `message_start` metadata is
/// carried in the host-owned slot and injected into every later chunk —
/// exactly what the multi-threaded tokio runtime needs.
#[test]
fn transform_stream_chunk_stateful_carries_anthropic_meta_across_chunks() {
    let p = BedrockProvider;
    let mut state: Option<Box<dyn std::any::Any + Send + Sync>> = None;
    let req = json!({"messages": []});

    let start = r#"data: {"type":"message_start","message":{"id":"msg_bd_1","model":"anthropic.claude-3-5-sonnet-v2:0","role":"assistant","content":[],"usage":{"input_tokens":9,"output_tokens":1}}}"#;
    let out = p
        .transform_stream_chunk_stateful(start, &mut state, &req)
        .expect("ok")
        .expect("start chunk");
    assert_eq!(out["id"], "msg_bd_1");

    // Second chunk on the same stream: must inherit id/model/created from
    // the state slot (the stateless path would lose them on a fresh thread).
    let delta = r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
    let out = p
        .transform_stream_chunk_stateful(delta, &mut state, &req)
        .expect("ok")
        .expect("delta chunk");
    assert_eq!(out["id"], "msg_bd_1");
    assert_eq!(out["model"], "anthropic.claude-3-5-sonnet-v2:0");
    assert!(out["created"].as_u64().is_some());
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

#[cfg(test)]
mod converse_tests {
    use super::{BedrockProvider, bedrock_deployment, bedrock_deployment_with_extra};
    use crate::bedrock::converse;
    use oxllm_core::provider::ChatProvider;
    use serde_json::{Value, json};
    use std::sync::Mutex;

    /// An empty `extra` map — the common case for tests that don't exercise
    /// deployment-level features (guardrails, performance config).
    fn empty_extra() -> serde_json::Map<String, Value> {
        serde_json::Map::new()
    }

    fn native_schema(body: &Value) -> Value {
        assert!(body["outputConfig"].get("format").is_none());
        assert!(body["outputConfig"].get("schema").is_none());
        assert_eq!(body["outputConfig"]["textFormat"]["type"], "json_schema");
        serde_json::from_str(
            body["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]
                .as_str()
                .expect("wire schema must be a JSON string"),
        )
        .expect("schema JSON")
    }

    /// Build an `extra` map from key-value pairs (convenience for tests that
    /// DO exercise deployment-level features).
    fn extra(pairs: &[(&str, Value)]) -> serde_json::Map<String, Value> {
        let mut m = serde_json::Map::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        m
    }

    // -----------------------------------------------------------------------
    // Request transform
    // -----------------------------------------------------------------------

    /// Basic text request: system message becomes a `system[]` array of
    /// `{text: …}` blocks; user message becomes a Converse `{role:"user",
    /// content:[{text:…}]}` entry; `max_tokens` / `temperature` /
    /// `top_p` / `stop` map onto `inferenceConfig` (camelCase).
    #[test]
    fn converse_request_basic_text_maps_messages_and_inference_config() {
        let req = json!({
            "model": "anthropic.claude-3-5-sonnet",
            "messages": [
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Hello!"},
            ],
            "max_tokens": 128,
            "temperature": 0.5,
            "top_p": 0.9,
            "stop": ["\n\n"],
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");

        // System blocks accumulate.
        let system = out["system"].as_array().expect("system array");
        assert_eq!(system.len(), 1);
        assert_eq!(system[0]["text"], "You are a pirate.");

        // Messages: single user turn.
        let messages = out["messages"].as_array().expect("messages array");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        let content = messages[0]["content"].as_array().expect("content array");
        assert_eq!(content[0]["text"], "Hello!");

        // inferenceConfig uses Converse camelCase keys.
        let ic = &out["inferenceConfig"];
        assert_eq!(ic["maxTokens"], 128);
        assert_eq!(ic["temperature"], 0.5);
        assert_eq!(ic["topP"], 0.9);
        assert_eq!(ic["stopSequences"][0], "\n\n");

        // No modelId in body — model id lives in the URL path on Converse.
        assert!(out.get("modelId").is_none());
        // No leftover OpenAI-only keys.
        assert!(out.get("max_tokens").is_none());
        assert!(out.get("temperature").is_none());
    }

    #[test]
    fn converse_thinking_drops_temperature_other_than_one() {
        let dropped = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        });
        let out = converse::transform_request_converse(
            &dropped,
            "anthropic.claude-sonnet-4-5",
            &empty_extra(),
        )
        .expect("ok");
        assert!(out["inferenceConfig"].get("temperature").is_none());
        assert_eq!(out["inferenceConfig"]["topP"], 0.9);
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["type"],
            "enabled"
        );

        let kept = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 1,
            "thinking": {"type": "adaptive"},
        });
        let out = converse::transform_request_converse(
            &kept,
            "anthropic.claude-sonnet-4-6",
            &empty_extra(),
        )
        .expect("ok");
        assert_eq!(out["inferenceConfig"]["temperature"], 1.0);
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["type"],
            "adaptive"
        );

        let plain = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.4,
        });
        let out = converse::transform_request_converse(
            &plain,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        assert_eq!(out["inferenceConfig"]["temperature"], 0.4);
    }

    #[test]
    fn converse_drops_blank_thinking_content() {
        let req = json!({
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "  ", "signature": "sig_blank"},
                    {"type": "text", "text": "Answer."},
                    {"type": "redacted_thinking", "data": "opaque"},
                ],
            }],
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5",
            &empty_extra(),
        )
        .expect("ok");
        let content = out["messages"][0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "Answer.");
        assert_eq!(content[1]["reasoningContent"]["redactedContent"], "opaque");
    }

    /// Empty inference params still produce an `inferenceConfig: {}` field
    /// (matches the reference — request shape stays stable).
    #[test]
    fn converse_request_always_emits_inference_config_even_when_empty() {
        let req = json!({
            "messages": [{"role": "user", "content": "Hi"}],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(out["inferenceConfig"].is_object());
        assert!(out["inferenceConfig"].as_object().unwrap().is_empty());
    }

    #[test]
    fn converse_gpt6_drops_sampling_and_clamps_max_tokens() {
        let req = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "temperature": 0.2,
            "top_p": 0.5,
        });
        let out =
            converse::transform_request_converse(&req, "us.openai.gpt-6-luna", &empty_extra())
                .expect("ok");
        let ic = &out["inferenceConfig"];
        assert_eq!(ic["maxTokens"], 16);
        assert!(ic.get("temperature").is_none());
        assert!(ic.get("topP").is_none());
    }

    /// Multi-modal: `image_url` with a `data:` URL becomes a Converse
    /// image block `{image:{format, source:{bytes}}}`. Media type prefix
    /// is stripped (`image/png` → `png`).
    #[test]
    fn converse_request_data_url_image_maps_to_converse_image_block() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                ],
            }],
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        let content = &out["messages"][0]["content"];
        let blocks = content.as_array().expect("content array");
        assert_eq!(blocks[0]["text"], "What's this?");
        assert_eq!(blocks[1]["image"]["format"], "png");
        assert_eq!(blocks[1]["image"]["source"]["bytes"], "iVBORw0KGgo=");
    }

    /// Remote images must have been inlined by the host; losing them silently
    /// would let a text-only prompt run in place of the multimodal request.
    #[test]
    fn converse_request_remote_image_url_is_rejected() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                ],
            }],
        });
        assert!(matches!(
            converse::transform_request_converse(&req, "m", &empty_extra()),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }

    #[test]
    fn converse_request_pdf_data_url_maps_to_document_and_injects_text() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,JVBERi0="}},
                ],
            }],
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        let blocks = out["messages"][0]["content"].as_array().expect("content");
        assert_eq!(
            blocks.len(),
            2,
            "document-only user turn needs a text sibling"
        );
        assert_eq!(blocks[0]["document"]["format"], "pdf");
        assert_eq!(blocks[0]["document"]["source"]["bytes"], "JVBERi0=");
        assert_eq!(blocks[1]["text"], ".");
    }

    #[test]
    fn converse_request_application_inference_profile_gets_output_config() {
        let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123";
        let req = json!({
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object", "properties": {}}}
            },
        });
        let out = converse::transform_request_converse(&req, arn, &empty_extra()).expect("ok");
        assert_eq!(native_schema(&out)["type"], "object");
        assert!(
            out.get("toolConfig").is_none(),
            "AIP ARNs use the provider-independent native wire shape; AWS validates model support"
        );
    }

    #[test]
    fn converse_request_glm5_json_schema_uses_text_format() {
        let req = json!({
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "item",
                    "schema": {"type": "object", "properties": {"n": {"type": "number"}}}
                }
            },
        });
        let out =
            converse::transform_request_converse(&req, "zai.glm-5", &empty_extra()).expect("ok");
        assert_eq!(out["outputConfig"]["textFormat"]["type"], "json_schema");
        assert!(
            out.get("toolConfig").is_none(),
            "GLM-5 uses native outputConfig.textFormat, not a synthetic tool"
        );
    }

    /// Tool spec mapping: OpenAI `{type:"function", function:{name,
    /// description, parameters}}` → Converse `{toolSpec:{name, description,
    /// inputSchema:{json}}}`.
    #[test]
    fn converse_request_tools_map_to_tool_spec_format() {
        let req = json!({
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let tools = &out["toolConfig"]["tools"];
        let arr = tools.as_array().expect("tools array");
        assert_eq!(arr.len(), 1);
        let spec = &arr[0]["toolSpec"];
        assert_eq!(spec["name"], "get_weather");
        assert_eq!(spec["description"], "Get weather");
        assert_eq!(
            spec["inputSchema"]["json"]["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(spec["inputSchema"]["json"]["required"][0], "city");
    }

    /// Converse tool input_schema sanitization: `type:"custom"` (the Claude
    /// Code custom-tool marker) must be normalized to `type:"object"` so
    /// Bedrock's JSON Schema validator accepts the schema. The reference does the
    /// equivalent in `normalize_json_schema_custom_types_to_object`.
    #[test]
    fn converse_tool_input_schema_type_custom_normalized() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "custom_tool",
                    "parameters": {
                        "type": "custom",
                        "properties": {"foo": {"type": "string"}},
                        "required": ["foo"],
                    },
                },
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let schema = &out["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"];
        assert_eq!(
            schema["type"], "object",
            "type:\"custom\" must be normalized to \"object\" on the Converse path"
        );
        assert_eq!(schema["properties"]["foo"]["type"], "string");
        assert_eq!(schema["required"][0], "foo");
    }

    /// Converse tool input_schema sanitization is recursive: a nested
    /// `type:"custom"` inside `properties` must be normalized too. Routing
    /// Converse through the shared sanitizer also gains the `oneOf` → `anyOf`
    /// rewrite and `uniqueItems` stripping that were previously missing.
    #[test]
    fn converse_tool_input_schema_nested_custom_normalized() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "nested_custom",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "shape": {
                                "type": "custom",
                                "properties": {"x": {"type": "integer"}}
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "uniqueItems": true
                            }
                        },
                    },
                },
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let props = &out["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"];

        // Nested custom type → object.
        assert_eq!(
            props["shape"]["type"], "object",
            "nested type:\"custom\" must be normalized recursively"
        );
        assert_eq!(props["shape"]["properties"]["x"]["type"], "integer");

        // Bonus: `uniqueItems` is now stripped on the Converse path too.
        assert!(
            props["tags"].get("uniqueItems").is_none(),
            "uniqueItems must be stripped on the Converse path"
        );
        assert_eq!(props["tags"]["items"]["type"], "string");
    }

    /// Tool choice mapping: each OpenAI form maps onto its Converse
    /// equivalent. `none` drops the entire `toolConfig`.
    #[test]
    fn converse_request_tool_choice_maps_correctly() {
        for (choice, expected_key) in [
            (json!({"type": "auto"}), "auto"),
            (json!({"type": "required"}), "any"),
        ] {
            let req = json!({
                "messages": [{"role": "user", "content": "go"}],
                "tools": [{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
                "tool_choice": choice,
            });
            let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
            let tc = &out["toolConfig"]["toolChoice"];
            assert!(
                tc.get(expected_key).is_some(),
                "expected toolChoice.{expected_key} for {choice}, got {tc}"
            );
        }

        // Function-specific tool choice.
        let req = json!({
            "messages": [{"role":"user","content":"go"}],
            "tools": [{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "f"}},
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(out["toolConfig"]["toolChoice"]["tool"]["name"], "f");

        // `none` drops toolConfig entirely.
        let req = json!({
            "messages": [{"role":"user","content":"go"}],
            "tools": [{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
            "tool_choice": {"type": "none"},
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(
            out.get("toolConfig").is_none(),
            "tool_choice:none must drop toolConfig"
        );
    }

    /// Bare-string `tool_choice` forms (`"none"`, `"auto"`, `"required"`)
    /// are OpenAI-legal and must map onto the same Converse shapes as their
    /// object equivalents. `"none"` drops `toolConfig` entirely (a safety
    /// concern: previously tools were still sent upstream).
    #[test]
    fn converse_request_string_tool_choice_forms_map_correctly() {
        // `"none"` suppresses tools entirely even when tools are declared.
        let req = json!({
            "messages": [{"role":"user","content":"go"}],
            "tools": [{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
            "tool_choice": "none",
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(
            out.get("toolConfig").is_none(),
            "string tool_choice:\"none\" must drop toolConfig entirely"
        );

        // `"required"` → {any: {}}.
        let req = json!({
            "messages": [{"role":"user","content":"go"}],
            "tools": [{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
            "tool_choice": "required",
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(
            out["toolConfig"]["toolChoice"]["any"]
                .as_object()
                .map(|o| o.len()),
            Some(0),
            "string tool_choice:\"required\" must map to {{any:{{}}}}"
        );

        // `"auto"` → {auto: {}}.
        let req = json!({
            "messages": [{"role":"user","content":"go"}],
            "tools": [{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
            "tool_choice": "auto",
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(
            out["toolConfig"]["toolChoice"]["auto"]
                .as_object()
                .map(|o| o.len()),
            Some(0),
            "string tool_choice:\"auto\" must map to {{auto:{{}}}}"
        );
    }

    /// Assistant tool-call message: `tool_calls[].function.arguments`
    /// (JSON-encoded string) becomes Converse `toolUse.input` (structured
    /// object). Validates round-trip back into a Converse-friendly shape
    /// for multi-turn tool-call conversations.
    #[test]
    fn converse_request_assistant_tool_calls_map_to_tool_use() {
        let req = json!({
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}",
                        },
                    }],
                },
            ],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let messages = out["messages"].as_array().expect("messages");
        // User + assistant (assistant's null content yields no text block).
        assert_eq!(messages.len(), 2);
        let assistant = &messages[1];
        assert_eq!(assistant["role"], "assistant");
        let blocks = assistant["content"].as_array().expect("content");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["toolUse"]["toolUseId"], "call_abc");
        assert_eq!(blocks[0]["toolUse"]["name"], "get_weather");
        // Arguments string was parsed back into a structured object.
        assert_eq!(blocks[0]["toolUse"]["input"]["city"], "Paris");
    }

    /// Tool-result message: OpenAI `{role:"tool", tool_call_id, content}`
    /// becomes Converse `{role:"user", content:[{toolResult:{toolUseId,
    /// content:[{text:…}]}}]}` — Bedrock has no `tool` role, so results
    /// ride inside a synthetic user message.
    #[test]
    fn converse_request_tool_result_message_maps_to_tool_result_block() {
        let req = json!({
            "messages": [{
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "Sunny, 22C",
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let messages = out["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        let block = &messages[0]["content"][0];
        assert_eq!(block["toolResult"]["toolUseId"], "call_abc");
        assert_eq!(block["toolResult"]["content"][0]["text"], "Sunny, 22C");
    }

    /// Pass-through top-level Converse fields. `guardrailConfig`,
    /// `performanceConfig`, etc. are Batch 3 features; when the caller
    /// supplies them we forward onto the canonical Converse field names.
    /// `thinking` is NOT a top-level Converse field — the Converse API only
    /// accepts it nested under `additionalModelRequestFields.thinking`, so a
    /// top-level copy must be absent (it would be a ValidationException).
    #[test]
    fn converse_request_passes_through_native_top_level_fields() {
        let req = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "guardrailConfig": {"guardrailIdentifier": "gr", "guardrailVersion": "1"},
            "performanceConfig": {"latency": "optimized"},
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(
            out.get("thinking").is_none(),
            "top-level `thinking` is not a Converse field; it must ride under additionalModelRequestFields"
        );
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["type"],
            "enabled"
        );
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["budget_tokens"],
            1024
        );
        assert_eq!(out["guardrailConfig"]["guardrailIdentifier"], "gr");
        assert_eq!(out["performanceConfig"]["latency"], "optimized");
    }

    /// snake_case aliases of the pass-through keys are renamed onto the
    /// camelCase Converse field instead of forwarded verbatim (Bedrock
    /// rejects the snake_case key names), and a snake_case
    /// `additional_model_request_fields` merges into the camelCase object.
    #[test]
    fn converse_request_renames_snake_case_aliases_to_converse_fields() {
        let req = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "guardrail_config": {"guardrailIdentifier": "gr", "guardrailVersion": "1"},
            "performance_config": {"latency": "optimized"},
            "request_metadata": {"k": "v"},
            "output_config": {"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {"schema": "{}"}}}},
            "additional_model_request_fields": {"custom": true},
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(out.get("guardrail_config").is_none());
        assert_eq!(out["guardrailConfig"]["guardrailIdentifier"], "gr");
        assert!(out.get("performance_config").is_none());
        assert_eq!(out["performanceConfig"]["latency"], "optimized");
        assert!(out.get("request_metadata").is_none());
        assert_eq!(out["requestMetadata"]["k"], "v");
        assert!(out.get("output_config").is_none());
        assert_eq!(out["outputConfig"]["textFormat"]["type"], "json_schema");
        assert!(out.get("additional_model_request_fields").is_none());
        assert_eq!(out["additionalModelRequestFields"]["custom"], true);
    }

    /// A caller-supplied `thinking` under the Bedrock-1024-token minimum is
    /// clamped (and maxTokens auto-set), proving the nested mapping feeds
    /// `clamp_thinking_budget` / `auto_set_max_tokens_for_thinking` instead
    /// of bypassing them.
    #[test]
    fn converse_request_caller_thinking_is_clamped_in_additional_fields() {
        let req = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 10},
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["budget_tokens"],
            1024
        );
        assert_eq!(out["inferenceConfig"]["maxTokens"], 1024 + 4096);
    }

    // -----------------------------------------------------------------------
    // Response transform
    // -----------------------------------------------------------------------

    /// Converse non-streaming response: text blocks concatenate into
    /// `message.content`; `stopReason` maps onto OpenAI `finish_reason`;
    /// `usage` tokens get renamed (`inputTokens` → `prompt_tokens` etc.).
    #[test]
    fn converse_response_text_and_usage_map_to_openai_shape() {
        let body = json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Hello "},
                        {"text": "from Bedrock!"},
                    ],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
        });
        let out = converse::transform_response_converse(&body, "anthropic.claude-3-5-sonnet")
            .expect("ok");
        assert_eq!(out["object"], "chat.completion");
        assert_eq!(out["model"], "anthropic.claude-3-5-sonnet");
        assert!(out.get("created").is_some());
        assert_eq!(
            out["choices"][0]["message"]["content"],
            "Hello from Bedrock!"
        );
        assert_eq!(out["choices"][0]["message"]["role"], "assistant");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
        assert_eq!(out["usage"]["prompt_tokens"], 10);
        assert_eq!(out["usage"]["completion_tokens"], 4);
        assert_eq!(out["usage"]["total_tokens"], 14);
    }

    /// Tool-use response: `toolUse` block becomes `message.tool_calls[]`
    /// with serialised `arguments` string (matching OpenAI's wire shape).
    #[test]
    fn converse_response_tool_use_maps_to_tool_calls() {
        let body = json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {
                            "toolUseId": "tooluse_abc",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }},
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 8, "outputTokens": 12, "totalTokens": 20},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
        let tc = &out["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "tooluse_abc");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
        // Arguments must be a JSON-encoded string.
        let args = tc["function"]["arguments"].as_str().expect("string");
        let parsed: Value = serde_json::from_str(args).expect("valid json");
        assert_eq!(parsed["city"], "Paris");
    }

    /// `tool_calls[].index` must be the position within the `tool_calls`
    /// array (0, 1, 2, …), not the content-block index. A
    /// `[text, toolUse, toolUse]` turn yields indices `[0, 1]`, matching
    /// OpenAI's non-streaming `chat.completion` shape and the reference
    /// `_get_indexed_tool_calls`.
    #[test]
    fn converse_response_tool_call_index_is_array_position_not_block_index() {
        let body = json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "thinking…"},
                        {"toolUse": {
                            "toolUseId": "tu_0",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }},
                        {"toolUse": {
                            "toolUseId": "tu_1",
                            "name": "get_time",
                            "input": {"zone": "UTC"},
                        }},
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        let tool_calls = out["choices"][0]["message"]["tool_calls"]
            .as_array()
            .expect("tool_calls array");
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0]["index"], 0);
        assert_eq!(tool_calls[1]["index"], 1);
    }

    /// Thinking blocks: `reasoningContent.reasoningText.text` is surfaced
    /// as both `reasoning_content` (concatenated text) and a round-trip
    /// `thinking_blocks` array (preserving `signature`). Redacted thinking
    /// contributes no text but is preserved as a `redacted_thinking` block.
    #[test]
    fn converse_response_reasoning_content_surfaces_as_reasoning_and_blocks() {
        let body = json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "Considering ", "signature": "sig_a"}}},
                        {"reasoningContent": {"reasoningText": {"text": "the question.", "signature": "sig_b"}}},
                        {"reasoningContent": {"redactedContent": "opaque-bytes"}},
                        {"text": "Answer: 42."},
                    ],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
        });
        let out = converse::transform_response_converse(&body, "anthropic.claude-3-7-sonnet")
            .expect("ok");
        let msg = &out["choices"][0]["message"];
        assert_eq!(msg["content"], "Answer: 42.");
        assert_eq!(msg["reasoning_content"], "Considering the question.");
        let blocks = msg["thinking_blocks"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["signature"], "sig_a");
        assert_eq!(blocks[2]["type"], "redacted_thinking");
        assert_eq!(blocks[2]["data"], "opaque-bytes");
    }

    /// Converse `stopReason` → OpenAI `finish_reason` mapping coverage.
    #[test]
    fn converse_response_finish_reason_mapping() {
        // Mapping table for present `stopReason` values.
        for (reason, expected) in [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("stop", "stop"),
            ("max_tokens", "length"),
            ("length", "length"),
            ("tool_use", "tool_calls"),
            ("tool_calls", "tool_calls"),
            ("content_filter", "content_filter"),
            ("content_filtered", "content_filter"),
        ] {
            let body = json!({
                "output": {"message": {"role": "assistant", "content": [{"text": "x"}]}},
                "stopReason": reason,
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            });
            let out = converse::transform_response_converse(&body, "m").expect("ok");
            assert_eq!(
                out["choices"][0]["finish_reason"], expected,
                "for stopReason={reason:?}"
            );
        }

        // Missing stopReason defaults to "stop".
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "x"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
    }

    // -----------------------------------------------------------------------
    // Stream chunk transform
    // -----------------------------------------------------------------------

    /// `messageStart` emits the role-only chunk that primes an OpenAI
    /// streaming response (`delta:{role:"assistant"}`).
    #[test]
    fn converse_stream_message_start_emits_role_chunk() {
        let chunk = r#"{"role":"assistant"}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(out["object"], "chat.completion.chunk");
        assert_eq!(out["choices"][0]["delta"]["role"], "assistant");
        assert!(out["choices"][0]["finish_reason"].is_null());
    }

    /// boto-SDK-style wrapper `{"messageStart":{...}}` is unwrapped
    /// transparently.
    #[test]
    fn converse_stream_unwraps_boto_wrapper() {
        let chunk = r#"{"messageStart":{"role":"assistant"}}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(out["choices"][0]["delta"]["role"], "assistant");
    }

    /// `contentBlockDelta` with `delta.text` emits a content delta.
    #[test]
    fn converse_stream_text_delta_emits_content_delta() {
        let chunk = r#"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(out["choices"][0]["delta"]["content"], "Hello");
    }

    /// `contentBlockStart` for a tool-use block emits a tool_call delta
    /// carrying the id + name (arguments empty until deltas arrive).
    #[test]
    fn converse_stream_tool_use_start_emits_tool_call_open() {
        let chunk = r#"{
            "contentBlockIndex": 0,
            "start": {"toolUse": {"toolUseId": "tooluse_abc", "name": "get_weather"}}
        }"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        let tc = &out["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 0);
        assert_eq!(tc["id"], "tooluse_abc");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
        assert_eq!(tc["function"]["arguments"], "");
    }

    /// `contentBlockDelta` for a tool-use block streams argument bytes
    /// (Converse sends them as a string fragment under `delta.toolUse.input`).
    #[test]
    fn converse_stream_tool_use_delta_emits_argument_fragment() {
        let chunk = r#"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"city\":"}}}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        let tc = &out["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 0);
        assert_eq!(tc["function"]["arguments"], "{\"city\":");
    }

    /// `contentBlockStop` is a no-op for OpenAI consumers (the tool_call
    /// delta is finalised by the subsequent `messageStop`).
    #[test]
    fn converse_stream_content_block_stop_is_no_op() {
        let chunk = r#"{"contentBlockIndex":0}"#;
        let out = converse::transform_stream_chunk_converse(chunk).expect("ok");
        assert!(out.is_none(), "contentBlockStop must produce no chunk");
    }

    /// `messageStop` emits the finish reason only (empty delta).
    #[test]
    fn converse_stream_message_stop_emits_finish_reason() {
        let chunk = r#"{"stopReason":"end_turn"}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
        // Delta is an empty object.
        assert_eq!(out["choices"][0]["delta"].as_object().unwrap().len(), 0);
    }

    /// `metadata` carries the final `usage` totals — emitted as a chunk
    /// with empty delta + a `usage` field (matches OpenAI's
    /// `stream_options.include_usage` shape).
    #[test]
    fn converse_stream_metadata_emits_usage_chunk() {
        let chunk = r#"{"usage":{"inputTokens":10,"outputTokens":4,"totalTokens":14}}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(out["usage"]["prompt_tokens"], 10);
        assert_eq!(out["usage"]["completion_tokens"], 4);
        assert_eq!(out["usage"]["total_tokens"], 14);
    }

    /// The streaming `metadata` event surfaces the reasoning-token
    /// breakdown via `completion_tokens_details` when Bedrock reports
    /// `outputTokensDetails.reasoningTokens`, mirroring the non-streaming
    /// Converse response transform. Previously streaming clients lost this
    /// detail.
    #[test]
    fn converse_stream_metadata_reasoning_tokens_detail() {
        let chunk = r#"{"usage":{"inputTokens":10,"outputTokens":12,"totalTokens":22,"outputTokensDetails":{"reasoningTokens":7}}}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(out["usage"]["completion_tokens"], 12);
        assert_eq!(
            out["usage"]["completion_tokens_details"]["reasoning_tokens"],
            7
        );
        assert_eq!(
            out["usage"]["completion_tokens_details"]["text_tokens"], 5,
            "text_tokens = completion_tokens - reasoning_tokens"
        );
    }

    /// Without `outputTokensDetails`, `completion_tokens_details` is omitted
    /// (no spurious zero breakdown).
    #[test]
    fn converse_stream_metadata_no_reasoning_detail_when_absent() {
        let chunk = r#"{"usage":{"inputTokens":10,"outputTokens":4,"totalTokens":14}}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert!(out["usage"].get("completion_tokens_details").is_none());
    }

    /// Reasoning content deltas are surfaced as `delta.reasoning_content`
    /// so extended-thinking streaming mirrors the Anthropic-on-Bedrock
    /// InvokeModel behaviour.
    #[test]
    fn converse_stream_reasoning_delta_emits_reasoning_content() {
        let chunk =
            r#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"Considering "}}}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(
            out["choices"][0]["delta"]["reasoning_content"],
            "Considering "
        );
    }

    /// SSE `data:` prefix and trailing whitespace are handled.
    #[test]
    fn converse_stream_strips_sse_data_prefix() {
        let chunk = "data: {\"role\":\"assistant\"}\n\n";
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(out["choices"][0]["delta"]["role"], "assistant");
    }

    /// Empty / [DONE] / unparseable chunks return `Ok(None)`.
    #[test]
    fn converse_stream_skips_empty_done_and_garbage() {
        assert!(
            converse::transform_stream_chunk_converse("")
                .expect("ok")
                .is_none()
        );
        assert!(
            converse::transform_stream_chunk_converse("[DONE]")
                .expect("ok")
                .is_none()
        );
        assert!(
            converse::transform_stream_chunk_converse("not json")
                .expect("ok")
                .is_none()
        );
    }

    // -----------------------------------------------------------------------
    // Dual-path wiring in BedrockProvider
    // -----------------------------------------------------------------------

    /// `use_converse_api: true` flips the URL suffix to `/converse`.
    #[test]
    fn endpoint_url_converse_when_flag_set() {
        let p = BedrockProvider;
        let dep = bedrock_deployment_with_extra(
            "bedrock/anthropic.claude-3-5-sonnet",
            &[("use_converse_api", json!(true))],
        );
        assert_eq!(
            p.endpoint_url(&dep),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/converse"
        );
    }

    /// The `use_converse` alias is also honoured.
    #[test]
    fn endpoint_url_converse_when_short_alias_set() {
        let p = BedrockProvider;
        let dep = bedrock_deployment_with_extra(
            "bedrock/anthropic.claude-3-5-sonnet",
            &[("use_converse", json!(true))],
        );
        assert!(
            p.endpoint_url(&dep).ends_with("/converse"),
            "use_converse alias must flip endpoint"
        );
    }

    /// Default (no flag) keeps the InvokeModel URL.
    #[test]
    fn endpoint_url_invoke_when_flag_absent() {
        let p = BedrockProvider;
        let dep = bedrock_deployment("bedrock/anthropic.claude-3-5-sonnet");
        assert!(p.endpoint_url(&dep).ends_with("/invoke"));
    }

    /// `use_converse_api: false` explicitly keeps the InvokeModel URL.
    #[test]
    fn endpoint_url_invoke_when_flag_false() {
        let p = BedrockProvider;
        let dep = bedrock_deployment_with_extra(
            "bedrock/anthropic.claude-3-5-sonnet",
            &[("use_converse_api", json!(false))],
        );
        assert!(p.endpoint_url(&dep).ends_with("/invoke"));
    }

    /// Full request round-trip via `BedrockProvider::transform_request`:
    /// when Converse is enabled, the body is Converse-shaped (no
    /// `anthropic_version`, no OpenAI params, has `inferenceConfig`).
    #[test]
    fn provider_transform_request_dispatches_to_converse_when_enabled() {
        let p = BedrockProvider;
        let dep = bedrock_deployment_with_extra(
            "bedrock/anthropic.claude-3-5-sonnet",
            &[("use_converse_api", json!(true))],
        );
        let req = json!({
            "model": "anthropic.claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
        });
        let out = p.transform_request(&req, &dep).expect("ok");
        // Converse shape signals.
        assert!(out.get("inferenceConfig").is_some());
        assert_eq!(out["inferenceConfig"]["maxTokens"], 50);
        assert_eq!(out["messages"][0]["role"], "user");
        assert_eq!(out["messages"][0]["content"][0]["text"], "Hi");
        // InvokeModel-specific fields absent.
        assert!(out.get("anthropic_version").is_none());
        assert!(out.get("max_tokens").is_none());
    }

    /// Provider-level response dispatch: a Converse-shaped body routes
    /// through the Converse transform even though `transform_response`
    /// doesn't receive deployment context.
    #[test]
    fn provider_transform_response_dispatches_on_converse_shape() {
        let p = BedrockProvider;
        let body = json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hi"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        });
        let out = p
            .transform_response(&body, "meta.llama3-1-70b-instruct-v1:0")
            .expect("ok");
        assert_eq!(out["object"], "chat.completion");
        assert_eq!(out["choices"][0]["message"]["content"], "Hi");
        assert_eq!(out["usage"]["prompt_tokens"], 1);
    }

    /// Provider-level stream dispatch: a Converse chunk shape routes
    /// through the Converse parser. Verified by sending a `messageStart`
    /// payload through `BedrockProvider::transform_stream_chunk`.
    #[test]
    fn provider_transform_stream_chunk_dispatches_on_converse_shape() {
        let p = BedrockProvider;
        let chunk = r#"{"role":"assistant"}"#;
        let out = p.transform_stream_chunk(chunk).expect("ok").expect("some");
        assert_eq!(out["object"], "chat.completion.chunk");
        assert_eq!(out["choices"][0]["delta"]["role"], "assistant");
    }

    /// Non-Converse chunks still flow through the legacy parser after the
    /// Converse dispatch is added (regression guard).
    #[test]
    fn provider_transform_stream_chunk_legacy_path_still_works() {
        let p = BedrockProvider;
        // Anthropic-style event — not Converse.
        let chunk =
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
        let out = p.transform_stream_chunk(chunk).expect("ok").expect("some");
        assert_eq!(out["choices"][0]["delta"]["content"], "Hi");
    }

    // -----------------------------------------------------------------------
    // Dispatch heuristic: `looks_like_converse_event`
    //
    // The streaming trait method only sees the raw chunk string — no
    // deployment context — so the Converse-vs-Anthropic dispatch is decided
    // by inspecting the parsed event shape. These tests pin the heuristic:
    // snake_case Anthropic `type` discriminators must read as "not Converse"
    // (they were previously misrouted because they also carry `delta` /
    // `usage` fields), and the canonical Converse wire shapes must read as
    // "Converse".
    // -----------------------------------------------------------------------

    /// Every snake_case Anthropic streaming event type is rejected — these
    /// are the events that previously collided with Converse's field-shape
    /// heuristic because they happen to carry `delta` / `usage` keys.
    #[test]
    fn converse_looks_like_rejects_snake_case_anthropic_events() {
        for raw in [
            r#"{"type":"message_start","message":{"id":"msg_1"}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":20}}"#,
            r#"{"type":"message_stop"}"#,
            r#"{"type":"ping"}"#,
        ] {
            let value: Value =
                serde_json::from_str(raw).unwrap_or_else(|e| panic!("bad json {raw:?}: {e}"));
            assert!(
                !converse::looks_like_converse_event(&value),
                "Anthropic snake_case event should NOT look like Converse: {raw}"
            );
        }
    }

    /// Canonical Converse wire shapes — both the boto-SDK wrapper form and
    /// the raw eventstream form — are accepted as Converse.
    #[test]
    fn converse_looks_like_accepts_camel_case_converse_events() {
        // boto-SDK wrapper form.
        for raw in [
            r#"{"messageStart":{"role":"assistant"}}"#,
            r#"{"contentBlockStart":{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"u","name":"f"}}}}"#,
            r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"Hi"}}}"#,
            r#"{"contentBlockStop":{"contentBlockIndex":0}}"#,
            r#"{"messageStop":{"stopReason":"end_turn"}}"#,
            r#"{"metadata":{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}}"#,
        ] {
            let value: Value =
                serde_json::from_str(raw).unwrap_or_else(|e| panic!("bad json {raw:?}: {e}"));
            assert!(
                converse::looks_like_converse_event(&value),
                "Converse camelCase event should look like Converse: {raw}"
            );
        }

        // Raw eventstream form (no wrapper) — identified by characteristic
        // Converse fields (`role`, `stopReason`, `contentBlockIndex`).
        for raw in [
            r#"{"role":"assistant"}"#,
            r#"{"stopReason":"end_turn"}"#,
            r#"{"contentBlockIndex":0,"delta":{"text":"Hi"}}"#,
            r#"{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"u","name":"f"}}}"#,
            r#"{"contentBlockIndex":0}"#,
            r#"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#,
        ] {
            let value: Value =
                serde_json::from_str(raw).unwrap_or_else(|e| panic!("bad json {raw:?}: {e}"));
            assert!(
                converse::looks_like_converse_event(&value),
                "Converse raw eventstream event should look like Converse: {raw}"
            );
        }
    }

    /// Unrecognised shapes default to "not Converse" so the legacy
    /// Anthropic / vendor path handles them (safer fallback).
    #[test]
    fn converse_looks_like_defaults_false_for_unknown_shapes() {
        for raw in [
            r#"{}"#,
            r#"{"foo":"bar"}"#,
            r#"{"type":"something_without_underscore"}"#,
        ] {
            let value: Value =
                serde_json::from_str(raw).unwrap_or_else(|e| panic!("bad json {raw:?}: {e}"));
            assert!(
                !converse::looks_like_converse_event(&value),
                "unknown shape should default to not-Converse: {raw}"
            );
        }
    }

    // ========================================================================
    // Batch 2: guardrailConfig (deployment-level)
    // ========================================================================

    /// When `extra.guardrail_id` + `extra.guardrail_version` are present,
    /// the Converse body includes `guardrailConfig: {guardrailIdentifier,
    /// guardrailVersion}`.
    #[test]
    fn converse_request_injects_guardrail_config_from_deployment_snake_case() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let extra = extra(&[
            ("guardrail_id", json!("ff6ujrregl1q")),
            ("guardrail_version", json!("DRAFT")),
        ]);
        let out = converse::transform_request_converse(&req, "anthropic.claude-3-5-sonnet", &extra)
            .expect("ok");
        assert_eq!(
            out["guardrailConfig"]["guardrailIdentifier"],
            "ff6ujrregl1q"
        );
        assert_eq!(out["guardrailConfig"]["guardrailVersion"], "DRAFT");
    }

    /// camelCase variant (`guardrailIdentifier` / `guardrailVersion`) is also
    /// accepted.
    #[test]
    fn converse_request_injects_guardrail_config_from_deployment_camel_case() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let extra = extra(&[
            ("guardrailIdentifier", json!("gr-abc")),
            ("guardrailVersion", json!("3")),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        assert_eq!(out["guardrailConfig"]["guardrailIdentifier"], "gr-abc");
        assert_eq!(out["guardrailConfig"]["guardrailVersion"], "3");
    }

    /// Without guardrail config, `guardrailConfig` must be absent.
    #[test]
    fn converse_request_omits_guardrail_config_when_unconfigured() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(out.get("guardrailConfig").is_none());
    }

    /// Guardrail requires BOTH id and version; one alone produces nothing.
    #[test]
    fn converse_request_omits_guardrail_config_with_only_id() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let extra = extra(&[("guardrail_id", json!("orphan"))]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        assert!(out.get("guardrailConfig").is_none());
    }

    /// Guardrail trace is a string within the Converse guardrail configuration.
    #[test]
    fn converse_request_injects_trace_when_guardrail_trace_enabled() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let extra = extra(&[
            ("guardrail_id", json!("gr")),
            ("guardrail_version", json!("1")),
            ("guardrail_trace", json!(true)),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        assert_eq!(out["guardrailConfig"]["trace"], "enabled");
        assert!(out.get("trace").is_none());
    }

    /// `trace` is absent when `guardrail_trace` is false or missing.
    #[test]
    fn converse_request_omits_trace_when_not_enabled() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let extra = extra(&[
            ("guardrail_id", json!("gr")),
            ("guardrail_version", json!("1")),
            ("guardrail_trace", json!(false)),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        assert!(out.get("trace").is_none());
    }

    // ========================================================================
    // Batch 2: guardrailConfig → guarded_text auto-conversion
    // (mirrors the reference consecutive-user-message guardrail conversion)
    // ========================================================================

    /// No `guardrailConfig` → user text is left as a plain `{text: "…"}`
    /// block (never wrapped in `guardContent`).
    #[test]
    fn converse_guardrail_absent_no_wrapping() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let blocks = out["messages"][0]["content"]
            .as_array()
            .expect("content array");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["text"], "Hi");
        assert!(blocks[0].get("guardContent").is_none());
    }

    /// With `guardrailConfig`, the trailing user message's text block is
    /// wrapped in a `guardContent` block.
    #[test]
    fn converse_guardrail_present_wraps_last_user_text() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}],
            }],
        });
        let extra = extra(&[
            ("guardrail_id", json!("gr")),
            ("guardrail_version", json!("1")),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        let blocks = out["messages"][0]["content"]
            .as_array()
            .expect("content array");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["guardContent"]["text"]["text"], "Hi");
        assert!(blocks[0].get("text").is_none());
    }

    /// String `content` on a trailing user message is converted to an array
    /// containing a single `guardContent` block.
    #[test]
    fn converse_guardrail_wraps_string_content() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let extra = extra(&[
            ("guardrail_id", json!("gr")),
            ("guardrail_version", json!("1")),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        let blocks = out["messages"][0]["content"]
            .as_array()
            .expect("content array");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["guardContent"]["text"]["text"], "Hi");
    }

    /// A message that already carries a `guarded_text` block is left alone
    /// (caller override) — no double-wrapping, and sibling text blocks stay
    /// plain text rather than being re-tagged.
    #[test]
    fn converse_guardrail_skips_already_guarded() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "guarded_text", "text": "pre-tagged"},
                    {"type": "text", "text": "plain"},
                ],
            }],
        });
        let extra = extra(&[
            ("guardrail_id", json!("gr")),
            ("guardrail_version", json!("1")),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        let blocks = out["messages"][0]["content"]
            .as_array()
            .expect("content array");
        assert_eq!(blocks.len(), 2);
        // Pre-tagged block → guardContent.
        assert_eq!(blocks[0]["guardContent"]["text"]["text"], "pre-tagged");
        // Sibling text block stays a plain text block (message skipped).
        assert_eq!(blocks[1]["text"], "plain");
        assert!(blocks[1].get("guardContent").is_none());
    }

    /// ALL trailing consecutive user messages are wrapped, not just the last.
    #[test]
    fn converse_guardrail_wraps_consecutive_user_messages() {
        let req = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "first"}]},
                {"role": "user", "content": [{"type": "text", "text": "second"}]},
            ],
        });
        let extra = extra(&[
            ("guardrail_id", json!("gr")),
            ("guardrail_version", json!("1")),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        let msgs = out["messages"].as_array().expect("messages");
        assert_eq!(msgs.len(), 1);
        assert_eq!(
            msgs[0]["content"][0]["guardContent"]["text"]["text"],
            "first"
        );
        assert_eq!(
            msgs[0]["content"][1]["guardContent"]["text"]["text"],
            "second"
        );
    }

    /// A user message that is NOT in the trailing run (followed by an
    /// assistant message) is left untouched.
    #[test]
    fn converse_guardrail_leaves_non_trailing_user_untouched() {
        let req = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "earlier"}]},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": [{"type": "text", "text": "latest"}]},
            ],
        });
        let extra = extra(&[
            ("guardrail_id", json!("gr")),
            ("guardrail_version", json!("1")),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        let msgs = out["messages"].as_array().expect("messages");
        assert_eq!(msgs.len(), 3);
        // Non-trailing user message → plain text.
        assert_eq!(msgs[0]["content"][0]["text"], "earlier");
        assert!(msgs[0]["content"][0].get("guardContent").is_none());
        // Trailing user message → guarded.
        assert_eq!(
            msgs[2]["content"][0]["guardContent"]["text"]["text"],
            "latest"
        );
    }

    /// Mixed text + image content: only the text block is wrapped, the
    /// image block is left untouched.
    #[test]
    fn converse_guardrail_leaves_image_blocks_untouched() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,Qk=="}},
                ],
            }],
        });
        let extra = extra(&[
            ("guardrail_id", json!("gr")),
            ("guardrail_version", json!("1")),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        let blocks = out["messages"][0]["content"]
            .as_array()
            .expect("content array");
        assert_eq!(blocks.len(), 2);
        // Text block → guardContent.
        assert_eq!(blocks[0]["guardContent"]["text"]["text"], "describe this");
        // Image block → still an image block.
        assert_eq!(blocks[1]["image"]["format"], "png");
        assert!(blocks[1].get("guardContent").is_none());
    }

    /// The final wire output contains a proper `guardContent` block when
    /// guardrails are active.
    #[test]
    fn converse_guardrail_outputs_guardcontent_block() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let extra = extra(&[
            ("guardrail_id", json!("ff6ujrregl1q")),
            ("guardrail_version", json!("DRAFT")),
        ]);
        let out = converse::transform_request_converse(&req, "anthropic.claude-3-5-sonnet", &extra)
            .expect("ok");
        assert_eq!(
            out["guardrailConfig"]["guardrailIdentifier"],
            "ff6ujrregl1q"
        );
        let block = &out["messages"][0]["content"][0];
        assert_eq!(block["guardContent"]["text"]["text"], "Hi");
    }

    /// Empty text in a trailing user block is skipped — no `guardContent`
    /// block is emitted for it (Bedrock rejects empty text blocks). The
    /// message ends up with no content blocks and is dropped, so the output
    /// must not contain any `guardContent` anywhere.
    #[test]
    fn converse_guardrail_empty_text_skipped() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": ""}],
            }],
        });
        let extra = extra(&[
            ("guardrail_id", json!("gr")),
            ("guardrail_version", json!("1")),
        ]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        let msgs = out["messages"].as_array().expect("messages array");
        // No message carries a guardContent block.
        for msg in msgs {
            if let Some(content) = msg["content"].as_array() {
                for block in content {
                    assert!(
                        block.get("guardContent").is_none(),
                        "empty guarded text must not emit a guardContent block"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Batch 2: performanceConfig (deployment-level)
    // ========================================================================

    #[test]
    fn converse_request_injects_performance_config_snake_case() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let extra = extra(&[("performance_config", json!({"latency": "optimized"}))]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        assert_eq!(out["performanceConfig"]["latency"], "optimized");
    }

    #[test]
    fn converse_request_injects_performance_config_camel_case() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let extra = extra(&[("performanceConfig", json!({"latency": "standard"}))]);
        let out = converse::transform_request_converse(&req, "m", &extra).expect("ok");
        assert_eq!(out["performanceConfig"]["latency"], "standard");
    }

    #[test]
    fn converse_request_omits_performance_config_when_unconfigured() {
        let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(out.get("performanceConfig").is_none());
    }

    // ========================================================================
    // Batch 2: reasoning_effort → additionalModelRequestFields.thinking
    // ========================================================================

    /// `reasoning_effort: "high"` on an Anthropic model maps to
    /// `additionalModelRequestFields.thinking = {type:"enabled", budget_tokens:10000}`.
    #[test]
    fn converse_request_reasoning_effort_high_maps_to_thinking_for_anthropic() {
        let req = json!({
            "messages": [{"role": "user", "content": "Think hard."}],
            "reasoning_effort": "high",
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-7-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["type"],
            "enabled"
        );
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["budget_tokens"],
            10000
        );
    }

    /// `medium` → 4096, `low` → 1024.
    #[test]
    fn converse_request_reasoning_effort_medium_and_low_budgets() {
        for (effort, budget) in [("low", 1024u64), ("medium", 4096)] {
            let req = json!({
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": effort,
            });
            let out = converse::transform_request_converse(
                &req,
                "anthropic.claude-3-7-sonnet",
                &empty_extra(),
            )
            .expect("ok");
            assert_eq!(
                out["additionalModelRequestFields"]["thinking"]["budget_tokens"], budget,
                "for effort={effort}"
            );
        }
    }

    /// `reasoning_effort` is skipped for non-Anthropic models — the
    /// `additionalModelRequestFields.thinking` field must not appear.
    #[test]
    fn converse_request_reasoning_effort_skipped_for_non_anthropic_model() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        });
        let out = converse::transform_request_converse(
            &req,
            "meta.llama3-1-70b-instruct-v1:0",
            &empty_extra(),
        )
        .expect("ok");
        // No thinking injected.
        assert!(
            out.get("additionalModelRequestFields").is_none()
                || out["additionalModelRequestFields"]
                    .get("thinking")
                    .is_none(),
            "reasoning_effort must not produce thinking for non-Anthropic models"
        );
    }

    /// Explicit `thinking` in the request takes precedence over
    /// `reasoning_effort` — the mapping is skipped.
    #[test]
    fn converse_request_explicit_thinking_overrides_reasoning_effort() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "reasoning_effort": "high",
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-7-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        // Explicit `thinking` is mapped into
        // `additionalModelRequestFields.thinking` (the only Converse-valid
        // location) and wins over the `reasoning_effort` mapping; a
        // top-level `thinking` key must NOT be forwarded (ValidationException).
        assert!(
            out.get("thinking").is_none(),
            "top-level thinking must not ride the Converse body verbatim"
        );
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["budget_tokens"],
            2048
        );
    }

    // ========================================================================
    // Batch 2: response_format → outputConfig / synthetic tool
    // ========================================================================

    /// Schema-less JSON mode cannot be represented by a closed native schema.
    #[test]
    fn converse_request_response_format_json_on_anthropic_uses_tool_fallback() {
        let req = json!({
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {"type": "json_object"},
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        assert!(out.get("outputConfig").is_none());
        assert_eq!(
            out["toolConfig"]["toolChoice"]["tool"]["name"],
            "json_tool_call"
        );
    }

    /// Meta's Converse toolChoice cannot force a specific JSON tool.
    #[test]
    fn converse_request_response_format_json_on_meta_is_rejected() {
        let req = json!({
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {"type": "json_object"},
        });
        let error = converse::transform_request_converse(
            &req,
            "meta.llama3-1-70b-instruct-v1:0",
            &empty_extra(),
        )
        .expect_err("forced JSON tool unsupported");
        assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
    }

    /// `response_format: json_object` with a JSON schema on non-Anthropic
    /// → the schema is embedded in the synthetic tool's `inputSchema`.
    #[test]
    fn converse_request_response_format_json_with_schema_embeds_in_synthetic_tool() {
        let req = json!({
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {
                "type": "json_object",
                "json_schema": {
                    "name": "weather",
                    "schema": {
                        "type": "object",
                        "properties": {"temp": {"type": "number"}},
                        "required": ["temp"],
                    },
                },
            },
        });
        let out =
            converse::transform_request_converse(&req, "amazon.nova-pro-v1:0", &empty_extra())
                .expect("ok");
        let schema = &out["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"];
        assert_eq!(schema["properties"]["temp"]["type"], "number");
        assert_eq!(schema["required"][0], "temp");
    }

    /// Non-`json_object` response_format (e.g. `text`) produces neither
    /// `outputConfig` nor a synthetic tool.
    #[test]
    fn converse_request_non_json_response_format_is_ignored() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"},
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        assert!(out.get("outputConfig").is_none());
        assert!(out.get("toolConfig").is_none());
    }

    /// Synthetic json_tool_call response extraction: when the response
    /// contains a `toolUse` block named `json_tool_call`, its input is
    /// lifted as `content` (non-streaming path).
    #[test]
    fn converse_response_json_tool_call_lifted_as_content() {
        let body = json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {
                            "toolUseId": "tu_1",
                            "name": "json_tool_call",
                            "input": {"answer": 42, "unit": "C"},
                        }},
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 5, "outputTokens": 8, "totalTokens": 13},
        });
        let out = converse::transform_response_converse(&body, "meta.llama3-1-70b").expect("ok");
        let msg = &out["choices"][0]["message"];
        // Content is the serialized JSON of the tool input.
        let content = msg["content"].as_str().expect("string content");
        let parsed: Value = serde_json::from_str(content).expect("valid json");
        assert_eq!(parsed["answer"], 42);
        assert_eq!(parsed["unit"], "C");
        // The synthetic tool call must not leak.
        assert!(
            msg.get("tool_calls").is_none(),
            "json_tool_call must be extracted, not left as a tool_call"
        );
    }

    // ========================================================================
    // Batch 3: cache point blocks
    // ========================================================================

    /// A message-level `cache_control` on a system message injects a
    /// `{cachePoint: {type: "default"}}` block immediately after the text.
    #[test]
    fn converse_request_cache_control_on_system_message_injects_cache_point() {
        let req = json!({
            "messages": [
                {"role": "system", "content": "You are helpful.", "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": "Hi"},
            ],
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        let system = out["system"].as_array().expect("system array");
        assert_eq!(system.len(), 2);
        assert_eq!(system[0]["text"], "You are helpful.");
        assert_eq!(system[1]["cachePoint"]["type"], "default");
    }

    /// A block-level `cache_control` on a system content element injects a
    /// cachePoint after that specific text block.
    #[test]
    fn converse_request_cache_control_on_system_content_block_injects_cache_point() {
        let req = json!({
            "messages": [{
                "role": "system",
                "content": [
                    {"type": "text", "text": "part 1"},
                    {"type": "text", "text": "part 2", "cache_control": {"type": "ephemeral"}},
                ],
            }],
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        let system = out["system"].as_array().expect("system array");
        // text, text, cachePoint
        assert_eq!(system.len(), 3);
        assert_eq!(system[0]["text"], "part 1");
        assert_eq!(system[1]["text"], "part 2");
        assert_eq!(system[2]["cachePoint"]["type"], "default");
    }

    /// A message-level `cache_control` on a user message (string content)
    /// injects a cachePoint after the text block.
    #[test]
    fn converse_request_cache_control_on_user_message_injects_cache_point() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": "cache me",
                "cache_control": {"type": "ephemeral"},
            }],
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        let content = out["messages"][0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "cache me");
        assert_eq!(content[1]["cachePoint"]["type"], "default");
    }

    /// A block-level `cache_control` on a user content element injects a
    /// cachePoint after that block.
    #[test]
    fn converse_request_cache_control_on_user_content_block_injects_cache_point() {
        let req = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "q"},
                    {"type": "text", "text": "context", "cache_control": {"type": "ephemeral"}},
                ],
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let content = out["messages"][0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 3);
        assert_eq!(content[2]["cachePoint"]["type"], "default");
    }

    /// A `cache_control` on a tool-result message injects a cachePoint as a
    /// sibling block inside the synthetic user message.
    #[test]
    fn converse_request_cache_control_on_tool_result_injects_cache_point() {
        let req = json!({
            "messages": [{
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "result",
                "cache_control": {"type": "ephemeral"},
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let content = out["messages"][0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["toolResult"]["toolUseId"], "call_1");
        assert_eq!(content[1]["cachePoint"]["type"], "default");
    }

    /// Without any `cache_control`, no cachePoint blocks appear anywhere.
    #[test]
    fn converse_request_no_cache_control_means_no_cache_points() {
        let req = json!({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let system = out["system"].as_array().expect("system");
        assert_eq!(system.len(), 1);
        assert!(system[0].get("cachePoint").is_none());
        let content = out["messages"][0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 1);
        assert!(content[0].get("cachePoint").is_none());
    }

    // ========================================================================
    // Batch 3: Nova web search grounding
    // ========================================================================

    /// `web_search_options` on a Nova model injects a `systemTool` for
    /// `nova_grounding` into the toolConfig.
    #[test]
    fn converse_request_web_search_options_on_nova_injects_grounding_tool() {
        let req = json!({
            "messages": [{"role": "user", "content": "search the web"}],
            "web_search_options": {},
        });
        let out =
            converse::transform_request_converse(&req, "amazon.nova-pro-v1:0", &empty_extra())
                .expect("ok");
        let tools = out["toolConfig"]["tools"].as_array().expect("tools");
        let grounding = tools
            .iter()
            .find(|t| t.get("systemTool").is_some())
            .expect("has systemTool");
        assert_eq!(grounding["systemTool"]["name"], "nova_grounding");
    }

    /// Cross-region Nova prefix (`us.amazon.nova-…`) also triggers grounding.
    #[test]
    fn converse_request_web_search_options_on_cross_region_nova_injects_grounding() {
        let req = json!({
            "messages": [{"role": "user", "content": "search"}],
            "web_search_options": {"search_context_size": "low"},
        });
        let out =
            converse::transform_request_converse(&req, "us.amazon.nova-lite-v1:0", &empty_extra())
                .expect("ok");
        let tools = out["toolConfig"]["tools"].as_array().expect("tools");
        assert!(
            tools
                .iter()
                .any(|t| t["systemTool"]["name"] == "nova_grounding")
        );
    }

    /// `web_search_options` on a NON-Nova model is skipped — no toolConfig
    /// grounding tool is injected.
    #[test]
    fn converse_request_web_search_options_skipped_for_non_nova_model() {
        let req = json!({
            "messages": [{"role": "user", "content": "search"}],
            "web_search_options": {},
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-5-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        // No grounding systemTool, and no toolConfig at all when no other tools.
        assert!(
            out.get("toolConfig").is_none(),
            "non-Nova model must not get nova_grounding tool"
        );
    }

    /// Without `web_search_options`, a Nova model gets no grounding tool.
    #[test]
    fn converse_request_nova_without_web_search_options_has_no_grounding() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
        });
        let out =
            converse::transform_request_converse(&req, "amazon.nova-pro-v1:0", &empty_extra())
                .expect("ok");
        assert!(out.get("toolConfig").is_none());
    }

    // ========================================================================
    // Batch 3: service tier
    // ========================================================================

    /// `service_tier: "flex"` → top-level `serviceTier: {type: "flex"}`.
    #[test]
    fn converse_request_service_tier_flex_maps_to_service_tier_block() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "flex",
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(out["serviceTier"]["type"], "flex");
    }

    /// `priority` and `default` pass through verbatim.
    #[test]
    fn converse_request_service_tier_priority_and_default() {
        for tier in ["priority", "default"] {
            let req = json!({
                "messages": [{"role": "user", "content": "hi"}],
                "service_tier": tier,
            });
            let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
            assert_eq!(out["serviceTier"]["type"], tier, "for tier={tier}");
        }
    }

    /// OpenAI's `"auto"` maps onto Bedrock's `"default"` (Bedrock has no auto).
    #[test]
    fn converse_request_service_tier_auto_maps_to_default() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "auto",
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(out["serviceTier"]["type"], "default");
    }

    /// Unknown `service_tier` values are dropped (Bedrock would reject them).
    #[test]
    fn converse_request_service_tier_unknown_value_is_dropped() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "bogus",
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(out.get("serviceTier").is_none());
    }

    /// Without `service_tier`, no `serviceTier` field appears.
    #[test]
    fn converse_request_no_service_tier_means_no_field() {
        let req = json!({"messages": [{"role": "user", "content": "hi"}]});
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(out.get("serviceTier").is_none());
    }

    /// A pre-shaped `serviceTier` object from the caller is forwarded
    /// verbatim (pass-through takes precedence over the string mapping).
    #[test]
    fn converse_request_preshaped_service_tier_object_passes_through() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "serviceTier": {"type": "flex"},
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(out["serviceTier"]["type"], "flex");
    }

    // ========================================================================
    // Batch 3: context management
    // ========================================================================

    /// `context_management` (list form) is forwarded into
    /// `additionalModelRequestFields.context_management`.
    #[test]
    fn converse_request_context_management_list_forwarded_to_amrf() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "context_management": [{"type": "compact_20260112"}],
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5",
            &empty_extra(),
        )
        .expect("ok");
        let cm = &out["additionalModelRequestFields"]["context_management"];
        assert!(cm.is_array());
        assert_eq!(cm[0]["type"], "compact_20260112");
    }

    /// `context_management` (dict form) is forwarded verbatim.
    #[test]
    fn converse_request_context_management_dict_forwarded_to_amrf() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "context_management": {"edits": [{"type": "compact_20260112"}]},
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let cm = &out["additionalModelRequestFields"]["context_management"];
        assert_eq!(cm["edits"][0]["type"], "compact_20260112");
    }

    /// Without `context_management`, no `context_management` key in AMRF.
    #[test]
    fn converse_request_no_context_management_means_no_field() {
        let req = json!({"messages": [{"role": "user", "content": "hi"}]});
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let amrf_present = out.get("additionalModelRequestFields").is_some();
        let cm_present = out
            .get("additionalModelRequestFields")
            .and_then(|a| a.get("context_management"))
            .is_some();
        assert!(
            !cm_present,
            "no context_management key expected (amrf={amrf_present})"
        );
    }

    // ========================================================================
    // Batch 3: tool name sanitization
    //
    // Converse enforces `^[a-zA-Z0-9_-]{1,64}$` on tool names. The
    // transform sanitizes caller names going up and restores them on the
    // response/stream way back via a process-wide reverse map. These tests
    // are serialized because they touch that shared cell.
    // ========================================================================

    /// Serializes tests that exercise the process-wide `tool_name_map_cell`
    /// (writer = request transform, reader = response / stream transform).
    static CONVERSE_TOOL_NAME_CELL_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// A tool name with invalid chars (e.g. dots) is sanitized to use `_`,
    /// and the response restores the original name via the reverse map.
    #[test]
    fn converse_request_sanitizes_tool_name_with_invalid_chars() {
        let _guard = CONVERSE_TOOL_NAME_CELL_TEST_LOCK.lock().unwrap();
        let req = json!({
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get.weather.v2",
                    "parameters": {"type": "object"},
                },
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let name = &out["toolConfig"]["tools"][0]["toolSpec"]["name"];
        assert_eq!(name, "get_weather_v2");
    }

    /// `tool_choice` referencing a sanitized tool name is also remapped.
    #[test]
    fn converse_request_sanitizes_tool_choice_name() {
        let _guard = CONVERSE_TOOL_NAME_CELL_TEST_LOCK.lock().unwrap();
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {"name": "my.tool", "parameters": {"type": "object"}},
            }],
            "tool_choice": {"type": "function", "function": {"name": "my.tool"}},
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(out["toolConfig"]["toolChoice"]["tool"]["name"], "my_tool");
    }

    /// An assistant `tool_calls` name in the message history is remapped to
    /// the sanitized form in the outgoing `toolUse` block.
    #[test]
    fn converse_request_sanitizes_assistant_tool_use_name() {
        let _guard = CONVERSE_TOOL_NAME_CELL_TEST_LOCK.lock().unwrap();
        let req = json!({
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "my.tool", "arguments": "{}"},
                }]},
            ],
            "tools": [{
                "type": "function",
                "function": {"name": "my.tool", "parameters": {"type": "object"}},
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let tu_name = &out["messages"][1]["content"][0]["toolUse"]["name"];
        assert_eq!(tu_name, "my_tool");
    }

    /// A valid tool name (`get_weather`, already matching the charset) is
    /// left unchanged.
    #[test]
    fn converse_request_valid_tool_name_left_unchanged() {
        let _guard = CONVERSE_TOOL_NAME_CELL_TEST_LOCK.lock().unwrap();
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(
            out["toolConfig"]["tools"][0]["toolSpec"]["name"],
            "get_weather"
        );
    }

    /// The response transform restores the original caller name from the
    /// sanitized name the model returned. Uses a distinctive name to avoid
    /// colliding with the regular response test's `get_weather` lookup.
    #[test]
    fn converse_response_restores_original_tool_name() {
        let _guard = CONVERSE_TOOL_NAME_CELL_TEST_LOCK.lock().unwrap();
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {"name": "lookup.place", "parameters": {"type": "object"}},
            }],
        });
        // Seed the per-thread reverse map the same way the production
        // `transform_response_with_request` hook does: rebuild it from the
        // original request (NOT from a process-wide cell that a concurrent
        // request would clobber).
        converse::seed_tool_name_map(converse::reverse_map_from_request(&req));
        // Model returns the sanitized name.
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "tu_1", "name": "lookup_place", "input": {}}},
            ]}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert_eq!(
            out["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "lookup.place",
            "original dotted name must be restored"
        );
    }

    /// The stream transform restores the original caller name on
    /// `contentBlockStart` tool-use events. Uses the stateful entry point so
    /// the reverse map is threaded in explicitly (per-stream), not read from
    /// a process-wide cell.
    #[test]
    fn converse_stream_restores_original_tool_name() {
        let _guard = CONVERSE_TOOL_NAME_CELL_TEST_LOCK.lock().unwrap();
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {"name": "search.web", "parameters": {"type": "object"}},
            }],
        });
        let reverse_map = converse::reverse_map_from_request(&req);
        let chunk = r#"{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"tu_1","name":"search_web"}}}"#;
        let out = converse::transform_stream_chunk_converse_stateful(chunk, &reverse_map)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(
            out["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "search.web"
        );
    }

    /// Collisions: two distinct invalid names that sanitize to the same
    /// string are disambiguated with a `_N` suffix.
    #[test]
    fn converse_request_collision_disambiguation() {
        let _guard = CONVERSE_TOOL_NAME_CELL_TEST_LOCK.lock().unwrap();
        // "a.b" and "a/b" both sanitize to "a_b" → second is disambiguated.
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {"type": "function", "function": {"name": "a.b", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "a/b", "parameters": {"type": "object"}}},
            ],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let names = out["toolConfig"]["tools"].as_array().expect("tools");
        assert_eq!(names[0]["toolSpec"]["name"], "a_b");
        assert_eq!(names[1]["toolSpec"]["name"], "a_b_2");
    }

    /// An already-VALID name that matches an earlier tool's sanitized form
    /// must not collapse onto it: the identity branch used to clobber the
    /// reverse-map entry for the first tool, sending BOTH tools upstream
    /// under the same name and restoring the wrong original afterwards.
    #[test]
    fn converse_request_identity_collision_with_sanitized_form() {
        let _guard = CONVERSE_TOOL_NAME_CELL_TEST_LOCK.lock().unwrap();
        // "my.tool" sanitizes to "my_tool", which is itself a valid name
        // supplied later — the second tool must be suffix-disambiguated.
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {"type": "function", "function": {"name": "my.tool", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "my_tool", "parameters": {"type": "object"}}},
            ],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let names = out["toolConfig"]["tools"].as_array().expect("tools");
        assert_eq!(names[0]["toolSpec"]["name"], "my_tool");
        assert_eq!(names[1]["toolSpec"]["name"], "my_tool_2");

        // The reverse map restores each upstream name to its own original.
        let reverse = converse::reverse_map_from_request(&req);
        assert_eq!(reverse.get("my_tool").map(String::as_str), Some("my.tool"));
        assert_eq!(
            reverse.get("my_tool_2").map(String::as_str),
            Some("my_tool")
        );
    }

    /// `max_completion_tokens` (the modern OpenAI alias) must feed
    /// `inferenceConfig.maxTokens` on the Converse path when `max_tokens`
    /// is absent — the InvokeModel path and the Anthropic transform both
    /// honour the alias.
    #[test]
    fn converse_request_max_completion_tokens_maps_to_max_tokens() {
        let req = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 512,
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(out["inferenceConfig"]["maxTokens"], 512);

        // Classic `max_tokens` still wins when both are present.
        let req = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 128,
            "max_completion_tokens": 512,
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert_eq!(out["inferenceConfig"]["maxTokens"], 128);
    }

    /// Two concurrent requests with distinct tool sets must each restore
    /// their own original tool names. Regression test for the old process-
    /// wide `OnceLock<Mutex<HashMap>>` reverse-map cell: a second request's
    /// `transform_request_converse` would clobber the first's reverse map, so
    /// the first request's response transform looked up names in the wrong
    /// map. The new per-thread / per-stream map design makes each request's
    /// reverse map independent — derived from its own original request, not
    /// from a shared global.
    #[test]
    fn converse_concurrent_requests_restore_own_tool_names() {
        use std::thread;

        // Each "request" has a distinct dotted tool name that sanitizes to
        // the same collision-free form on the way up but must restore to its
        // own original on the way back.
        let run = |tool_name: &'static str, sanitized: &'static str| {
            thread::spawn(move || {
                let req = json!({
                    "messages": [{"role": "user", "content": "go"}],
                    "tools": [{
                        "type": "function",
                        "function": {"name": tool_name, "parameters": {"type": "object"}},
                    }],
                });
                // Simulate a concurrent in-flight second request between the
                // request and response transforms — exactly the interleave
                // that broke the old global cell.
                let other = json!({
                    "messages": [{"role": "user", "content": "go"}],
                    "tools": [{
                        "type": "function",
                        "function": {"name": "other.tool", "parameters": {"type": "object"}},
                    }],
                });
                let _ = converse::transform_request_converse(&other, "m", &empty_extra());

                // Non-streaming response: reverse map rebuilt from THIS
                // request, so it must restore THIS request's name.
                converse::seed_tool_name_map(converse::reverse_map_from_request(&req));
                let body = json!({
                    "output": {"message": {"role": "assistant", "content": [
                        {"toolUse": {"toolUseId": "tu_1", "name": sanitized, "input": {}}},
                    ]}},
                    "stopReason": "tool_use",
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                });
                let out = converse::transform_response_converse(&body, "m").expect("ok");
                let restored = out["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
                    .as_str()
                    .unwrap_or("");
                assert_eq!(
                    restored, tool_name,
                    "non-streaming response must restore this request's own tool name"
                );

                // Streaming response: per-stream reverse map.
                let reverse_map = converse::reverse_map_from_request(&req);
                let chunk = format!(
                    r#"{{"contentBlockIndex":0,"start":{{"toolUse":{{"toolUseId":"tu_1","name":"{sanitized}"}}}}}}"#
                );
                let out = converse::transform_stream_chunk_converse_stateful(&chunk, &reverse_map)
                    .expect("ok")
                    .expect("some chunk");
                let restored = out["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
                    .as_str()
                    .unwrap_or("");
                assert_eq!(
                    restored, tool_name,
                    "streaming response must restore this request's own tool name"
                );
            })
        };

        let t1 = run("get.weather", "get_weather");
        let t2 = run("search.web", "search_web");
        let r1 = t1.join();
        let r2 = t2.join();
        // Surface panics from the worker threads clearly.
        r1.expect("thread 1 panicked");
        r2.expect("thread 2 panicked");
    }

    // ========================================================================
    // Tier 2 — Item 1: Cache token details in Converse usage
    // ========================================================================

    /// Non-streaming response with cache tokens: `cacheReadInputTokens` and
    /// `cacheWriteInputTokens` are surfaced as `cached_tokens` /
    /// `cache_creation_tokens`, and `prompt_tokens` is inflated by their sum
    /// (matching the OpenAI `cached_tokens ⊆ prompt_tokens` invariant).
    #[test]
    fn converse_response_cache_tokens_inflate_prompt_and_populate_details() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "totalTokens": 14,
                "cacheReadInputTokens": 7,
                "cacheWriteInputTokens": 3,
            },
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        // prompt = input(10) + cacheRead(7) + cacheWrite(3) = 20
        assert_eq!(out["usage"]["prompt_tokens"], 20);
        assert_eq!(out["usage"]["completion_tokens"], 4);
        assert_eq!(out["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
        // Cache-creation rides only the `prompt_tokens_details` subset form
        // (no top-level anthropic-style mirror — the host extractor treats
        // that as additive and would double-bill the folded tokens).
        assert_eq!(
            out["usage"]["prompt_tokens_details"]["cache_creation_tokens"],
            3
        );
        assert!(out["usage"].get("cache_creation_input_tokens").is_none());
    }

    /// Response without cache tokens: prompt_tokens == inputTokens, no
    /// `prompt_tokens_details` field.
    #[test]
    fn converse_response_no_cache_tokens_means_no_details() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert!(out["usage"].get("prompt_tokens_details").is_none());
        assert!(out["usage"].get("cache_creation_input_tokens").is_none());
    }

    /// Adversarial `u64::MAX` totals must saturate rather than overflow:
    /// in debug builds plain `+` panics, in release it wraps to a tiny
    /// number that flows into cost metering. The prompt/total must clamp
    /// to `u64::MAX` regardless of cache-token inflation.
    #[test]
    fn converse_response_usage_arithmetic_saturates_on_overflow() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "x"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": u64::MAX,
                "outputTokens": u64::MAX,
                "cacheReadInputTokens": u64::MAX,
                "cacheWriteInputTokens": u64::MAX,
            },
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert_eq!(out["usage"]["prompt_tokens"], u64::MAX);
        assert_eq!(out["usage"]["completion_tokens"], u64::MAX);
        // No explicit totalTokens → derived from prompt + completion.
        assert_eq!(out["usage"]["total_tokens"], u64::MAX);
    }

    /// Streaming metadata chunk surfaces cache tokens the same way as the
    /// non-streaming response.
    #[test]
    fn converse_stream_metadata_cache_tokens_surface() {
        let chunk = r#"{"usage":{"inputTokens":10,"outputTokens":4,"totalTokens":14,"cacheReadInputTokens":7,"cacheWriteInputTokens":3}}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(out["usage"]["prompt_tokens"], 20);
        assert_eq!(out["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
        assert_eq!(
            out["usage"]["prompt_tokens_details"]["cache_creation_tokens"],
            3
        );
        assert!(out["usage"].get("cache_creation_input_tokens").is_none());
    }

    /// Streaming metadata: adversarial `u64::MAX` totals saturate rather
    /// than overflow the streaming usage path (mirrors the non-streaming
    /// guard above).
    #[test]
    fn converse_stream_metadata_usage_arithmetic_saturates_on_overflow() {
        let chunk = r#"{"usage":{"inputTokens":18446744073709551615,"outputTokens":18446744073709551615,"cacheReadInputTokens":18446744073709551615,"cacheWriteInputTokens":18446744073709551615}}"#;
        let out = converse::transform_stream_chunk_converse(chunk)
            .expect("ok")
            .expect("some chunk");
        assert_eq!(out["usage"]["prompt_tokens"], u64::MAX);
        assert_eq!(out["usage"]["completion_tokens"], u64::MAX);
        assert_eq!(out["usage"]["total_tokens"], u64::MAX);
    }

    // ========================================================================
    // Tier 2 — Item 2: Nova 2 reasoningConfig
    // ========================================================================

    /// Nova 2 model: `reasoning_effort` maps to
    /// `additionalModelRequestFields.reasoningConfig` (NOT `thinking`).
    #[test]
    fn converse_request_nova2_reasoning_effort_maps_to_reasoning_config() {
        for effort in ["low", "medium", "high"] {
            let req = json!({
                "messages": [{"role": "user", "content": "think"}],
                "reasoning_effort": effort,
            });
            let out = converse::transform_request_converse(
                &req,
                "amazon.nova-2-lite-v1:0",
                &empty_extra(),
            )
            .expect("ok");
            assert_eq!(
                out["additionalModelRequestFields"]["reasoningConfig"]["type"], "enabled",
                "for effort={effort}"
            );
            assert_eq!(
                out["additionalModelRequestFields"]["reasoningConfig"]["maxReasoningEffort"],
                effort,
            );
            // thinking must NOT be injected for Nova 2.
            assert!(
                out["additionalModelRequestFields"]
                    .get("thinking")
                    .is_none(),
                "Nova 2 must use reasoningConfig, not thinking"
            );
        }
    }

    /// Nova 2 `max` / `xhigh` / unknown effort clamp to `high`.
    #[test]
    fn converse_request_nova2_max_and_xhigh_clamp_to_high() {
        for effort in ["max", "xhigh", "bogus"] {
            let req = json!({
                "messages": [{"role": "user", "content": "go"}],
                "reasoning_effort": effort,
            });
            let out = converse::transform_request_converse(
                &req,
                "amazon.nova-2-pro-v1:0",
                &empty_extra(),
            )
            .expect("ok");
            assert_eq!(
                out["additionalModelRequestFields"]["reasoningConfig"]["maxReasoningEffort"],
                "high",
                "Nova 2 must clamp {effort} to high"
            );
        }
    }

    /// Nova 2 reasoningConfig is also applied for cross-region prefixed ids.
    #[test]
    fn converse_request_nova2_cross_region_prefix_detected() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "reasoning_effort": "high",
        });
        let out = converse::transform_request_converse(
            &req,
            "us.amazon.nova-2-lite-v1:0",
            &empty_extra(),
        )
        .expect("ok");
        assert_eq!(
            out["additionalModelRequestFields"]["reasoningConfig"]["maxReasoningEffort"],
            "high"
        );
    }

    /// Non-Nova-2 / non-Anthropic model still drops `reasoning_effort`
    /// (regression guard — Nova 1 and other vendors get no reasoning block).
    #[test]
    fn converse_request_reasoning_effort_dropped_for_non_nova2_non_anthropic() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "reasoning_effort": "high",
        });
        let out = converse::transform_request_converse(
            &req,
            "meta.llama3-1-70b-instruct-v1:0",
            &empty_extra(),
        )
        .expect("ok");
        assert!(
            out.get("additionalModelRequestFields").is_none()
                || out["additionalModelRequestFields"]
                    .get("reasoningConfig")
                    .is_none(),
            "non-Nova-2 / non-Anthropic must not get reasoningConfig"
        );
        assert!(
            out.get("additionalModelRequestFields").is_none()
                || out["additionalModelRequestFields"]
                    .get("thinking")
                    .is_none(),
            "non-Nova-2 / non-Anthropic must not get thinking"
        );
    }

    // ========================================================================
    // Tier 2 — Item 3: Counter-based tool names
    // ========================================================================

    /// A single unnamed tool gets a counter-suffixed placeholder name.
    #[test]
    fn converse_request_unnamed_tool_gets_counter_name() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {"parameters": {"type": "object"}},
            }],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let name = &out["toolConfig"]["tools"][0]["toolSpec"]["name"];
        assert_eq!(name, "litellm_unnamed_tool_1");
    }

    /// Multiple unnamed tools each get a unique counter-based name.
    #[test]
    fn converse_request_multiple_unnamed_tools_get_unique_names() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {"type": "function", "function": {"parameters": {"type": "object"}}},
                {"type": "function", "function": {"parameters": {"type": "object"}}},
                {"type": "function", "function": {"parameters": {"type": "object"}}},
            ],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let tools = out["toolConfig"]["tools"].as_array().expect("tools");
        assert_eq!(tools[0]["toolSpec"]["name"], "litellm_unnamed_tool_1");
        assert_eq!(tools[1]["toolSpec"]["name"], "litellm_unnamed_tool_2");
        assert_eq!(tools[2]["toolSpec"]["name"], "litellm_unnamed_tool_3");
    }

    // ========================================================================
    // Tier 2 — Item 4: Output config effort passthrough
    // ========================================================================

    /// `outputConfig.effort` is a documented Converse member
    /// (`low|medium|high|xhigh|max`; Bedrock itself caps at `high` when
    /// thinking is disabled). The gateway must forward it verbatim in place —
    /// no model-specific clamp and no relocation to additionalModelRequestFields.
    #[test]
    fn converse_request_effort_stays_in_output_config() {
        for effort in ["low", "medium", "high", "xhigh", "max"] {
            for model in [
                "anthropic.opus-4-5-20251201-v1:0",
                "anthropic.opus-4-6-v1:0",
            ] {
                let req = json!({
                    "messages": [{"role": "user", "content": "go"}],
                    "outputConfig": {"effort": effort},
                });
                let out =
                    converse::transform_request_converse(&req, model, &empty_extra()).expect("ok");
                assert_eq!(
                    out["outputConfig"]["effort"], effort,
                    "effort={effort} model={model}"
                );
                assert!(
                    out.get("additionalModelRequestFields")
                        .and_then(|fields| fields.get("output_config"))
                        .is_none(),
                    "effort must not be relocated for model={model}"
                );
            }
        }
    }

    /// The Anthropic-shaped `output_config.effort` alias is normalized into
    /// the Converse `outputConfig` location and stays there.
    #[test]
    fn converse_request_snake_case_effort_lands_in_output_config() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "output_config": {"effort": "xhigh"},
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-opus-4-5-v1:0",
            &empty_extra(),
        )
        .expect("ok");
        assert_eq!(out["outputConfig"]["effort"], "xhigh");
        assert!(
            out.get("additionalModelRequestFields")
                .and_then(|fields| fields.get("output_config"))
                .is_none()
        );
    }

    // ========================================================================
    // Tier 2 — Item 5: Computer use / bash / text_editor hosted tools
    // ========================================================================

    /// Computer-use tools cannot be represented by a Converse toolSpec.
    #[test]
    fn converse_request_computer_use_tool_requires_invoke() {
        let req = json!({
            "messages": [{"role": "user", "content": "use the computer"}],
            "tools": [{
                "type": "computer_20250124",
                "function": {
                    "name": "computer",
                    "parameters": {
                        "display_width_px": 1024,
                        "display_height_px": 768,
                    },
                },
            }],
        });
        assert!(matches!(
            converse::transform_request_converse(&req, "m", &empty_extra()),
            Err(oxllm_core::Error::BadRequest(message)) if message.contains("InvokeModel")
        ));
    }

    /// Bash tool semantics must not be replaced with an ordinary function.
    #[test]
    fn converse_request_bash_tool_requires_invoke() {
        let req = json!({
            "messages": [{"role": "user", "content": "run bash"}],
            "tools": [{"type": "bash_20250124"}],
        });
        assert!(matches!(
            converse::transform_request_converse(&req, "m", &empty_extra()),
            Err(oxllm_core::Error::BadRequest(message)) if message.contains("InvokeModel")
        ));
    }

    /// Text-editor tool semantics require the supported native Invoke path.
    #[test]
    fn converse_request_text_editor_tool_requires_invoke() {
        let req = json!({
            "messages": [{"role": "user", "content": "edit text"}],
            "tools": [{"type": "text_editor_20250124"}],
        });
        assert!(matches!(
            converse::transform_request_converse(&req, "m", &empty_extra()),
            Err(oxllm_core::Error::BadRequest(message)) if message.contains("InvokeModel")
        ));
    }

    // ========================================================================
    // Tier 2 — Item 6: additionalProperties:false for structured outputs
    // ========================================================================

    /// `response_format.type: "json_schema"` on an Anthropic model produces
    /// an `outputConfig` carrying the schema with `additionalProperties:
    /// false` injected at the root.
    #[test]
    fn converse_request_json_schema_produces_output_config_with_schema() {
        let req = json!({
            "messages": [{"role": "user", "content": "return JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "schema": {
                        "type": "object",
                        "properties": {"temp": {"type": "number"}},
                        "required": ["temp"],
                    },
                },
            },
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5",
            &empty_extra(),
        )
        .expect("ok");
        let schema = native_schema(&out);
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["temp"]["type"], "number");
    }

    /// `additionalProperties: false` is injected recursively into nested
    /// object nodes (under `properties`, `items`, and combiners).
    #[test]
    fn converse_request_additional_properties_injected_recursively() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "nested": {
                                "type": "object",
                                "properties": {"x": {"type": "string"}},
                            },
                            "list": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"y": {"type": "integer"}},
                                },
                            }
                        },
                    },
                },
            },
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5",
            &empty_extra(),
        )
        .expect("ok");
        let schema = native_schema(&out);
        // Root.
        assert_eq!(schema["additionalProperties"], false);
        // Nested object.
        assert_eq!(
            schema["properties"]["nested"]["additionalProperties"],
            false
        );
        // Array item object.
        assert_eq!(
            schema["properties"]["list"]["items"]["additionalProperties"],
            false
        );
    }

    /// A deeply nested caller-controlled schema must not overflow the stack
    /// when walked by `inject_additional_properties_false`. The walker caps
    /// recursion defensively; serde_json's own parse limit bounds the input
    /// depth in practice, but the cap guarantees termination. Construct a
    /// chain of nested `properties` objects well within serde_json's limit
    /// but deeper than the walker's cap, and assert the request transform
    /// completes without recursing into the deepest levels.
    #[test]
    fn converse_request_deeply_nested_schema_does_not_overflow() {
        // Build a schema nested via `properties.deep` repeatedly, deeper than
        // the SCHEMA_MAX_DEPTH cap (64). Each level adds one object layer.
        let depth = 100;
        let mut inner = json!({"type": "object", "properties": {"leaf": {"type": "string"}}});
        for _ in 0..depth {
            inner = json!({"type": "object", "properties": {"deep": inner}});
        }
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": inner},
            },
        });
        // Must complete (not overflow / panic).
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5",
            &empty_extra(),
        )
        .expect("transform must not overflow on deep schema");
        let schema = out["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]
            .as_str()
            .expect("wire schema string");
        assert!(schema.contains("\"additionalProperties\":false"));
    }

    /// `response_format.type: "json_object"` with a schema also gets the
    /// schema embedded + additionalProperties injected (the existing
    /// bare-`{format:{type:"json"}}` shape is kept when no schema).
    #[test]
    fn converse_request_json_object_with_schema_embeds_and_injects() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "response_format": {
                "type": "json_object",
                "json_schema": {
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                },
            },
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5",
            &empty_extra(),
        )
        .expect("ok");
        let schema = native_schema(&out);
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["a"]["type"], "string");
    }

    // ========================================================================
    // Tier 2 — Item 7: Filter unsupported beta headers for Converse
    // ========================================================================

    /// `advanced-tool-use-*` beta tokens are stripped from the
    /// `anthropic-beta` header; other tokens are kept.
    #[test]
    fn converse_beta_filter_strips_advanced_tool_use() {
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "anthropic-beta".to_string(),
            "advanced-tool-use-2025-01-24,computer-use-2025-01-24".to_string(),
        );
        converse::filter_converse_beta_headers(&mut headers);
        assert_eq!(
            headers.get("anthropic-beta").map(String::as_str),
            Some("computer-use-2025-01-24"),
        );
    }

    /// `prompt-caching-*` beta tokens are stripped.
    #[test]
    fn converse_beta_filter_strips_prompt_caching() {
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "anthropic-beta".to_string(),
            "prompt-caching-2024-07-31,extended-thinking-2025-01-24".to_string(),
        );
        converse::filter_converse_beta_headers(&mut headers);
        assert_eq!(
            headers.get("anthropic-beta").map(String::as_str),
            Some("extended-thinking-2025-01-24"),
        );
    }

    /// `compact-*` beta tokens are stripped.
    #[test]
    fn converse_beta_filter_strips_compact() {
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "anthropic-beta".to_string(),
            "compact-2026-01-12,fine-grained-tool-streaming-2025-05-14".to_string(),
        );
        converse::filter_converse_beta_headers(&mut headers);
        assert_eq!(
            headers.get("anthropic-beta").map(String::as_str),
            Some("fine-grained-tool-streaming-2025-05-14"),
        );
    }

    /// Valid beta tokens are kept untouched (no false-positive stripping).
    #[test]
    fn converse_beta_filter_keeps_valid_betas() {
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "anthropic-beta".to_string(),
            "computer-use-2025-01-24,extended-thinking-2025-01-24".to_string(),
        );
        converse::filter_converse_beta_headers(&mut headers);
        assert_eq!(
            headers.get("anthropic-beta").map(String::as_str),
            Some("computer-use-2025-01-24,extended-thinking-2025-01-24"),
        );
    }

    /// When all tokens are stripped, the header is removed entirely.
    #[test]
    fn converse_beta_filter_removes_header_when_all_stripped() {
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "anthropic-beta".to_string(),
            "prompt-caching-2024-07-31,compact-2026-01-12".to_string(),
        );
        converse::filter_converse_beta_headers(&mut headers);
        assert!(!headers.contains_key("anthropic-beta"));
    }

    /// Case-insensitive header lookup works.
    #[test]
    fn converse_beta_filter_case_insensitive_header_lookup() {
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "Anthropic-Beta".to_string(),
            "prompt-caching-2024-07-31,ok-beta".to_string(),
        );
        converse::filter_converse_beta_headers(&mut headers);
        // Header is preserved under the original key with the surviving token.
        assert_eq!(
            headers.get("Anthropic-Beta").map(String::as_str),
            Some("ok-beta"),
        );
    }

    /// No-op when the header is absent.
    #[test]
    fn converse_beta_filter_noop_when_absent() {
        let mut headers = std::collections::HashMap::new();
        headers.insert("x-other".to_string(), "v".to_string());
        converse::filter_converse_beta_headers(&mut headers);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers.get("x-other").map(String::as_str), Some("v"));
    }

    // ========================================================================
    // H9: thinking budget clamped to 1024 minimum
    // ========================================================================

    /// A thinking budget below 1024 is clamped up to 1024. This exercises the
    /// caller-supplied `additionalModelRequestFields.thinking` path (via
    /// pass-through), since the `reasoning_effort` mapping never produces a
    /// value below 1024.
    #[test]
    fn converse_request_thinking_budget_below_1024_clamped() {
        let req = json!({
            "messages": [{"role": "user", "content": "think"}],
            "additionalModelRequestFields": {
                "thinking": {"type": "enabled", "budget_tokens": 512}
            },
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-7-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["budget_tokens"], 1024,
            "budget below 1024 must be clamped up"
        );
    }

    /// A thinking budget ≥ 1024 is left unchanged.
    #[test]
    fn converse_request_thinking_budget_at_or_above_1024_unchanged() {
        for budget in [1024u64, 2048, 10000] {
            let req = json!({
                "messages": [{"role": "user", "content": "think"}],
                "additionalModelRequestFields": {
                    "thinking": {"type": "enabled", "budget_tokens": budget}
                },
            });
            let out = converse::transform_request_converse(
                &req,
                "anthropic.claude-3-7-sonnet",
                &empty_extra(),
            )
            .expect("ok");
            assert_eq!(
                out["additionalModelRequestFields"]["thinking"]["budget_tokens"], budget,
                "budget >= 1024 must be unchanged (got budget={budget})"
            );
        }
    }

    // ========================================================================
    // H10: $defs / definitions walked in additionalProperties injection
    // ========================================================================

    /// `$defs` sub-schemas receive `additionalProperties: false` injection.
    #[test]
    fn converse_request_defs_walked_in_additional_properties_injection() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {"item": {"$ref": "#/$defs/thing"}},
                        "$defs": {
                            "thing": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            }
                        },
                    },
                },
            },
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5",
            &empty_extra(),
        )
        .expect("ok");
        let schema = native_schema(&out);
        let defs = &schema["$defs"]["thing"];
        assert_eq!(
            defs["additionalProperties"], false,
            "$defs sub-schema must get additionalProperties:false"
        );
        assert_eq!(defs["properties"]["name"]["type"], "string");
    }

    /// `definitions` (draft-4 keyword) sub-schemas also receive the injection.
    #[test]
    fn converse_request_definitions_walked_in_additional_properties_injection() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "definitions": {
                            "widget": {
                                "type": "object",
                                "properties": {"id": {"type": "integer"}},
                            }
                        },
                    },
                },
            },
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5",
            &empty_extra(),
        )
        .expect("ok");
        let schema = native_schema(&out);
        let def = &schema["definitions"]["widget"];
        assert_eq!(
            def["additionalProperties"], false,
            "definitions sub-schema must get additionalProperties:false"
        );
    }

    // ========================================================================
    // H14: completion_tokens_details + cache-creation details field name
    // ========================================================================

    /// When Bedrock returns `outputTokensDetails.reasoningTokens`, the response
    /// usage includes `completion_tokens_details` with the exact count and the
    /// derived `text_tokens`.
    #[test]
    fn converse_response_completion_tokens_details_from_bedrock_field() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "answer"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30,
                "outputTokensDetails": {"reasoningTokens": 15}
            },
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert_eq!(
            out["usage"]["completion_tokens_details"]["reasoning_tokens"],
            15
        );
        assert_eq!(out["usage"]["completion_tokens_details"]["text_tokens"], 5);
    }

    /// When reasoning content is present but Bedrock doesn't report
    /// `outputTokensDetails`, the reasoning token count is estimated from the
    /// reasoning text length (~4 chars/token).
    #[test]
    fn converse_response_completion_tokens_details_estimated_from_reasoning() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "considering the answer carefully"}}},
                {"text": "42"},
            ]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 40, "totalTokens": 50},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        let rt = out["usage"]["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .expect("reasoning_tokens present");
        assert!(rt > 0, "estimated reasoning_tokens must be > 0");
    }

    /// The reasoning-token estimate counts Unicode scalar values, NOT bytes.
    /// For 100 emoji chars (4 bytes each = 400 bytes), the byte-based estimate
    /// would have been 100 tokens, but the char-based estimate is 25. More
    /// importantly, the estimate must never exceed `completion_tokens` for
    /// obviously-wrong metering. Regression test for the byte-vs-char bug.
    #[test]
    fn converse_response_reasoning_estimate_uses_chars_not_bytes() {
        let emoji_text = "😀".repeat(100);
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": emoji_text}}},
                {"text": "ok"},
            ]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 30, "totalTokens": 40},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        let rt = out["usage"]["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .expect("reasoning_tokens present");
        let completion = out["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        // 100 chars / 4 = 25 tokens (char-based). The old byte-based estimate
        // would have been 400/4 = 100, exceeding completion_tokens (30).
        assert_eq!(rt, 25, "estimate must count chars (25), not bytes (100)");
        assert!(
            rt <= completion,
            "reasoning_tokens ({rt}) must not exceed completion_tokens ({completion})"
        );
    }

    /// No reasoning content → no `completion_tokens_details`.
    #[test]
    fn converse_response_no_reasoning_means_no_completion_details() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert!(out["usage"].get("completion_tokens_details").is_none());
    }

    /// Cache write tokens ride the `prompt_tokens_details.cache_creation_tokens`
    /// subset form (the top-level anthropic-style mirror was removed: the
    /// host usage extractor treats it as additive while `prompt_tokens`
    /// already folds the write tokens in — emitting both double-billed).
    #[test]
    fn converse_response_cache_creation_uses_input_field_name() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "totalTokens": 14,
                "cacheWriteInputTokens": 9,
            },
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert_eq!(
            out["usage"]["prompt_tokens_details"]["cache_creation_tokens"],
            9
        );
        assert!(
            out["usage"].get("cache_creation_input_tokens").is_none(),
            "top-level anthropic-style mirror must not be present (double-bill guard)"
        );
    }

    // ========================================================================
    // M5: guardrailIntervened → content_filter
    // ========================================================================

    /// Both `guardrail_intervened` and `guardrailIntervened` stop reasons map
    /// to OpenAI's `content_filter`.
    #[test]
    fn converse_response_guardrail_intervened_maps_to_content_filter() {
        for reason in ["guardrail_intervened", "guardrailIntervened"] {
            let body = json!({
                "output": {"message": {"role": "assistant", "content": [{"text": "x"}]}},
                "stopReason": reason,
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            });
            let out = converse::transform_response_converse(&body, "m").expect("ok");
            assert_eq!(
                out["choices"][0]["finish_reason"], "content_filter",
                "for stopReason={reason:?}"
            );
        }
    }

    // ========================================================================
    // M6: serviceTier surfaced from Converse response
    // ========================================================================

    /// Bedrock reports `serviceTier: {type: "flex"}` in the response; surface
    /// it as top-level `service_tier` (snake_case, matching the request side).
    #[test]
    fn converse_response_service_tier_surfaced() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "serviceTier": {"type": "flex"},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert_eq!(out["service_tier"], "flex");
    }

    /// No `serviceTier` in the response → no `service_tier` field in output.
    #[test]
    fn converse_response_no_service_tier_means_no_field() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert!(out.get("service_tier").is_none());
    }

    // ========================================================================
    // M7: trace surfaced from Converse response
    // ========================================================================

    /// Guardrail trace block is forwarded verbatim as top-level `trace`.
    #[test]
    fn converse_response_trace_surfaced() {
        let body = json!({
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "trace": {"guardrail": {"input_assessment": {"some_policy": {}}}},
        });
        let out = converse::transform_response_converse(&body, "m").expect("ok");
        assert!(out.get("trace").is_some());
        assert_eq!(
            out["trace"]["guardrail"]["input_assessment"]["some_policy"],
            json!({})
        );
    }

    // ========================================================================
    // Parallel-tool suppression: model-specific for Claude, rejected elsewhere
    // ========================================================================

    /// Non-Claude families have no serial-tool field on Converse, so the
    /// request is rejected rather than silently becoming a no-op.
    #[test]
    fn converse_request_parallel_tool_calls_false_is_rejected_for_non_claude() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "parallel_tool_calls": false,
        });
        assert!(matches!(
            converse::transform_request_converse(&req, "m", &empty_extra()),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }

    /// Claude on Converse expresses serial tool use through the model-specific
    /// `additionalModelRequestFields.tool_choice` (Anthropic's field, with the
    /// required `type` discriminator); the native Converse choice is removed
    /// so the request does not carry two `tool_choice` encodings.
    #[test]
    fn converse_request_parallel_tool_calls_false_moves_choice_for_claude() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "parallel_tool_calls": false,
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            &empty_extra(),
        )
        .expect("ok");
        assert_eq!(
            out["additionalModelRequestFields"]["tool_choice"],
            json!({"type": "auto", "disable_parallel_tool_use": true})
        );
        assert!(
            out.get("toolConfig")
                .and_then(|config| config.get("toolChoice"))
                .is_none(),
            "the native Converse choice must be removed once moved"
        );
    }

    /// `parallel_tool_calls: true` (the default) does NOT inject the flag.
    #[test]
    fn converse_request_parallel_tool_calls_true_no_injection() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "parallel_tool_calls": true,
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        let has_flag = out
            .get("toolConfig")
            .and_then(|t| t.get("toolChoice"))
            .and_then(|c| {
                let obj = c.as_object()?;
                obj.values()
                    .next()?
                    .as_object()?
                    .get("disableParallelToolUse")
            })
            .is_some();
        assert!(
            !has_flag,
            "parallel_tool_calls:true must not inject the flag"
        );
    }

    /// A forced tool survives the move for Claude: Anthropic receives the
    /// explicit `type` plus the disable flag.
    #[test]
    fn converse_request_parallel_tool_calls_false_preserves_explicit_choice() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "required"},
            "parallel_tool_calls": false,
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            &empty_extra(),
        )
        .expect("ok");
        assert_eq!(
            out["additionalModelRequestFields"]["tool_choice"],
            json!({"type": "any", "disable_parallel_tool_use": true})
        );
    }

    /// Non-Claude models still cannot express serial-tool use, even with an
    /// explicit choice.
    #[test]
    fn converse_request_parallel_tool_calls_false_is_rejected_with_explicit_choice() {
        let req = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "required"},
            "parallel_tool_calls": false,
        });
        assert!(matches!(
            converse::transform_request_converse(&req, "m", &empty_extra()),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }

    // ========================================================================
    // M8: auto-calculate max_tokens when thinking enabled
    // ========================================================================

    /// When thinking is enabled (via reasoning_effort) and no maxTokens is
    /// set, inferenceConfig.maxTokens is auto-set to budget + 4096.
    #[test]
    fn converse_request_max_tokens_auto_calculated_when_thinking_enabled() {
        let req = json!({
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "low",
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-7-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        // low → budget 1024 → maxTokens = 1024 + 4096 = 5120
        assert_eq!(
            out["additionalModelRequestFields"]["thinking"]["budget_tokens"],
            1024
        );
        assert_eq!(out["inferenceConfig"]["maxTokens"], 5120);
    }

    /// When max_tokens is explicitly set, it is NOT overridden by auto-calc.
    #[test]
    fn converse_request_max_tokens_not_overridden_when_explicitly_set() {
        let req = json!({
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "high",
            "max_tokens": 20000,
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-7-sonnet",
            &empty_extra(),
        )
        .expect("ok");
        assert_eq!(out["inferenceConfig"]["maxTokens"], 20000);
    }

    /// When thinking is absent, no auto maxTokens is injected.
    #[test]
    fn converse_request_no_thinking_means_no_auto_max_tokens() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
        });
        let out = converse::transform_request_converse(&req, "m", &empty_extra()).expect("ok");
        assert!(out["inferenceConfig"].get("maxTokens").is_none());
    }

    /// A hostile `budget_tokens` near `u64::MAX` must NOT cause an arithmetic
    /// overflow panic in `auto_set_max_tokens_for_thinking`. The transform
    /// saturates the addition instead. `budget_tokens` reaches this path
    /// verbatim via `PASS_THROUGH_TOP_LEVEL_KEYS`, so this is caller-supplied
    /// input — panicking on it would violate the no-panic-on-user-input rule.
    #[test]
    fn converse_request_max_tokens_saturates_on_huge_budget_tokens() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "additionalModelRequestFields": {
                "thinking": {"type": "enabled", "budget_tokens": u64::MAX}
            },
        });
        let out = converse::transform_request_converse(
            &req,
            "anthropic.claude-3-7-sonnet",
            &empty_extra(),
        )
        .expect("transform must succeed without panicking");
        // budget.saturating_add(4096) == u64::MAX
        assert_eq!(out["inferenceConfig"]["maxTokens"], u64::MAX);
    }
}

// =============================================================================
// Feature: per-vendor InvokeModel request/response transforms.
//
// Non-Anthropic Bedrock models each accept a vendor-native InvokeModel body
// (Titan's `inputText`/`textGenerationConfig`, Llama's `prompt`/`max_gen_len`,
// AI21's camelCase params, Qwen's ChatML prompt template, …). The provider
// detects the vendor from the model id and dispatches to the matching
// transform; responses flow back through `extract_vendor_text` plus any
// vendor-specific post-processing (DeepSeek `<think>` extraction, Qwen
// ChatML token stripping).
// =============================================================================

/// Titan: messages → `inputText`; inference params nest under
/// `textGenerationConfig` with camelCase keys.
#[test]
fn transform_request_titan_maps_messages_to_input_text_and_text_gen_config() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/amazon.titan-text-express-v1");
    let req = json!({
        "model": "amazon.titan-text-express-v1",
        "messages": [{"role": "user", "content": "Hello Titan"}],
        "max_tokens": 128,
        "temperature": 0.5,
        "top_p": 0.9,
        "stop": ["\n\n"],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["inputText"], "Hello Titan\n");
    let tgc = &out["textGenerationConfig"];
    assert_eq!(tgc["maxTokenCount"], 128);
    assert_eq!(tgc["temperature"], 0.5);
    assert_eq!(tgc["topP"], 0.9);
    assert_eq!(tgc["stopSequences"][0], "\n\n");
    // No OpenAI passthrough leakage.
    assert!(out.get("messages").is_none());
    assert!(out.get("model").is_none());
    assert!(out.get("anthropic_version").is_none());
}

/// Llama: messages → `prompt`; OpenAI `max_tokens` maps to `max_gen_len`.
#[test]
fn transform_request_llama_maps_messages_to_prompt_and_max_gen_len() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/meta.llama3-1-70b-instruct-v1:0");
    let req = json!({
        "model": "meta.llama3-1-70b-instruct-v1:0",
        "messages": [{"role": "user", "content": "Hello Llama"}],
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    // Llama uses the Llama-3 chat template — not plain concatenation.
    assert_eq!(
        out["prompt"],
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello Llama<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    assert_eq!(out["max_gen_len"], 256);
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["top_p"], 0.95);
    // No OpenAI max_tokens passthrough (renamed to max_gen_len).
    assert!(out.get("max_tokens").is_none());
    assert!(out.get("messages").is_none());
}

/// Mistral: messages → `prompt`; standard OpenAI-style param names.
#[test]
fn transform_request_mistral_maps_messages_to_prompt_with_standard_params() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/mistral.mistral-7b-instruct-v0:2");
    let req = json!({
        "model": "mistral.mistral-7b-instruct-v0:2",
        "messages": [{"role": "user", "content": "Bonjour"}],
        "max_tokens": 64,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 40,
        "stop": ["</s>"],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["prompt"], "Bonjour\n");
    assert_eq!(out["max_tokens"], 64);
    assert_eq!(out["temperature"], 0.3);
    assert_eq!(out["top_p"], 0.8);
    assert_eq!(out["top_k"], 40);
    assert_eq!(out["stop"][0], "</s>");
}

/// AI21: messages → `prompt`; camelCase param names (`maxTokens`, `topP`,
/// `frequencyPenalty`, `presencePenalty`).
#[test]
fn transform_request_ai21_uses_camel_case_param_names() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/ai21.j2-ultra-v1:0");
    let req = json!({
        "model": "ai21.j2-ultra-v1:0",
        "messages": [{"role": "user", "content": "Hello AI21"}],
        "max_tokens": 32,
        "temperature": 0.6,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["prompt"], "Hello AI21\n");
    assert_eq!(out["maxTokens"], 32);
    assert_eq!(out["temperature"], 0.6);
    assert_eq!(out["topP"], 0.9);
    assert_eq!(out["frequencyPenalty"], 0.1);
    assert_eq!(out["presencePenalty"], 0.2);
    // No snake_case leakage.
    assert!(out.get("max_tokens").is_none());
    assert!(out.get("top_p").is_none());
}

/// Cohere: messages → `prompt`; uses `p`/`k` (not `top_p`/`top_k`).
#[test]
fn transform_request_cohere_uses_p_and_k_param_names() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/cohere.command-r-v1:0");
    let req = json!({
        "model": "cohere.command-r-v1:0",
        "messages": [{"role": "user", "content": "Hello Cohere"}],
        "max_tokens": 100,
        "temperature": 0.4,
        "top_p": 0.85,
        "top_k": 50,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["prompt"], "Hello Cohere\n");
    assert_eq!(out["max_tokens"], 100);
    assert_eq!(out["temperature"], 0.4);
    assert_eq!(out["p"], 0.85);
    assert_eq!(out["k"], 50);
    // No top_p / top_k leakage.
    assert!(out.get("top_p").is_none());
    assert!(out.get("top_k").is_none());
}

/// Legacy-Mistral InvokeModel transform coverage (messages → `prompt`).
/// DeepSeek and new-generation Mistral now auto-route to Converse (verified
/// live: their InvokeModel bodies require `messages`), so this prompt-shape
/// coverage runs against a legacy vendor id that still accepts it.
#[test]
fn transform_request_deepseek_maps_messages_to_prompt_with_standard_params() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/mistral.mixtral-8x7b-instruct-v0:1");
    let req = json!({
        "model": "mistral.mixtral-8x7b-instruct-v0:1",
        "messages": [{"role": "user", "content": "Think carefully"}],
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.95,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["prompt"], "Think carefully\n");
    assert_eq!(out["max_tokens"], 512);
    assert_eq!(out["temperature"], 0.2);
    assert_eq!(out["top_p"], 0.95);
}

/// Qwen: messages → ChatML `<|im_start|>` template prompt.
#[test]
fn transform_request_qwen_uses_chatml_prompt_template() {
    let p = BedrockProvider;
    let dep = bedrock_deployment("bedrock/qwen2-72b");
    let req = json!({
        "model": "qwen2-72b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "max_tokens": 64,
        "temperature": 0.5,
        "top_p": 0.9,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    let prompt = out["prompt"].as_str().expect("prompt string");
    assert!(
        prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"),
        "system block must use ChatML form: {prompt}"
    );
    assert!(
        prompt.contains("<|im_start|>user\nHi<|im_end|>"),
        "user block must use ChatML form: {prompt}"
    );
    assert!(
        prompt.ends_with("<|im_start|>assistant\n"),
        "prompt must end with empty assistant block: {prompt}"
    );
    assert_eq!(out["max_tokens"], 64);
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["top_p"], 0.9);
}

/// Cross-region inference prefix (`us.deepseek.r1-v1:0`) is transparent to
/// vendor detection — the vendor segment can sit in any path position.
#[test]
fn detect_invoke_vendor_handles_cross_region_prefix() {
    assert_eq!(
        detect_invoke_vendor("us.deepseek.r1-v1:0"),
        InvokeVendor::DeepSeek
    );
    assert_eq!(
        detect_invoke_vendor("eu.meta.llama3-1-70b"),
        InvokeVendor::Llama
    );
    assert_eq!(
        detect_invoke_vendor("apac.mistral.mistral-large"),
        InvokeVendor::Mistral
    );
    assert_eq!(
        detect_invoke_vendor("amazon.titan-text-express-v1"),
        InvokeVendor::Titan
    );
    assert_eq!(
        detect_invoke_vendor("amazon.nova-pro-v1:0"),
        InvokeVendor::Unknown
    );
    assert_eq!(
        detect_invoke_vendor("custom.imported-model"),
        InvokeVendor::Unknown
    );
    assert_eq!(detect_invoke_vendor("qwen3-32b"), InvokeVendor::Qwen);
}

/// Titan response: `results[0].outputText` is extracted into `content`.
#[test]
fn transform_response_titan_extracts_output_text_from_results_array() {
    let p = BedrockProvider;
    let body = json!({
    "results": [{
    "outputText": "Hello from Titan",
    "tokenCount": 4,
    }],
    "inputTextTokenCount": 7,
    });
    let out = p
        .transform_response(&body, "amazon.titan-text-express-v1")
        .expect("ok");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(out["choices"][0]["message"]["content"], "Hello from Titan");
    assert_eq!(out["choices"][0]["message"]["role"], "assistant");
}

/// AI21 response: `completions[0].data.text` is extracted into `content`.
#[test]
fn transform_response_ai21_extracts_text_from_completions_data() {
    let p = BedrockProvider;
    let body = json!({
    "completions": [{
    "data": {"text": "Hello from AI21"},
    "finishReason": "STOP",
    }],
    });
    let out = p
        .transform_response(&body, "ai21.j2-ultra-v1:0")
        .expect("ok");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(out["choices"][0]["message"]["content"], "Hello from AI21");
}

/// DeepSeek response: a `<think>...</think>` block in the `generation` is
/// split into `reasoning_content` and `content`. This matches the OpenAI
/// reasoning-model convention clients expect.
#[test]
fn transform_response_deepseek_splits_think_block_into_reasoning_and_content() {
    let p = BedrockProvider;
    let body = json!({
    "generation": "<think>Reasoning about the question.</think>The answer is 42.",
    "prompt_token_count": 10,
    "generation_token_count": 8,
    });
    let out = p.transform_response(&body, "deepseek.r1-v1:0").expect("ok");
    let message = &out["choices"][0]["message"];
    assert_eq!(message["content"], "The answer is 42.");
    assert_eq!(
        message["reasoning_content"],
        "Reasoning about the question."
    );
}

/// DeepSeek response without a `<think>` block: the whole `generation`
/// becomes `content` and no `reasoning_content` field is set.
#[test]
fn transform_response_deepseek_without_think_block_returns_plain_content() {
    let p = BedrockProvider;
    let body = json!({"generation": "Plain answer."});
    let out = p.transform_response(&body, "deepseek.r1-v1:0").expect("ok");
    let message = &out["choices"][0]["message"];
    assert_eq!(message["content"], "Plain answer.");
    assert!(message.get("reasoning_content").is_none());
}

/// Vendor InvokeModel usage arithmetic saturates: adversarial
/// `u64::MAX` token counts must clamp the derived `total_tokens` to
/// `u64::MAX` rather than panic (debug) or wrap (release).
#[test]
fn transform_response_vendor_usage_arithmetic_saturates_on_overflow() {
    let p = BedrockProvider;
    let body = json!({
        "generation": "x",
        "prompt_token_count": u64::MAX,
        "generation_token_count": u64::MAX,
    });
    let out = p.transform_response(&body, "deepseek.r1-v1:0").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], u64::MAX);
    assert_eq!(out["usage"]["completion_tokens"], u64::MAX);
    assert_eq!(out["usage"]["total_tokens"], u64::MAX);
}

/// Qwen response: the leading `<|im_start|>assistant\n` prefix and the
/// trailing `<|im_end|>` suffix are stripped before the text is returned
/// as `content`.
#[test]
fn transform_response_qwen_strips_chatml_framing_tokens() {
    let p = BedrockProvider;
    let body = json!({
    "generation": "<|im_start|>assistant\nHello from Qwen!<|im_end|>",
    });
    let out = p.transform_response(&body, "qwen2-72b").expect("ok");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(out["choices"][0]["message"]["content"], "Hello from Qwen!");
}

/// Qwen response without framing tokens: text returned unchanged.
#[test]
fn transform_response_qwen_without_framing_tokens_passes_through() {
    let p = BedrockProvider;
    let body = json!({"generation": "no framing here"});
    let out = p.transform_response(&body, "qwen2-72b").expect("ok");
    assert_eq!(out["choices"][0]["message"]["content"], "no framing here");
}

/// Unknown vendor: unrecognised body shape passes through verbatim
/// (preserves the existing Nova-embedding passthrough contract).
#[test]
fn transform_response_unknown_vendor_unrecognised_body_passes_through() {
    let p = BedrockProvider;
    let body = json!({"embeddingType": "TEXT", "embeddings": [{"embedding": [0.1]}]});
    let out = p
        .transform_response(&body, "custom.imported-model-v1:0")
        .expect("ok");
    assert_eq!(out, body);
}

/// `messages_to_prompt` concatenates messages by role with simple
/// separators; system messages get a blank-line separator, user/assistant
/// messages get a single trailing newline.
#[test]
fn messages_to_prompt_concatenates_system_user_assistant_in_order() {
    let messages = json!([
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
    ])
    .as_array()
    .cloned()
    .unwrap();
    let prompt = messages_to_prompt(&messages);
    assert_eq!(prompt, "Be helpful.\n\nHi\nHello!\nBye\n");
}

/// `messages_to_prompt` skips roles it does not recognise (e.g. `tool`)
/// rather than panicking — vendor prompts only model system/user/assistant.
#[test]
fn messages_to_prompt_skips_unrecognised_roles() {
    let messages = json!([
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "tool result"},
    ])
    .as_array()
    .cloned()
    .unwrap();
    let prompt = messages_to_prompt(&messages);
    assert_eq!(prompt, "hi\n");
}

/// Multi-turn conversation: messages alternate user/assistant and the
/// prompt preserves the turn order so the model sees full history.
#[test]
fn messages_to_prompt_preserves_multi_turn_order() {
    let messages = json!([
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And 3+3?"},
    ])
    .as_array()
    .cloned()
    .unwrap();
    let prompt = messages_to_prompt(&messages);
    assert_eq!(prompt, "What is 2+2?\n4.\nAnd 3+3?\n");
}

/// `messages_to_qwen_prompt` emits a ChatML block per message, joins with
/// newlines, and ends with an empty assistant block to elicit the reply.
#[test]
fn messages_to_qwen_prompt_emits_chatml_blocks_for_each_turn() {
    let messages = json!([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ])
    .as_array()
    .cloned()
    .unwrap();
    let prompt = messages_to_qwen_prompt(&messages);
    let expected = "<|im_start|>system\nsys<|im_end|>\n\
                    <|im_start|>user\nu1<|im_end|>\n\
                    <|im_start|>assistant\na1<|im_end|>\n\
                    <|im_start|>user\nu2<|im_end|>\n\
                    <|im_start|>assistant\n";
    assert_eq!(prompt, expected);
}

/// `split_deepseek_reasoning` returns the inside of `<think>…</think>` as
/// the reasoning and the rest as content; the closing tag may be followed
/// by any amount of trailing content.
#[test]
fn split_deepseek_reasoning_extracts_inside_think_block() {
    let (reasoning, content) = split_deepseek_reasoning("<think>step 1</think>final");
    assert_eq!(reasoning.as_deref(), Some("step 1"));
    assert_eq!(content, "final");
}

/// `split_deepseek_reasoning` returns `(None, original)` when no
/// `<think>…</think>` block is present.
#[test]
fn split_deepseek_reasoning_returns_none_when_no_think_block() {
    let (reasoning, content) = split_deepseek_reasoning("just content");
    assert!(reasoning.is_none());
    assert_eq!(content, "just content");
}

/// `strip_qwen_tokens` removes both the leading assistant prefix and the
/// trailing end token when present.
#[test]
fn strip_qwen_tokens_removes_assistant_prefix_and_end_suffix() {
    let s = strip_qwen_tokens("<|im_start|>assistant\nbody text<|im_end|>");
    assert_eq!(s, "body text");
}

/// `strip_qwen_tokens` is a no-op when neither framing token is present.
#[test]
fn strip_qwen_tokens_noop_when_no_framing_present() {
    let s = strip_qwen_tokens("plain");
    assert_eq!(s, "plain");
}
