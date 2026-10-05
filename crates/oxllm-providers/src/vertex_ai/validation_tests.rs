use super::*;

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
