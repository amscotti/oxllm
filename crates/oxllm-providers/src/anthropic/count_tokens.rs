//! Native Count Tokens body shaping, including OpenAI image parts in tool results.

use oxllm_core::{Error, Result};
use serde_json::{Value, json};

/// Validate token-counting tools without accepting malformed lists as empty.
pub fn validate_tools(request: &Value) -> Result<()> {
    if let Some(tools) = request.get("tools").filter(|value| !value.is_null()) {
        if tools
            .as_array()
            .is_none_or(|tools| tools.iter().any(|tool| !tool.is_object()))
        {
            return Err(Error::BadRequest(
                "count_tokens tools must be an array of objects".into(),
            ));
        }
    }
    Ok(())
}

/// Preserve native system/tools fields and normalize image_url blocks, including
/// blocks nested in tool_result content. Generation-only fields are omitted.
pub fn normalize_request(request: &Value, model: &str) -> Result<Value> {
    validate_tools(request)?;
    let messages = request
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::BadRequest("count_tokens requires a messages array".into()))?;
    let mut messages = messages.clone();
    for message in &mut messages {
        if !message.is_object() {
            return Err(Error::BadRequest(
                "count_tokens messages must contain objects".into(),
            ));
        }
        if let Some(content) = message.get_mut("content") {
            normalize_content(content)?;
        }
    }
    let mut body = json!({"model":model, "messages":messages});
    for field in [
        "system",
        "tools",
        "thinking",
        "tool_choice",
        "output_config",
    ] {
        if let Some(value) = request.get(field).filter(|value| !value.is_null()) {
            body[field] = value.clone();
        }
    }
    Ok(body)
}

fn normalize_content(content: &mut Value) -> Result<()> {
    if let Some(blocks) = content.as_array_mut() {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                if let Some(content) = block.get_mut("content") {
                    normalize_content(content)?;
                }
                continue;
            }
            if block.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }
            let url = block
                .get("image_url")
                .and_then(|image| {
                    image
                        .as_str()
                        .or_else(|| image.get("url").and_then(Value::as_str))
                })
                .ok_or_else(|| Error::BadRequest("count_tokens image_url requires a URL".into()))?;
            let image = crate::util::data_url_to_anthropic_image(url)
                .or_else(|| {
                    (url.starts_with("https://") || url.starts_with("http://"))
                        .then(|| json!({"type":"image", "source":{"type":"url", "url":url}}))
                })
                .ok_or_else(|| {
                    Error::BadRequest(
                        "count_tokens image requires an HTTP URL or base64 data URL".into(),
                    )
                })?;
            if let Some(object) = block.as_object_mut() {
                object.remove("image_url");
                object.insert("type".into(), image["type"].clone());
                object.insert("source".into(), image["source"].clone());
            }
        }
    }
    Ok(())
}
