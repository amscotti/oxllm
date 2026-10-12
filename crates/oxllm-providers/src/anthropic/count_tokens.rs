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
/// blocks nested in tool_result content. Leading system messages follow the
/// caller's top-level system content. Generation-only fields are omitted.
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
    let system = lift_leading_system(&mut messages, request.get("system"))?;
    let mut body = json!({"model":model, "messages":messages});
    if let Some(system) = system {
        body["system"] = system;
    }
    for field in ["tools", "thinking", "tool_choice", "output_config"] {
        if let Some(value) = request.get(field).filter(|value| !value.is_null()) {
            body[field] = value.clone();
        }
    }
    Ok(body)
}

/// Merge a caller's system prefix ahead of system content extracted from messages.
/// A lone value keeps its native string or block-list representation.
pub fn merge_system(caller: Option<&Value>, extracted: Option<&Value>) -> Result<Option<Value>> {
    let caller = caller.filter(|value| !value.is_null());
    let extracted = extracted.filter(|value| !value.is_null());
    let (Some(caller), Some(extracted)) = (caller, extracted) else {
        return Ok(caller.or(extracted).cloned());
    };
    let mut blocks = system_blocks(caller)?;
    blocks.extend(system_blocks(extracted)?);
    Ok(Some(Value::Array(blocks)))
}

fn system_blocks(system: &Value) -> Result<Vec<Value>> {
    match system {
        Value::String(text) if text.is_empty() => Ok(Vec::new()),
        Value::String(text) => Ok(vec![json!({"type": "text", "text": text})]),
        Value::Array(blocks) => Ok(blocks.clone()),
        _ => Err(Error::BadRequest(
            "count_tokens system must be text or a block array".into(),
        )),
    }
}

fn lift_leading_system(messages: &mut Vec<Value>, system: Option<&Value>) -> Result<Option<Value>> {
    let system = system.filter(|value| !value.is_null());
    let leading = messages
        .iter()
        .take_while(|message| super::is_system_role(message))
        .count();
    // Leave a malformed native system and its messages for the provider to reject.
    if leading == 0 || system.is_some_and(|value| !value.is_string() && !value.is_array()) {
        return Ok(system.cloned());
    }
    let mut lifted = Vec::new();
    for message in messages.drain(..leading) {
        if let Some(content) = super::coerce_message_content(&message) {
            lifted.extend(system_blocks(&content)?.into_iter().filter(|block| {
                block.get("type").and_then(Value::as_str).unwrap_or("text") == "text"
                    && block
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| !text.is_empty())
            }));
        }
    }
    if lifted.is_empty() {
        return Ok(system.cloned());
    }
    merge_system(system, Some(&Value::Array(lifted)))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_system_precedes_lifted_messages_and_keeps_cache_tools_and_images() {
        let caller_blocks = json!([{
            "type": "text", "text": "Caller blocks",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }]);
        for caller in [
            None,
            Some(Value::Null),
            Some(json!("")),
            Some(json!([])),
            Some(json!("Caller text")),
            Some(caller_blocks.clone()),
        ] {
            let mut request = json!({
                "messages": [
                    {"role": "system", "content": "Lift 1", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    {"role": "system", "content": [
                        {"type": "text", "text": "Lift 2", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": "Lift 3"},
                        {"type": "text", "text": ""},
                        {"type": "tool_reference", "tool_name": "not system text"},
                    ]},
                    {"role": "system", "content": ""},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": "data:image/png;base64,QQ=="},
                        {"type": "tool_result", "tool_use_id": "call_1", "content": [
                            {"type": "image_url", "image_url": {"url": "https://images.example/image.png"},
                             "cache_control": {"type": "ephemeral"}},
                        ]},
                    ]},
                    {"role": "system", "content": "later reminder"},
                ],
                "tools": [{"name": "lookup", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}],
                "thinking": {"type": "adaptive"},
                "tool_choice": {"type": "auto"},
                "output_config": {"effort": "high"},
                "max_tokens": 32,
                "stream": true,
            });
            if let Some(caller) = &caller {
                request["system"] = caller.clone();
            }
            let original = request.clone();
            let body = normalize_request(&request, "claude-sonnet-5-5").unwrap();
            let mut expected_system = match &caller {
                Some(Value::String(text)) if !text.is_empty() => {
                    vec![json!({"type": "text", "text": text})]
                }
                Some(Value::Array(blocks)) => blocks.clone(),
                _ => Vec::new(),
            };
            expected_system.extend([
                json!({"type": "text", "text": "Lift 1", "cache_control": {"type": "ephemeral", "ttl": "1h"}}),
                json!({"type": "text", "text": "Lift 2", "cache_control": {"type": "ephemeral"}}),
                json!({"type": "text", "text": "Lift 3"}),
            ]);
            assert_eq!(body["system"], json!(expected_system));
            assert_eq!(
                body["messages"],
                json!([
                    {"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QQ=="}},
                        {"type": "tool_result", "tool_use_id": "call_1", "content": [
                            {"type": "image", "source": {"type": "url", "url": "https://images.example/image.png"},
                             "cache_control": {"type": "ephemeral"}},
                        ]},
                    ]},
                    {"role": "system", "content": "later reminder"},
                ])
            );
            for field in ["tools", "thinking", "tool_choice", "output_config"] {
                assert_eq!(body[field], request[field]);
            }
            assert_eq!(body["model"], "claude-sonnet-5-5");
            assert!(body.get("max_tokens").is_none());
            assert!(body.get("stream").is_none());
            assert_eq!(normalize_request(&body, "claude-sonnet-5-5").unwrap(), body);
            assert_eq!(request, original);
        }
    }

    #[test]
    fn leading_system_lift_preserves_text_order_and_developer_alias() {
        let request = json!({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]},
            {"role": "developer", "content": "third"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "later"},
        ]});
        let body = normalize_request(&request, "claude-sonnet-5-5").unwrap();
        let text: String = body["system"]
            .as_array()
            .unwrap()
            .iter()
            .map(|part| part["text"].as_str().unwrap())
            .collect();
        assert_eq!(text, "firstsecondthird");
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "later"},
            ])
        );
    }

    #[test]
    fn empty_leading_system_does_not_create_empty_blocks() {
        let request = json!({"messages": [
            {"role": "system", "content": ""},
            {"role": "system", "content": [{"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": "hello"},
        ]});
        let body = normalize_request(&request, "claude-sonnet-5-5").unwrap();
        assert!(body.get("system").is_none());
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": "hello"}])
        );
    }

    #[test]
    fn native_system_shape_without_a_leading_run_stays_unchanged() {
        for system in [
            json!("caller text"),
            json!([{"type": "text", "text": "caller blocks", "cache_control": {"type": "ephemeral"}}]),
        ] {
            let request = json!({"system": system, "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "later"},
            ]});
            let body = normalize_request(&request, "claude-sonnet-5-5").unwrap();
            assert_eq!(body["system"], request["system"]);
            assert_eq!(body["messages"], request["messages"]);
        }
    }

    #[test]
    fn malformed_native_system_is_not_hidden_by_lifting() {
        for system in [json!(5), json!(false), json!({"text": "invalid wrapper"})] {
            let request = json!({"system": system, "messages": [
                {"role": "system", "content": "leading"},
                {"role": "user", "content": "hello"},
            ]});
            let body = normalize_request(&request, "claude-sonnet-5-5").unwrap();
            assert_eq!(body["system"], request["system"]);
            assert_eq!(body["messages"], request["messages"]);
        }
    }

    #[test]
    fn merging_system_strings_and_cached_blocks_keeps_the_prefix_first() {
        for caller in [
            json!("caller"),
            json!([{"type": "text", "text": "caller", "cache_control": {"type": "ephemeral"}}]),
        ] {
            for extracted in [
                json!("extracted"),
                json!([{"type": "text", "text": "extracted", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]),
            ] {
                let merged = merge_system(Some(&caller), Some(&extracted))
                    .unwrap()
                    .unwrap();
                assert_eq!(merged[0]["text"], "caller");
                assert_eq!(merged[1]["text"], "extracted");
                assert_eq!(merged.as_array().unwrap().len(), 2);
                if caller.is_array() {
                    assert_eq!(merged[0], caller[0]);
                }
                if extracted.is_array() {
                    assert_eq!(merged[1], extracted[0]);
                }
            }
        }
        assert!(matches!(
            merge_system(Some(&json!(5)), Some(&json!("extracted"))),
            Err(Error::BadRequest(_))
        ));
    }

    #[test]
    fn leading_system_lift_does_not_bypass_message_or_tool_validation() {
        for request in [
            json!({"messages": [{"role": "system", "content": "leading"}, 5]}),
            json!({"messages": [{"role": "system", "content": "leading"}], "tools": ["invalid"]}),
            json!({"messages": [{"role": "system", "content": "leading"}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png,payload"}}]}]}),
        ] {
            assert!(matches!(
                normalize_request(&request, "claude-sonnet-5-5"),
                Err(Error::BadRequest(_))
            ));
        }
    }
}
