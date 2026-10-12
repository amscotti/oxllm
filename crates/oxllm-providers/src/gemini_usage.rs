//! Proof-gated usage repair for Google Gemini's compatible API surfaces.

use serde_json::{Value, json};

/// Fold reasoning only when integer counters prove it was excluded from
/// completion. Inconsistent/unknown shapes retain their sanitized counters.
pub(crate) fn normalize_gemini_usage(body: &mut Value) {
    crate::openai::sanitize_null_tokens(body);
    let Some(usage) = body.get_mut("usage").and_then(Value::as_object_mut) else {
        return;
    };
    let Some(prompt) = usage.get("prompt_tokens").and_then(Value::as_u64) else {
        return;
    };
    let Some(total) = usage.get("total_tokens").and_then(Value::as_u64) else {
        return;
    };
    let missing_completion = !usage.contains_key("completion_tokens");
    let completion = match usage.get("completion_tokens") {
        None => 0,
        Some(value) => {
            let Some(completion) = value.as_u64() else {
                return;
            };
            completion
        }
    };
    let reasoning_detail = usage
        .get("completion_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"));
    let zero_reasoning = reasoning_detail.is_none_or(|value| value.as_u64() == Some(0));
    let reasoning = reasoning_detail.and_then(Value::as_u64);
    if let Some(reasoning) = reasoning {
        if prompt
            .checked_add(completion)
            .and_then(|tokens| tokens.checked_add(reasoning))
            == Some(total)
        {
            if let Some(completion) = completion.checked_add(reasoning) {
                usage.insert("completion_tokens".into(), json!(completion));
                return;
            }
        }
    }
    if missing_completion && total == prompt && zero_reasoning {
        usage.insert("completion_tokens".into(), json!(0));
    }
}
