//! Span-aware handling of consecutive user/tool messages.

use std::collections::HashSet;

use oxllm_core::{Error, Result};
use serde_json::{Value, json};

use super::{has_cache_control, placeholder_text_block};

/// Keep marked empty users in source order, unless a correlated tool result
/// occurs later in this same run (reference `_cache_breakpoint_placeholder`).
/// Unmarked empty siblings contribute nothing; a wholly empty run still gets
/// one placeholder. Output configuration must never disappear with a turn.
///
/// This is linear in the run's messages and content blocks: a single forward
/// scan records the last correlated result, then a second pass emits messages.
/// There is no per-message suffix lookahead or sorting. Stable partitioning
/// keeps tool results before ordinary user content without touching assistants.
pub(super) fn flush_user_run(
    run: &mut Vec<(&Value, Option<Value>)>,
    out: &mut Vec<Value>,
    pending_tool_ids: &mut HashSet<String>,
) -> Result<()> {
    if run.is_empty() {
        return Ok(());
    }
    let last_result = last_correlated_tool_result(
        run.iter().map(|(_, normalized)| normalized.as_ref()),
        pending_tool_ids,
    );
    let start = out.len();
    let mut ordinary = Vec::new();
    let mut ordinary_has_config = false;
    for (index, (source, normalized)) in run.drain(..).enumerate() {
        if let Some(message) = normalized {
            partition_message(message, out, &mut ordinary, &mut ordinary_has_config)?;
            continue;
        }
        let config = source.get("output_config").filter(|value| !value.is_null());
        if last_result.is_some_and(|last| index < last) {
            if config.is_some() {
                return Err(Error::BadRequest(
                    "cannot preserve message-level output_config on an empty user turn before a later correlated tool_result in the same turn".into(),
                ));
            }
            continue;
        }
        if has_cache_control(source) || config.is_some() {
            let mut message = json!({
                "role": "user",
                "content": [placeholder_text_block(source.get("cache_control").cloned())],
            });
            if let Some(config) = config {
                message["output_config"] = config.clone();
                ordinary_has_config = true;
            }
            ordinary.push(message);
        }
    }
    if out.len() == start && ordinary.is_empty() {
        ordinary.push(json!({
            "role": "user",
            "content": [placeholder_text_block(None)],
        }));
    }
    out.extend(ordinary);
    Ok(())
}

fn is_tool_result(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("tool_result")
}

/// An orphaned/malformed result is left for provider validation, but must not
/// erase a caller's breakpoint. Consuming ids also avoids treating a duplicate
/// result as a second valid correlation. The caller resets ids at boundaries,
/// including assistants whose empty content is otherwise dropped.
fn last_correlated_tool_result<'a>(
    messages: impl Iterator<Item = Option<&'a Value>>,
    pending_tool_ids: &mut HashSet<String>,
) -> Option<usize> {
    let mut last = None;
    for (index, message) in messages.enumerate() {
        if let Some(blocks) = message
            .and_then(|message| message.get("content"))
            .and_then(Value::as_array)
        {
            for block in blocks {
                if is_tool_result(block)
                    && let Some(id) = block.get("tool_use_id").and_then(Value::as_str)
                    && pending_tool_ids.remove(id)
                {
                    last = Some(index);
                }
            }
        }
    }
    last
}

fn partition_message(
    mut message: Value,
    results: &mut Vec<Value>,
    ordinary: &mut Vec<Value>,
    ordinary_has_config: &mut bool,
) -> Result<()> {
    let configured = message.get("output_config").is_some();
    let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
        *ordinary_has_config |= configured;
        ordinary.push(message);
        return Ok(());
    };
    if !blocks.iter().any(is_tool_result) {
        *ordinary_has_config |= configured;
        ordinary.push(message);
        return Ok(());
    }
    // Moving a result across ordinary content must not move either source's
    // configuration event. Compare presence, not formats/values: effort and
    // future native options matter even when formats are identical or absent.
    if *ordinary_has_config || configured && !ordinary.is_empty() {
        return Err(Error::BadRequest(
            "cannot reorder message-level output_config across tool_result and ordinary user content; send tool results before configured user messages".into(),
        ));
    }
    let (tool_blocks, ordinary_blocks): (Vec<_>, Vec<_>) =
        blocks.drain(..).partition(is_tool_result);
    if !ordinary_blocks.is_empty() {
        if message.get("output_config").is_some() {
            // Do not duplicate one source constraint across two native turns,
            // or arbitrarily choose which half should inherit it.
            return Err(Error::BadRequest(
                "cannot split tool_result and ordinary user content while preserving message-level output_config; send separate messages".into(),
            ));
        }
        ordinary.push(json!({"role": "user", "content": ordinary_blocks}));
    }
    message["content"] = Value::Array(tool_blocks);
    results.push(message);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn result_scan_visits_long_empty_suffixes_exactly_once() {
        const LEN: usize = 65_536;
        let visits = Cell::new(0);
        let result = json!({"content": [{"type": "tool_result", "tool_use_id": "call_1"}]});
        let mut ids = HashSet::from(["call_1".into()]);
        let input = std::iter::repeat_n(None, LEN)
            .chain(std::iter::once(Some(&result)))
            .chain(std::iter::repeat_n(None, LEN))
            .inspect(|_| visits.set(visits.get() + 1));
        assert_eq!(last_correlated_tool_result(input, &mut ids), Some(LEN));
        assert_eq!(visits.get(), 2 * LEN + 1);
        assert!(ids.is_empty());
    }
}
