//! Token-usage extraction helpers shared by the chat orchestrator and the
//! cache-hit path.
//!
//! Centralising these avoids drift between the non-streaming, streaming, and
//! cache-short-circuit code paths that all read usage off a chat completion
//! body.

use serde_json::Value;

/// Extract token usage from a chat completion response body.
///
/// Conventions are normalised here so the rest of the pipeline (notably
/// `calculate_cost`) can assume a single invariant: `cache_read_input_tokens`
/// is a SUBSET of `prompt_tokens`.
///
/// - OpenAI reports cache hits as `prompt_tokens_details.cached_tokens` (Chat)
///   or `input_tokens_details.cached_tokens` (Responses),
///   already counted inside `prompt_tokens` — left as-is.
/// - Anthropic reports cache hits as a top-level `cache_read_input_tokens`
///   that is ADDITIVE to `input_tokens` (Anthropic's `input_tokens` excludes
///   cache hits). We fold the cache-read tokens into `prompt_tokens` so the
///   returned [`UsageSummary`] obeys the subset invariant.
pub(crate) fn extract_usage(value: &Value) -> Option<oxllm_core::UsageSummary> {
    let usage = value.get("usage")?.as_object()?;
    // Missing usage is not a provider-reported zero. Only numeric counters
    // disable estimation; null placeholders and metadata-only objects do not.
    if ![
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ]
    .iter()
    .any(|field| usage.get(*field).and_then(Value::as_u64).is_some())
        && ![
            "prompt_tokens_details",
            "input_tokens_details",
            "completion_tokens_details",
            "cache_creation",
            "output_tokens_details",
        ]
        .iter()
        .any(|field| {
            usage
                .get(*field)
                .and_then(Value::as_object)
                .is_some_and(|details| details.values().any(|value| value.as_u64().is_some()))
        })
    {
        return None;
    }
    let prompt = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| usage.get("input_tokens").and_then(|v| v.as_u64()))
        .or_else(|| {
            usage
                .get("total_tokens")
                .and_then(Value::as_u64)
                .map(|total| {
                    let completion = usage
                        .get("completion_tokens")
                        .or_else(|| usage.get("output_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    total.saturating_sub(completion)
                })
        })
        .unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| usage.get("output_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(0);

    // OpenAI Chat/Responses detail buckets are already subsets of the input.
    let input_bucket = |field: &str| {
        usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get(field))
            .and_then(Value::as_u64)
            .or_else(|| {
                usage
                    .get("input_tokens_details")
                    .and_then(|details| details.get(field))
                    .and_then(Value::as_u64)
            })
    };
    let openai_cached = input_bucket("cached_tokens");

    // Anthropic: top-level cache_read_input_tokens, ADDITIVE to prompt_tokens.
    let anthropic_cache_read = usage
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64());

    // Normalise to the subset invariant. The OpenAI field wins when both are
    // present (it is already folded into prompt_tokens by the provider).
    let (prompt_normalized, cache_read_input_tokens) = match (openai_cached, anthropic_cache_read) {
        (Some(cached), _) => (prompt, Some(cached)),
        (None, Some(cache_read)) => (prompt.saturating_add(cache_read), Some(cache_read)),
        (None, None) => (prompt, None),
    };

    // Cache-creation (cache-write) tokens arrive under two provider conventions:
    //   - Anthropic top-level `cache_creation_input_tokens`, which is ADDITIVE
    //     to `prompt_tokens` (Anthropic's `input_tokens` excludes cache writes).
    //   - OpenAI `prompt_tokens_details.cache_write_tokens` /
    //     `cache_creation_tokens`, which is a SUBSET of `prompt_tokens` (every
    //     field under `prompt_tokens_details` is already counted inside
    //     `prompt_tokens` by OpenAI convention).
    // `cost_from_rates` assumes cache-creation is ADDITIVE — it bills
    // `prompt_tokens * input_rate` and then a separate
    // `cache_creation * rate`. To preserve that invariant we therefore SUBTRACT
    // the OpenAI subset form from `prompt_tokens` (mirroring how the Anthropic
    // additive cache-read is folded in above). Without this normalisation the
    // OpenAI cache-write tokens would be charged once inside `prompt_tokens`
    // and again in the cache-creation bucket — a latent 2x over-bill.
    let anthropic_cache_creation = usage
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64());
    let openai_cache_creation =
        input_bucket("cache_write_tokens").or_else(|| input_bucket("cache_creation_tokens"));
    let (prompt_normalized, cache_creation_input_tokens) =
        match (anthropic_cache_creation, openai_cache_creation) {
            // Anthropic additive form wins when present (already outside prompt).
            (Some(cc), _) => (prompt_normalized, Some(cc)),
            // OpenAI subset form: restore the additive invariant by removing the
            // cache-write tokens from `prompt_tokens`.
            (None, Some(cc)) => (prompt_normalized.saturating_sub(cc), Some(cc)),
            (None, None) => (prompt_normalized, None),
        };
    // Anthropic 1h-ephemeral cache-creation TTL bucket. The Anthropic
    // provider mirrors `usage.cache_creation.ephemeral_1h_input_tokens` into
    // `prompt_tokens_details.ephemeral_1h_input_tokens`; it is a subset of
    // `cache_creation_input_tokens` and bills at a distinct (usually higher)
    // rate, so it must survive extraction for `cost_from_rates`.
    let cache_creation_1h_input_tokens = input_bucket("ephemeral_1h_input_tokens").or_else(|| {
        usage
            .get("cache_creation")
            .and_then(|details| details.get("ephemeral_1h_input_tokens"))
            .and_then(Value::as_u64)
    });
    let audio_input_tokens = input_bucket("audio_tokens")
        .or_else(|| usage.get("audio_input_tokens").and_then(|v| v.as_u64()));
    // Reasoning is an output subset: Chat uses completion_tokens_details,
    // Responses uses output_tokens_details, and Anthropic calls it thinking.
    // A separate configured rate must survive both native and converted paths.
    let reasoning_tokens = usage
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|v| v.as_u64())
        .or_else(|| {
            usage
                .get("output_tokens_details")
                .and_then(|details| details.get("reasoning_tokens"))
                .and_then(Value::as_u64)
        })
        .or_else(|| {
            usage
                .get("output_tokens_details")
                .and_then(|details| details.get("thinking_tokens"))
                .and_then(Value::as_u64)
        });

    // Citation tokens (Perplexity Sonar/online). Perplexity does not report a
    // pre-counted value, so we derive an estimate from the response `citations`
    // array length when present — matching the reference
    // `_enhance_usage_with_perplexity_fields` heuristic (~1 token per
    // citation, floor 1 when citations exist). A provider transform may also
    // surface a precise count under `usage.citation_tokens`, which wins.
    let citation_tokens = usage
        .get("citation_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            value
                .get("citations")
                .and_then(|c| c.as_array())
                .map(|c| c.len() as u64)
                .filter(|n| *n > 0)
        });

    // Web-search requests charged for the call. Read from the OpenAI-standard
    // `prompt_tokens_details.web_search_requests` (OpenAI/xAI web-search) or
    // Perplexity's `usage.num_search_queries` / top-level `search_queries`.
    // Anthropic-family hosted web search reports per-tool counts under
    // `usage.server_tool_use` (mirrored into the chat shape by the Anthropic
    // transform), and native Responses bodies report tool usage under
    // `tool_usage` (top-level or inside `usage`) — all map onto the same
    // per-query billing bucket.
    let web_search_requests = input_bucket("web_search_requests")
        .or_else(|| usage.get("num_search_queries").and_then(|v| v.as_u64()))
        .or_else(|| value.get("num_search_queries").and_then(|v| v.as_u64()))
        .or_else(|| usage.get("search_queries").and_then(|v| v.as_u64()))
        .or_else(|| {
            usage
                .get("server_tool_use")
                .and_then(|tool| tool.get("web_search_requests"))
                .and_then(Value::as_u64)
        })
        .or_else(|| {
            value
                .get("tool_usage")
                .or_else(|| usage.get("tool_usage"))
                .and_then(|tool| tool.get("web_search"))
                .and_then(|web| web.get("num_requests"))
                .and_then(Value::as_u64)
        });

    // Groq `browser.open` executed actions, mirrored by the Groq transform
    // under the same `server_tool_use` bucket. Only that transform ever sets
    // the counter; it bills at Groq's flat provider rate.
    let browser_open_requests = usage
        .get("server_tool_use")
        .and_then(|tool| tool.get("browser_open_requests"))
        .and_then(Value::as_u64);

    Some(oxllm_core::UsageSummary {
        prompt_tokens: prompt_normalized,
        completion_tokens: completion,
        // Always recompute from the normalised prompt so `total_tokens`
        // honours the `total == prompt + completion` invariant even when the
        // upstream-supplied total was computed against the pre-normalisation
        // prompt (e.g. it excluded Anthropic cache-read tokens folded in
        // above, or still included OpenAI cache-write tokens subtracted out).
        // `record_deployment_tpm` reads this field, so an inconsistent value
        // would enforce deployment TPM limits against the wrong count.
        // Saturating add: these fields come from provider-supplied JSON; a
        // value near `u64::MAX` must not panic debug builds (DoS) or wrap
        // release builds (bogus TPM enforcement).
        total_tokens: prompt_normalized.saturating_add(completion),
        cache_read_input_tokens,
        cache_creation_input_tokens,
        cache_creation_1h_input_tokens,
        audio_input_tokens,
        reasoning_tokens,
        citation_tokens,
        web_search_requests,
        browser_open_requests,
    })
}

/// Merge an incoming usage snapshot into the accumulator. The stream emits a
/// running usage tally; we keep the max of each field and recompute the
/// total from the prompt/completion maxima. Subset buckets (cache-read,
/// cache-creation, audio, reasoning, and the 1h cache-creation TTL split)
/// are monotonic running tallies too, so the larger value wins whenever a
/// provider re-emits one — a later chunk reporting fewer tokens than an
/// earlier one is a protocol violation and must not shrink the bill.
pub(crate) fn merge_streaming_usage(
    acc: &mut Option<oxllm_core::UsageSummary>,
    incoming: oxllm_core::UsageSummary,
) {
    match acc {
        None => {
            *acc = Some(incoming);
        }
        Some(existing) => {
            existing.prompt_tokens = existing.prompt_tokens.max(incoming.prompt_tokens);
            existing.completion_tokens = existing.completion_tokens.max(incoming.completion_tokens);
            existing.total_tokens = existing
                .prompt_tokens
                .saturating_add(existing.completion_tokens);
            merge_max_bucket(
                &mut existing.cache_read_input_tokens,
                incoming.cache_read_input_tokens,
            );
            merge_max_bucket(
                &mut existing.cache_creation_input_tokens,
                incoming.cache_creation_input_tokens,
            );
            merge_max_bucket(
                &mut existing.cache_creation_1h_input_tokens,
                incoming.cache_creation_1h_input_tokens,
            );
            merge_max_bucket(
                &mut existing.audio_input_tokens,
                incoming.audio_input_tokens,
            );
            // H6: reasoning tokens arrive once in the final usage chunk from
            // `stream_options.include_usage`; keep the larger value so the
            // aggregated snapshot surfaces it to cost / billing.
            merge_max_bucket(&mut existing.reasoning_tokens, incoming.reasoning_tokens);
            merge_max_bucket(&mut existing.citation_tokens, incoming.citation_tokens);
            merge_max_bucket(
                &mut existing.web_search_requests,
                incoming.web_search_requests,
            );
            merge_max_bucket(
                &mut existing.browser_open_requests,
                incoming.browser_open_requests,
            );
        }
    }
}

/// Keep the larger of two optional running-token tallies. A `None` incoming
/// value never clobbers an existing one.
fn merge_max_bucket(slot: &mut Option<u64>, incoming: Option<u64>) {
    let Some(new) = incoming else {
        return;
    };
    match slot {
        Some(existing) => *existing = (*existing).max(new),
        None => *slot = Some(new),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_usage_handles_full_object() {
        let v = json!({
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn extract_usage_returns_none_when_missing() {
        assert!(extract_usage(&json!({"foo": "bar"})).is_none());
    }

    #[test]
    fn placeholders_do_not_disable_usage_estimation() {
        for usage in [
            Value::Null,
            json!({}),
            json!({"prompt_tokens":null,"completion_tokens":null}),
            json!({"speed":"fast","cost":0.25}),
        ] {
            assert!(extract_usage(&json!({"usage":usage})).is_none());
        }
        let zero = extract_usage(&json!({"usage":{"prompt_tokens":0,"completion_tokens":0}}))
            .expect("explicit zero counters");
        assert_eq!(zero.total_tokens, 0);
    }

    #[test]
    fn native_usage_retains_one_hour_cache_writes_and_thinking() {
        let usage = extract_usage(&json!({"usage":{
            "input_tokens":10,"output_tokens":20,"cache_creation_input_tokens":50,
            "cache_creation":{"ephemeral_1h_input_tokens":40},
            "output_tokens_details":{"thinking_tokens":12}
        }}))
        .expect("native usage");
        assert_eq!(usage.cache_creation_1h_input_tokens, Some(40));
        assert_eq!(usage.reasoning_tokens, Some(12));
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn anthropic_server_tool_use_web_search_is_billed_per_query() {
        let usage = extract_usage(&json!({"usage":{
            "input_tokens":100,"output_tokens":10,
            "server_tool_use":{"web_search_requests":3,"tool_search_requests":2}
        }}))
        .expect("native usage");
        assert_eq!(usage.web_search_requests, Some(3));
    }

    #[test]
    fn groq_server_tool_use_browser_opens_map_to_their_own_bucket() {
        // Groq's transform mirrors executed `browser.open` actions beside the
        // search count; the extraction must land them in their own bucket so
        // the flat provider rate bills them.
        let usage = extract_usage(&json!({"usage":{
            "input_tokens":100,"output_tokens":10,
            "server_tool_use":{"web_search_requests":3,"browser_open_requests":4}
        }}))
        .expect("native usage");
        assert_eq!(usage.web_search_requests, Some(3));
        assert_eq!(usage.browser_open_requests, Some(4));
    }

    #[test]
    fn responses_tool_usage_web_search_is_billed_per_query() {
        for body in [
            json!({"usage":{"input_tokens":100,"output_tokens":10,
                "tool_usage":{"web_search":{"num_requests":4}}}}),
            json!({"tool_usage":{"web_search":{"num_requests":4}},
                "usage":{"input_tokens":100,"output_tokens":10}}),
        ] {
            let usage = extract_usage(&body).expect("native usage");
            assert_eq!(usage.web_search_requests, Some(4));
        }
    }

    #[test]
    fn responses_usage_retains_cached_write_audio_and_reasoning_buckets() {
        let usage = extract_usage(&json!({"usage": {
            "input_tokens":1000, "output_tokens":80, "total_tokens":1080,
            "input_tokens_details":{"cached_tokens":300,"cache_write_tokens":200,"audio_tokens":10},
            "output_tokens_details":{"reasoning_tokens":60}
        }}))
        .expect("native Responses usage");
        assert_eq!(
            usage.prompt_tokens, 800,
            "writes are normalized to the additive bucket"
        );
        assert_eq!(usage.cache_read_input_tokens, Some(300));
        assert_eq!(usage.cache_creation_input_tokens, Some(200));
        assert_eq!(usage.audio_input_tokens, Some(10));
        assert_eq!(usage.reasoning_tokens, Some(60));
        assert_eq!(usage.total_tokens, 880);
    }

    #[test]
    fn extract_usage_computes_total_when_missing() {
        let v = json!({"usage": {"prompt_tokens": 10, "completion_tokens": 5}});
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn extract_usage_reads_cached_tokens_from_details() {
        let v = json!({
        "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 10,
        "total_tokens": 1010,
        "prompt_tokens_details": {"cached_tokens": 400}
        }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.cache_read_input_tokens, Some(400));
    }

    #[test]
    fn extract_usage_reads_cache_creation_tokens() {
        let v = json!({
        "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cache_creation_input_tokens": 50
        }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.cache_creation_input_tokens, Some(50));
    }

    /// OpenAI reports cache-creation tokens under
    /// `prompt_tokens_details.cache_write_tokens` rather than the Anthropic
    /// top-level field. The extractor must read this location too.
    #[test]
    fn extract_usage_reads_cache_creation_from_openai_format() {
        let v = json!({
        "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cache_write_tokens": 100}
        }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(
            u.cache_creation_input_tokens,
            Some(100),
            "cache_creation must be read from prompt_tokens_details.cache_write_tokens"
        );
    }

    /// When BOTH the Anthropic top-level field and the OpenAI
    /// `prompt_tokens_details` form are present, the Anthropic field wins.
    #[test]
    fn extract_usage_prefers_anthropic_format_for_cache_creation() {
        let v = json!({
        "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cache_creation_input_tokens": 50,
        "prompt_tokens_details": {"cache_write_tokens": 100}
        }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(
            u.cache_creation_input_tokens,
            Some(50),
            "Anthropic top-level cache_creation must take precedence"
        );
    }

    /// U46: the `prompt_tokens_details.cache_creation_tokens` spelling is an
    /// alternate OpenAI-compatible alias for cache creation. It must be read
    /// when neither the Anthropic top-level field nor `cache_write_tokens`
    /// is present.
    #[test]
    fn extract_usage_reads_cache_creation_tokens_alias_from_prompt_details() {
        let v = json!({
        "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cache_creation_tokens": 77}
        }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(
            u.cache_creation_input_tokens,
            Some(77),
            "cache_creation_tokens alias in prompt_tokens_details must be read"
        );
    }

    /// Bug #2: Anthropic reports `cache_read_input_tokens` as a top-level
    /// field that is ADDITIVE to `input_tokens`. The extractor must fold it
    /// into `prompt_tokens` so the result obeys the "cache_read ⊆ prompt"
    /// invariant that `calculate_cost` relies on.
    #[test]
    fn extract_usage_normalises_anthropic_cache_read_as_subset() {
        let v = json!({
        "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cache_read_input_tokens": 1_000_000u64
        }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(
            u.prompt_tokens, 1_000_100,
            "cache_read must be folded into prompt_tokens"
        );
        assert_eq!(u.cache_read_input_tokens, Some(1_000_000));
        assert_eq!(u.completion_tokens, 50);
        assert_eq!(
            u.total_tokens, 1_000_150,
            "total must be recomputed from the normalised prompt"
        );
    }

    /// The OpenAI subset form must NOT be folded a second time — otherwise
    /// the cache hits would be double-counted inside `prompt_tokens`.
    #[test]
    fn extract_usage_does_not_refold_openai_cached_tokens() {
        let v = json!({
        "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 400}
        }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(
            u.prompt_tokens, 1000,
            "OpenAI prompt already includes cache"
        );
        assert_eq!(u.cache_read_input_tokens, Some(400));
    }

    #[test]
    fn extract_usage_reads_audio_from_prompt_details() {
        let v = json!({
        "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {"audio_tokens": 40},
        "completion_tokens_details": {"audio_tokens": 99}
        }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(
            u.audio_input_tokens,
            Some(40),
            "audio input must come from prompt_tokens_details, not completion"
        );
    }

    #[test]
    fn merge_streaming_usage_combines_prompt_and_completion_tokens() {
        let prompt_chunk = json!({
        "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42}
        });
        let completion_chunk = json!({
        "usage": {"prompt_tokens": 0, "completion_tokens": 13, "total_tokens": 13}
        });
        let mut acc: Option<oxllm_core::UsageSummary> = None;
        merge_streaming_usage(&mut acc, extract_usage(&prompt_chunk).expect("present"));
        merge_streaming_usage(&mut acc, extract_usage(&completion_chunk).expect("present"));
        let merged = acc.expect("accumulated");
        assert_eq!(merged.prompt_tokens, 42);
        assert_eq!(merged.completion_tokens, 13);
        assert_eq!(merged.total_tokens, 55);
    }

    #[test]
    fn merge_streaming_usage_takes_max_on_repeats() {
        let mut acc: Option<oxllm_core::UsageSummary> = None;
        merge_streaming_usage(
            &mut acc,
            extract_usage(&json!({"usage": {"prompt_tokens": 10, "completion_tokens": 4}}))
                .expect("present"),
        );
        merge_streaming_usage(
            &mut acc,
            extract_usage(&json!({"usage": {"prompt_tokens": 7, "completion_tokens": 20}}))
                .expect("present"),
        );
        let merged = acc.expect("accumulated");
        assert_eq!(merged.prompt_tokens, 10);
        assert_eq!(merged.completion_tokens, 20);
        assert_eq!(merged.total_tokens, 30);
    }

    // =================================================================
    // Ported from the reference test_convert_dict_to_chat_completion.py
    // Edge cases for usage extraction on provider response shapes.
    // =================================================================

    /// Ported from `_usage_format_tests` invariant in base_llm_unit_tests.py:
    /// `total_tokens == prompt_tokens + completion_tokens`.
    /// The extractor must preserve this invariant when the upstream body
    /// supplies all three fields consistently.
    #[test]
    fn extract_usage_preserves_total_equals_prompt_plus_completion() {
        let v = json!({
            "usage": {
                "prompt_tokens": 19,
                "completion_tokens": 10,
                "total_tokens": 29,
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.total_tokens, u.prompt_tokens + u.completion_tokens);
        assert_eq!(u.total_tokens, 29);
    }

    /// Ported from `test_convert_to_model_response_object_with_thinking_content`:
    /// reasoning_content / thinking blocks must NOT disturb the usage
    /// extraction. The usage object has the Anthropic-style
    /// `cache_creation_input_tokens` and `cache_read_input_tokens` alongside
    /// the standard OpenAI fields.
    #[test]
    fn extract_usage_handles_anthropic_thinking_response_shape() {
        let v = json!({
            "usage": {
                "prompt_tokens": 65,
                "completion_tokens": 460,
                "total_tokens": 525,
                "prompt_tokens_details": {"cached_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 65);
        assert_eq!(u.completion_tokens, 460);
        assert_eq!(u.total_tokens, 525);
        assert_eq!(u.cache_read_input_tokens, Some(0));
        assert_eq!(u.cache_creation_input_tokens, Some(0));
    }

    /// Ported from `test_convert_to_model_response_object_tool_calls_*`:
    /// when the upstream returns a tool-call-only response (content=null),
    /// usage extraction is unaffected — the body still carries standard
    /// usage counters that downstream code depends on for billing.
    #[test]
    fn extract_usage_handles_null_content_tool_call_response() {
        let v = json!({
            "id": "chatcmpl-toolcall",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"q\":\"Paris\"}"}
                    }],
                },
            }],
            "usage": {
                "prompt_tokens": 82,
                "completion_tokens": 17,
                "total_tokens": 99,
            },
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 82);
        assert_eq!(u.completion_tokens, 17);
        assert_eq!(u.total_tokens, 99);
    }

    /// Ported from `test_convert_to_model_response_object_with_empty_str`:
    /// a response with empty string content still has well-formed usage.
    /// The extractor must not conflate "empty content" with "no usage".
    #[test]
    fn extract_usage_handles_empty_content_response() {
        let v = json!({
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": ""},
            }],
            "usage": {
                "prompt_tokens": 121,
                "completion_tokens": 1,
                "total_tokens": 122,
            },
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 121);
        assert_eq!(u.completion_tokens, 1);
        assert_eq!(u.total_tokens, 122);
    }

    /// Ported from `test_convert_to_model_response_object_default_usage_overwritten`:
    /// the extractor returns Some even when only the standard three counters
    /// are present (no cache fields), so downstream code can rely on
    /// `extract_usage(...).is_some()` for any well-formed provider response.
    #[test]
    fn extract_usage_returns_some_for_minimal_usage_object() {
        let v = json!({
            "usage": {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22}
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 15);
        assert_eq!(u.completion_tokens, 7);
        assert_eq!(u.total_tokens, 22);
        assert!(u.cache_read_input_tokens.is_none());
        assert!(u.cache_creation_input_tokens.is_none());
        assert!(u.audio_input_tokens.is_none());
    }

    /// Ported from `test_convert_to_model_response_object_with_null_top_logprobs`:
    /// null fields elsewhere in the response must not trip up usage
    /// extraction. logprobs is unrelated to usage but the extractor walks
    /// the same body.
    #[test]
    fn extract_usage_ignores_null_logprobs_field() {
        let v = json!({
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
                "logprobs": {
                    "content": [{
                        "token": "ok", "logprob": -0.1, "top_logprobs": null,
                    }],
                },
            }],
            "usage": {"prompt_tokens": 73, "completion_tokens": 5, "total_tokens": 78},
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.total_tokens, 78);
    }

    /// Ported from `TestMissingChoicesGuard`: when `choices` is missing or
    /// empty (an upstream protocol violation), usage extraction must still
    /// succeed if a usage object is present — the orchestrator's error
    /// handling depends on usage being available even when choices aren't.
    #[test]
    fn extract_usage_works_without_choices_field() {
        let v = json!({
            "id": "msg_123",
            "model": "some-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 1);
        assert_eq!(u.total_tokens, 11);
    }

    #[test]
    fn extract_usage_works_with_empty_choices_array() {
        let v = json!({
            "id": "msg_456",
            "model": "some-model",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.total_tokens, 6);
    }

    /// `prompt_tokens_details.cached_tokens` must obey the subset invariant
    /// `cached_tokens <= prompt_tokens` after extraction. The OpenAI branch
    /// leaves prompt_tokens untouched (already inclusive of cache hits); the
    /// extractor must not double-count.
    #[test]
    fn extract_usage_openai_cached_tokens_are_subset_of_prompt() {
        let v = json!({
            "usage": {
                "prompt_tokens": 2006,
                "completion_tokens": 300,
                "total_tokens": 2306,
                "prompt_tokens_details": {"cached_tokens": 1920},
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 2006);
        assert_eq!(u.cache_read_input_tokens, Some(1920));
        assert!(
            u.cache_read_input_tokens.unwrap() <= u.prompt_tokens,
            "cached_tokens must be a subset of prompt_tokens"
        );
    }

    /// When the upstream reports prompt_tokens=0 but a non-zero cache_read
    /// (Anthropic edge case where everything came from cache), the fold
    /// still produces a non-zero prompt_tokens obeying the subset invariant.
    #[test]
    fn extract_usage_anthropic_zero_input_with_cache_read() {
        let v = json!({
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 5,
                "cache_read_input_tokens": 1000,
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 1000, "cache_read folded into prompt");
        assert_eq!(u.cache_read_input_tokens, Some(1000));
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 1005);
    }

    #[test]
    fn extract_usage_reads_native_anthropic_input_output_tokens() {
        let v = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 1000,
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 1100, "cache_read folded into input");
        assert_eq!(u.cache_read_input_tokens, Some(1000));
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 1120);
    }

    // ---- H6: reasoning_tokens extraction / streaming aggregation ----

    /// H6: OpenAI o-series and DeepSeek/Grok report reasoning (thinking)
    /// tokens under `usage.completion_tokens_details.reasoning_tokens`. The
    /// extractor must surface them so streaming and non-streaming paths
    /// alike can price reasoning output. This is the core fix for the bug
    /// where reasoning_tokens was hardcoded to `None`.
    #[test]
    fn extract_usage_reads_reasoning_tokens_from_completion_details() {
        let v = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 80,
                "total_tokens": 180,
                "completion_tokens_details": {"reasoning_tokens": 60}
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(
            u.reasoning_tokens,
            Some(60),
            "reasoning_tokens must be read from completion_tokens_details"
        );
        // Reasoning tokens are a subset of completion_tokens; the extractor
        // must NOT subtract them from the completion count.
        assert_eq!(u.completion_tokens, 80);
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.total_tokens, 180);
    }

    #[test]
    fn extract_usage_reads_citation_tokens_from_usage_field() {
        let v = json!({
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "citation_tokens": 42
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.citation_tokens, Some(42));
        assert_eq!(u.web_search_requests, None);
    }

    #[test]
    fn extract_usage_derives_citation_tokens_from_citations_array() {
        let v = json!({
            "citations": ["https://a.test", "https://b.test", "https://c.test"],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.citation_tokens, Some(3));
    }

    #[test]
    fn extract_usage_no_citation_tokens_when_citations_empty() {
        let v = json!({
            "citations": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.citation_tokens, None);
    }

    #[test]
    fn extract_usage_reads_web_search_requests_from_num_search_queries() {
        let v = json!({
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "num_search_queries": 5
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.web_search_requests, Some(5));
    }

    #[test]
    fn extract_usage_reads_web_search_requests_from_prompt_tokens_details() {
        let v = json!({
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "prompt_tokens_details": {"web_search_requests": 3}
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.web_search_requests, Some(3));
    }

    #[test]
    fn extract_usage_reads_web_search_requests_from_top_level() {
        let v = json!({
            "num_search_queries": 2,
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.web_search_requests, Some(2));
    }

    #[test]
    fn merge_streaming_usage_keeps_max_citation_and_search() {
        let mut acc = Some(oxllm_core::UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: Some(2),
            web_search_requests: Some(1),
            browser_open_requests: Some(3),
        });
        let incoming = oxllm_core::UsageSummary {
            prompt_tokens: 12,
            completion_tokens: 6,
            total_tokens: 18,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: Some(5),
            web_search_requests: None,
            browser_open_requests: None,
        };
        merge_streaming_usage(&mut acc, incoming);
        let acc = acc.expect("present");
        assert_eq!(acc.citation_tokens, Some(5), "larger citation count wins");
        assert_eq!(
            acc.web_search_requests,
            Some(1),
            "existing search count preserved when incoming is None"
        );
        assert_eq!(
            acc.browser_open_requests,
            Some(3),
            "existing browser-open count preserved when incoming is None"
        );
    }

    /// H6: when `completion_tokens_details` is absent or has no
    /// `reasoning_tokens` key, the field stays `None` (no synthesis).
    #[test]
    fn extract_usage_reasoning_tokens_none_when_absent() {
        let v = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 80
            }
        });
        let u = extract_usage(&v).expect("present");
        assert!(
            u.reasoning_tokens.is_none(),
            "reasoning_tokens must be None when completion_tokens_details is absent"
        );
    }

    /// The caller is expected to compute `text_tokens = completion_tokens -
    /// reasoning_tokens` itself. The extractor preserves both fields so the
    /// subtraction can happen without re-parsing the body: 80 - 50 = 30.
    #[test]
    fn extract_usage_text_tokens_derivable_as_completion_minus_reasoning() {
        let v = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 80,
                "completion_tokens_details": {"reasoning_tokens": 50}
            }
        });
        let u = extract_usage(&v).expect("present");
        let reasoning = u.reasoning_tokens.expect("reasoning present");
        assert_eq!(
            u.completion_tokens.saturating_sub(reasoning),
            30,
            "text_tokens = completion - reasoning"
        );
    }

    /// `prompt_tokens_details.audio_tokens` already feeds
    /// `audio_input_tokens`; this pins the extraction path so future
    /// refactors don't drop it.
    #[test]
    fn extract_usage_reads_audio_tokens_from_prompt_details_h4() {
        let v = json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "prompt_tokens_details": {"audio_tokens": 250}
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.audio_input_tokens, Some(250));
    }

    /// `prompt_tokens_details.image_tokens` is reported by some multimodal
    /// upstreams. We don't store it on `UsageSummary` yet (no billing field),
    /// but extraction of the surrounding details must not be disturbed.
    #[test]
    fn extract_usage_tolerates_image_tokens_in_prompt_details() {
        let v = json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "prompt_tokens_details": {"image_tokens": 700, "cached_tokens": 100}
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 1000);
        assert_eq!(u.cache_read_input_tokens, Some(100));
        assert_eq!(u.reasoning_tokens, None);
    }

    /// When `completion_tokens_details` is absent (non-reasoning models),
    /// `reasoning_tokens` must be `None` — never `Some(0)` — so the cost
    /// calculator can distinguish "did not report" from "reported zero".
    #[test]
    fn extract_usage_reasoning_tokens_none_when_details_missing() {
        let v = json!({
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let u = extract_usage(&v).expect("present");
        assert!(u.reasoning_tokens.is_none());

        // Even with completion_tokens_details present but no reasoning key.
        let v = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"audio_tokens": 2}
            }
        });
        let u = extract_usage(&v).expect("present");
        assert!(
            u.reasoning_tokens.is_none(),
            "reasoning_tokens must not be synthesized from unrelated details"
        );
    }

    /// H6: the streaming usage merger must propagate `reasoning_tokens` from
    /// the final usage chunk (sent once via `stream_options.include_usage`)
    /// into the accumulator — otherwise the finish chunk / post-success
    /// billing loses the reasoning token count entirely.
    #[test]
    fn merge_streaming_usage_propagates_reasoning_tokens() {
        let mut acc: Option<oxllm_core::UsageSummary> = Some(oxllm_core::UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
            ..Default::default()
        });
        let incoming = oxllm_core::UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            reasoning_tokens: Some(15),
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
            ..Default::default()
        };
        merge_streaming_usage(&mut acc, incoming);
        let merged = acc.expect("accumulated");
        assert_eq!(merged.reasoning_tokens, Some(15));
    }

    /// H6: once the accumulator has reasoning_tokens set, a subsequent
    /// incoming chunk without it must NOT clobber the value (mirrors the
    /// cache/audio "take incoming when Some" semantics).
    #[test]
    fn merge_streaming_usage_keeps_reasoning_tokens_when_incoming_none() {
        let mut acc: Option<oxllm_core::UsageSummary> = Some(oxllm_core::UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            reasoning_tokens: Some(15),
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
            ..Default::default()
        });
        let incoming = oxllm_core::UsageSummary {
            prompt_tokens: 12,
            completion_tokens: 25,
            total_tokens: 37,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
            ..Default::default()
        };
        merge_streaming_usage(&mut acc, incoming);
        let merged = acc.expect("accumulated");
        assert_eq!(
            merged.reasoning_tokens,
            Some(15),
            "existing reasoning_tokens must survive a None-incoming merge"
        );
        assert_eq!(merged.prompt_tokens, 12);
        assert_eq!(merged.completion_tokens, 25);
    }

    /// OpenAI reports cache-write tokens under
    /// `prompt_tokens_details.cache_write_tokens` as a SUBSET of
    /// `prompt_tokens`. Because `cost_from_rates` treats
    /// `cache_creation_input_tokens` as ADDITIVE to `prompt_tokens`, the
    /// extractor must remove the subset tokens from `prompt_tokens` so they
    /// are billed exactly once — otherwise they are double-charged (once
    /// inside `prompt_tokens` at the input rate, once in the cache-creation
    /// bucket). End-to-end through `extract_usage` → `calculate_cost`.
    #[test]
    fn openai_cache_write_tokens_billed_once_not_twice() {
        let pricing = oxllm_core::lookup_pricing("gpt-4o").expect("gpt-4o priced");
        let input_rate = pricing.input_cost_per_token;
        let v = json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 0,
                "prompt_tokens_details": {"cache_write_tokens": 300}
            }
        });
        let u = extract_usage(&v).expect("present");
        // Normalisation: the 300 cache-write tokens leave prompt_tokens.
        assert_eq!(
            u.prompt_tokens, 700,
            "OpenAI cache_write_tokens must be removed from prompt_tokens"
        );
        assert_eq!(u.cache_creation_input_tokens, Some(300));

        // Billed exactly once: gpt-4o declares no separate cache-creation
        // rate, so on the price-table path the bucket bills at 0.0 (reference
        // `_get_cost_per_unit` default) — the bill is just the 700 regular
        // input tokens. The pre-fix double-charge would have been
        // 1000*input + 300*input == 1300*input.
        let cost = oxllm_core::calculate_cost("gpt-4o", &u).expect("priced");
        let once = 700.0 * input_rate;
        let twice = 1300.0 * input_rate;
        assert!(
            (cost - once).abs() < 1e-12,
            "cache-write tokens must be billed once (and free when unpriced): got {cost}, expected {once}"
        );
        assert!(
            (cost - twice).abs() > 1e-9,
            "cache-write tokens must NOT be double-charged: got {cost}, double {twice}"
        );
    }

    /// The Anthropic provider mirrors `usage.cache_creation.ephemeral_1h_input_tokens`
    /// into `prompt_tokens_details.ephemeral_1h_input_tokens`. The extractor
    /// must surface it so `cost_from_rates` can bill the 1h TTL bucket at its
    /// distinct rate instead of lumping it into the 5m bucket.
    #[test]
    fn extract_usage_reads_ephemeral_1h_cache_creation_bucket() {
        let v = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "cache_creation_input_tokens": 300,
                "prompt_tokens_details": {
                    "cache_creation_tokens": 300,
                    "ephemeral_5m_input_tokens": 200,
                    "ephemeral_1h_input_tokens": 100
                }
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.cache_creation_input_tokens, Some(300));
        assert_eq!(u.cache_creation_1h_input_tokens, Some(100));
    }

    /// End-to-end (issue: Anthropic 1h cache-creation TTL pricing): a usage
    /// body carrying the 5m/1h TTL split bills 1h tokens at
    /// `cache_creation_input_token_cost_above_1hr` and the remainder at the
    /// 5m rate — never both at the cheaper 5m rate.
    #[test]
    fn anthropic_1h_cache_creation_billed_at_distinct_rate() {
        let pricing =
            oxllm_core::lookup_pricing("claude-sonnet-4-20250514").expect("claude priced");
        let five_m = pricing.cache_creation_input_token_cost.expect("5m rate");
        let one_h = pricing
            .cache_creation_input_token_cost_above_1hr
            .expect("1h rate");
        let v = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cache_creation_input_tokens": 500,
                "prompt_tokens_details": {
                    "cache_creation_tokens": 500,
                    "ephemeral_5m_input_tokens": 300,
                    "ephemeral_1h_input_tokens": 200
                }
            }
        });
        let u = extract_usage(&v).expect("present");
        let cost = oxllm_core::calculate_cost("claude-sonnet-4-20250514", &u).expect("priced");
        let expected = 100.0 * pricing.input_cost_per_token
            + 300.0 * five_m
            + 200.0 * one_h
            + 50.0 * pricing.output_cost_per_token;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
        // The pre-fix behavior billed every cache-creation token at the 5m
        // rate — strictly cheaper when the 1h rate is higher.
        let under_billed = 100.0 * pricing.input_cost_per_token
            + 500.0 * five_m
            + 50.0 * pricing.output_cost_per_token;
        assert!(
            (cost - under_billed).abs() > 1e-9,
            "1h bucket must bill at its distinct rate, got {cost}"
        );
    }

    /// Issue: the host usage extractor must saturate `total_tokens` like the
    /// Anthropic provider's `saturating_add` — a provider-supplied count near
    /// `u64::MAX` must not panic debug builds or wrap release builds.
    #[test]
    fn extract_usage_saturates_total_on_overflow() {
        let v = json!({
            "usage": {
                "prompt_tokens": 18_446_744_073_709_551_615u64,
                "completion_tokens": 1,
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(
            u.total_tokens,
            u64::MAX,
            "total must saturate, not panic or wrap"
        );
        assert_eq!(u.prompt_tokens, u64::MAX);
    }

    /// Issue: the streaming merger takes the larger value for the subset
    /// buckets too — a provider lowering a running tally must not shrink the
    /// bill. Mirrors the max-based prompt/completion semantics.
    #[test]
    fn merge_streaming_usage_keeps_larger_subset_bucket() {
        let mut acc: Option<oxllm_core::UsageSummary> = Some(oxllm_core::UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            cache_read_input_tokens: Some(400),
            cache_creation_input_tokens: Some(300),
            cache_creation_1h_input_tokens: Some(100),
            audio_input_tokens: Some(50),
            reasoning_tokens: Some(10),
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
        });
        let incoming = oxllm_core::UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            // A non-monotonic later chunk reports FEWER tokens in every
            // subset bucket — the accumulator must keep the larger values.
            cache_read_input_tokens: Some(100),
            cache_creation_input_tokens: Some(100),
            cache_creation_1h_input_tokens: Some(20),
            audio_input_tokens: Some(10),
            reasoning_tokens: Some(5),
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
        };
        merge_streaming_usage(&mut acc, incoming);
        let merged = acc.expect("accumulated");
        assert_eq!(merged.cache_read_input_tokens, Some(400));
        assert_eq!(merged.cache_creation_input_tokens, Some(300));
        assert_eq!(merged.cache_creation_1h_input_tokens, Some(100));
        assert_eq!(merged.audio_input_tokens, Some(50));
        assert_eq!(merged.reasoning_tokens, Some(10));
    }

    /// A `None` incoming subset bucket must never clobber an existing value.
    #[test]
    fn merge_streaming_usage_keeps_subset_bucket_when_incoming_none() {
        let mut acc: Option<oxllm_core::UsageSummary> = Some(oxllm_core::UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            cache_read_input_tokens: Some(400),
            cache_creation_input_tokens: Some(300),
            cache_creation_1h_input_tokens: Some(100),
            audio_input_tokens: Some(50),
            reasoning_tokens: Some(10),
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
        });
        let incoming = oxllm_core::UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            ..Default::default()
        };
        merge_streaming_usage(&mut acc, incoming);
        let merged = acc.expect("accumulated");
        assert_eq!(merged.cache_read_input_tokens, Some(400));
        assert_eq!(merged.cache_creation_input_tokens, Some(300));
        assert_eq!(merged.cache_creation_1h_input_tokens, Some(100));
        assert_eq!(merged.audio_input_tokens, Some(50));
        assert_eq!(merged.reasoning_tokens, Some(10));
    }

    /// H6 end-to-end: a streaming final usage chunk carrying
    /// `completion_tokens_details.reasoning_tokens` flows through
    /// `extract_usage` → `merge_streaming_usage` and surfaces on the
    /// aggregated snapshot.
    #[test]
    fn streaming_final_usage_chunk_surfaces_reasoning_tokens() {
        let final_chunk = json!({
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 120,
                "total_tokens": 170,
                "completion_tokens_details": {"reasoning_tokens": 90}
            }
        });
        let mut acc: Option<oxllm_core::UsageSummary> = None;
        merge_streaming_usage(&mut acc, extract_usage(&final_chunk).expect("present"));
        let merged = acc.expect("accumulated");
        assert_eq!(merged.prompt_tokens, 50);
        assert_eq!(merged.completion_tokens, 120);
        assert_eq!(merged.reasoning_tokens, Some(90));
    }

    /// `total_tokens` must be recomputed from the *normalised* prompt even
    /// when the upstream body supplies its own `total_tokens`. OpenAI
    /// `prompt_tokens_details.cache_write_tokens` is a subset of
    /// `prompt_tokens` and is subtracted out during normalisation; keeping
    /// the upstream `total_tokens` (computed pre-normalisation) would break
    /// the `total == prompt + completion` invariant and mis-enforce
    /// deployment TPM limits (`record_deployment_tpm` reads `total_tokens`).
    #[test]
    fn extract_usage_recomputes_total_after_cache_normalisation() {
        let v = json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "total_tokens": 1010,
                "prompt_tokens_details": {"cache_write_tokens": 300}
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 700, "cache_write_tokens subtracted");
        assert_eq!(u.completion_tokens, 10);
        assert_eq!(
            u.total_tokens,
            u.prompt_tokens + u.completion_tokens,
            "total must be recomputed from the normalised prompt"
        );
        assert_eq!(u.total_tokens, 710);
        assert_ne!(
            u.total_tokens, 1010,
            "must not keep upstream total verbatim"
        );
    }

    /// Anthropic cache-read fold: when `cache_read_input_tokens` is folded
    /// into `prompt_tokens`, the recomputed `total_tokens` must reflect the
    /// larger prompt — even if the upstream supplied a smaller `total_tokens`
    /// computed against the pre-fold `input_tokens`.
    #[test]
    fn extract_usage_recomputes_total_after_anthropic_cache_read_fold() {
        let v = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_read_input_tokens": 1_000_000u64
            }
        });
        let u = extract_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 1_000_100);
        assert_eq!(
            u.total_tokens,
            u.prompt_tokens + u.completion_tokens,
            "total must follow the folded prompt, not the upstream value"
        );
        assert_eq!(u.total_tokens, 1_000_150);
    }
}
