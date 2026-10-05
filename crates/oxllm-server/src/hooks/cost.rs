//! `CostTrackingHook` — calculates the cost of each successful request,
//! emits a structured spend log line, and (when a DB is configured) queues a
//! spend entry to the batched spend writer.

use async_trait::async_trait;
use oxllm_core::{CallContext, Hook, ResponseSummary, UsageSummary};

use crate::spend_writer::{SpendEntry, tags_json};

/// A hook that calculates cost and emits a structured spend log line.
///
/// This hook is **post-success-only**: it implements [`Hook::post_success`]
/// (which skips cache hits) and intentionally has no `pre_call` side effects
/// (no budget reservation, no counter increment). Because there is nothing to
/// roll back, the default no-op [`Hook::post_failure`] is correct as-is. A
/// future maintainer adding a `pre_call` reservation here MUST also add the
/// corresponding `post_failure` release so the runner's reverse-order
/// rollback can undo it.
pub struct CostTrackingHook {
    spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
}

impl CostTrackingHook {
    /// Create a cost-tracking hook with no spend writer (stdout logging only).
    pub fn new() -> Self {
        Self { spend_tx: None }
    }

    /// Create a cost-tracking hook that also queues spend entries to the
    /// batched writer when `spend_tx` is `Some`.
    pub fn with_spend_writer(spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>) -> Self {
        Self { spend_tx }
    }
}

impl Default for CostTrackingHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for CostTrackingHook {
    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        if ctx.cache_hit {
            return;
        }
        let cost = crate::hooks::cost_fallback::fallback_cost(ctx, resp);
        let tags = tags_json(&ctx.request_tags);

        match (resp.usage.as_ref(), cost) {
            (Some(usage), Some(c)) => {
                tracing::info!(
                event = "spend",
                model = %resp.model,
                provider = %resp.provider,
                prompt_tokens = usage.prompt_tokens,
                completion_tokens = usage.completion_tokens,
                total_tokens = derived_total_tokens(usage),
                cost = c,
                cost_known = true,
                request_id = %ctx.request_id,
                tag_name = ctx.request_tag.as_deref(),
                tags = %tags.as_deref().unwrap_or("[]"),
                "spend recorded"
                );
                if let Some(tx) = &self.spend_tx {
                    let entry = SpendEntry {
                        timestamp: now_iso(),
                        request_id: Some(ctx.request_id.clone()),
                        model: resp.model.clone(),
                        provider: resp.provider.clone(),
                        prompt_tokens: Some(to_i64_tokens(usage.prompt_tokens)),
                        completion_tokens: Some(to_i64_tokens(usage.completion_tokens)),
                        total_tokens: Some(to_i64_tokens(derived_total_tokens(usage))),
                        cost: Some(c),
                        key_id: ctx.key_id.clone(),
                        team_id: ctx.team_id.clone(),
                        user_id: ctx.user_id.clone(),
                        end_user_id: ctx.end_user_id.clone(),
                        request_tag: ctx.request_tag.clone(),
                        request_tags: tags,
                        status: Some("success".to_string()),
                        latency_ms: latency_ms_from_ctx(ctx),
                    };
                    queue_spend(tx, entry).await;
                }
            }
            (Some(usage), None) => {
                tracing::info!(
                event = "spend",
                model = %resp.model,
                provider = %resp.provider,
                prompt_tokens = usage.prompt_tokens,
                completion_tokens = usage.completion_tokens,
                total_tokens = derived_total_tokens(usage),
                cost_known = false,
                request_id = %ctx.request_id,
                tag_name = ctx.request_tag.as_deref(),
                tags = %tags.as_deref().unwrap_or("[]"),
                "no price-table entry; cost omitted"
                );
                if let Some(tx) = &self.spend_tx {
                    let entry = SpendEntry {
                        timestamp: now_iso(),
                        request_id: Some(ctx.request_id.clone()),
                        model: resp.model.clone(),
                        provider: resp.provider.clone(),
                        prompt_tokens: Some(to_i64_tokens(usage.prompt_tokens)),
                        completion_tokens: Some(to_i64_tokens(usage.completion_tokens)),
                        total_tokens: Some(to_i64_tokens(derived_total_tokens(usage))),
                        cost: None,
                        key_id: ctx.key_id.clone(),
                        team_id: ctx.team_id.clone(),
                        user_id: ctx.user_id.clone(),
                        end_user_id: ctx.end_user_id.clone(),
                        request_tag: ctx.request_tag.clone(),
                        request_tags: tags,
                        status: Some("success".to_string()),
                        latency_ms: latency_ms_from_ctx(ctx),
                    };
                    queue_spend(tx, entry).await;
                }
            }
            (None, cost) => {
                // Per-request/per-second/provider-reported prices can be
                // authoritative even without token usage. Preserve that cost
                // so durable spend agrees with BudgetHook's settlement.
                tracing::info!(
                event = "spend",
                model = %resp.model,
                provider = %resp.provider,
                request_id = %ctx.request_id,
                cost = cost,
                cost_known = cost.is_some(),
                tag_name = ctx.request_tag.as_deref(),
                tags = %tags.as_deref().unwrap_or("[]"),
                "spend recorded without token usage"
                );
                if let Some(tx) = &self.spend_tx {
                    let entry = SpendEntry {
                        timestamp: now_iso(),
                        request_id: Some(ctx.request_id.clone()),
                        model: resp.model.clone(),
                        provider: resp.provider.clone(),
                        prompt_tokens: None,
                        completion_tokens: None,
                        total_tokens: None,
                        cost,
                        key_id: ctx.key_id.clone(),
                        team_id: ctx.team_id.clone(),
                        user_id: ctx.user_id.clone(),
                        end_user_id: ctx.end_user_id.clone(),
                        request_tag: ctx.request_tag.clone(),
                        request_tags: tags,
                        status: Some("success".to_string()),
                        latency_ms: latency_ms_from_ctx(ctx),
                    };
                    queue_spend(tx, entry).await;
                }
            }
        }
    }
}

fn latency_ms_from_ctx(ctx: &CallContext) -> Option<i64> {
    ctx.start_time
        .map(|s| s.elapsed().as_millis() as i64)
        .filter(|&ms| ms >= 0)
}

async fn queue_spend(tx: &tokio::sync::mpsc::Sender<SpendEntry>, entry: SpendEntry) {
    match tx.try_send(entry) {
        Ok(()) => {}
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
            tracing::warn!("spend writer channel full; dropping spend entry");
            crate::metrics::record_spend_dropped("full", 1);
        }
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
            tracing::warn!("spend writer channel closed; dropping spend entry");
            crate::metrics::record_spend_dropped("closed", 1);
        }
    }
}

/// Saturating `u64 -> i64` cast. `UsageSummary` token fields are `u64` while
/// `SpendEntry` uses `i64`; a plain `as i64` would silently wrap a count above
/// `i64::MAX` to a negative value. Realistic counts are many orders of
/// magnitude smaller, but saturating is defence in depth.
fn to_i64_tokens(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// Resolve the `total_tokens` to persist/emit. Some providers send
/// `total_tokens: 0` (or a value smaller than `prompt + completion`) while the
/// subset fields are non-zero; the persisted row would then carry a misleading
/// total that downstream analytics may treat as authoritative. Prefer the
/// provider's value when it is consistent, otherwise derive the sum of the
/// prompt and completion counts.
fn derived_total_tokens(usage: &UsageSummary) -> u64 {
    let sum = usage.prompt_tokens.saturating_add(usage.completion_tokens);
    if usage.total_tokens == 0 || usage.total_tokens < sum {
        sum
    } else {
        usage.total_tokens
    }
}

fn now_iso() -> String {
    #[cfg(feature = "db")]
    {
        oxllm_db::now_iso()
    }
    #[cfg(not(feature = "db"))]
    {
        jiff::Timestamp::now().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::UsageSummary;

    fn make_ctx() -> CallContext {
        CallContext {
            key_id: Some("k-1".to_string()),
            ..CallContext::for_test("gpt-4o")
        }
    }

    fn make_response(model: &str, prompt: u64, completion: u64) -> ResponseSummary {
        ResponseSummary {
            model: model.to_string(),
            cost: None,
            usage: Some(UsageSummary {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[tokio::test]
    async fn post_success_with_known_model_does_not_panic() {
        let hook = CostTrackingHook::new();
        let ctx = make_ctx();
        let resp = make_response("gpt-4o", 100, 50);
        hook.post_success(&ctx, &resp).await;
    }

    #[tokio::test]
    async fn post_success_with_unknown_model_does_not_panic() {
        let hook = CostTrackingHook::new();
        let ctx = make_ctx();
        let resp = make_response("made-up-model", 100, 50);
        hook.post_success(&ctx, &resp).await;
    }

    #[tokio::test]
    async fn post_success_with_no_usage_does_not_panic() {
        let hook = CostTrackingHook::new();
        let ctx = make_ctx();
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: None,
            provider: "openai".to_string(),
        };
        hook.post_success(&ctx, &resp).await;
    }

    #[tokio::test]
    async fn queues_entry_when_writer_present() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SpendEntry>(8);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let ctx = make_ctx();
        let resp = make_response("gpt-4o", 10, 5);
        hook.post_success(&ctx, &resp).await;

        let entry = rx.recv().await.expect("entry queued");
        assert_eq!(entry.model, "gpt-4o");
        assert_eq!(entry.key_id.as_deref(), Some("k-1"));
        assert!(entry.cost.unwrap_or(-1.0) > 0.0);
    }

    #[tokio::test]
    async fn known_cost_without_token_usage_is_preserved_with_all_tag_attribution() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SpendEntry>(1);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let mut ctx = make_ctx();
        ctx.request_tag = Some("app".into());
        ctx.request_tags = vec!["app".into(), "prod".into()];
        ctx.end_user_id = Some("customer".into());
        hook.post_success(
            &ctx,
            &ResponseSummary {
                model: "unpriced-token-model".into(),
                provider: "openai".into(),
                cost: Some(0.25),
                usage: None,
                body: None,
            },
        )
        .await;
        let entry = rx.try_recv().expect("known-cost entry queued");
        assert_eq!(
            entry.cost,
            Some(0.25),
            "resolved non-token cost must reach the ledger"
        );
        assert!(entry.prompt_tokens.is_none() && entry.total_tokens.is_none());
        assert_eq!(entry.end_user_id.as_deref(), Some("customer"));
        let tags: Vec<String> =
            serde_json::from_str(entry.request_tags.as_deref().unwrap()).unwrap();
        assert_eq!(tags, vec!["app", "prod"]);
    }

    #[test]
    fn structured_spend_tags_match_the_ledger_for_every_usage_branch() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        for (usage, cost) in [
            (Some(UsageSummary::default()), Some(0.25)),
            (Some(UsageSummary::default()), None),
            (None, Some(0.25)),
        ] {
            let mut ctx = CallContext::for_test("unpriced-log-test");
            ctx.request_tag = Some("app".into());
            ctx.request_tags = vec!["app".into(), "quoted\"tag\nnext-line".into()];
            let expected_tags = tags_json(&ctx.request_tags).expect("tags JSON");
            let (entry, log) = crate::security::capture_test_logs(|| {
                runtime.block_on(async {
                    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
                    CostTrackingHook::with_spend_writer(Some(tx))
                        .post_success(
                            &ctx,
                            &ResponseSummary {
                                model: "unpriced-log-test".into(),
                                provider: "openai".into(),
                                usage,
                                cost,
                                body: None,
                            },
                        )
                        .await;
                    rx.try_recv().expect("ledger entry")
                })
            });
            assert_eq!(entry.request_tags.as_deref(), Some(expected_tags.as_str()));
            let spend_lines: Vec<_> = log
                .lines()
                .filter(|line| line.contains("event=\"spend\""))
                .collect();
            assert_eq!(
                spend_lines.len(),
                1,
                "tags must not inject additional log lines: {log}"
            );
            assert!(
                spend_lines[0].contains(&format!("tags={expected_tags}")),
                "log and ledger tags must agree: {log}"
            );
            assert!(spend_lines[0].contains(&ctx.request_id));
        }
    }

    #[tokio::test]
    async fn cache_hit_does_not_queue_spend_entry() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SpendEntry>(8);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let mut ctx = make_ctx();
        ctx.cache_hit = true;
        let resp = make_response("gpt-4o", 10, 5);
        hook.post_success(&ctx, &resp).await;
        assert!(
            rx.try_recv().is_err(),
            "cache hits should not queue spend entries"
        );
    }

    /// Documents the contract: `CostTrackingHook` is post-success-only with no
    /// `pre_call` reservation, so the default no-op `post_failure` is correct —
    /// a failed request must never queue a spend entry. Pins the doc-comment so
    /// a future maintainer who adds a `pre_call` reservation also adds the
    /// matching `post_failure` release.
    #[tokio::test]
    async fn post_failure_on_cost_hook_is_a_noop() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SpendEntry>(8);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let ctx = make_ctx();
        hook.post_failure(&ctx, &oxllm_core::Error::Network("boom".into()))
            .await;
        assert!(
            rx.try_recv().is_err(),
            "post_failure must not queue a spend entry"
        );
    }

    #[tokio::test]
    async fn derived_total_tokens_when_provider_reports_zero() {
        // Provider sends total_tokens=0 while prompt/completion are non-zero;
        // the queued entry must carry the derived sum, not the misleading 0.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SpendEntry>(8);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let ctx = make_ctx();
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: Some(UsageSummary {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 0,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        };
        hook.post_success(&ctx, &resp).await;
        let entry = rx.recv().await.expect("entry queued");
        assert_eq!(
            entry.total_tokens,
            Some(150),
            "provider total=0 must be replaced by prompt+completion"
        );
        assert_eq!(entry.prompt_tokens, Some(100));
        assert_eq!(entry.completion_tokens, Some(50));
    }

    #[tokio::test]
    async fn derived_total_tokens_when_provider_reports_less_than_sum() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SpendEntry>(8);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let ctx = make_ctx();
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: Some(UsageSummary {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 40,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        };
        hook.post_success(&ctx, &resp).await;
        let entry = rx.recv().await.expect("entry queued");
        assert_eq!(
            entry.total_tokens,
            Some(150),
            "inconsistent provider total (< sum) must be replaced by the sum"
        );
    }

    #[tokio::test]
    async fn saturating_cast_keeps_pathological_count_nonnegative() {
        // u64::MAX would wrap to -1 under a plain `as i64`; the entry must
        // carry i64::MAX instead.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SpendEntry>(8);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let ctx = make_ctx();
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: Some(UsageSummary {
                prompt_tokens: u64::MAX,
                completion_tokens: u64::MAX,
                total_tokens: u64::MAX,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        };
        hook.post_success(&ctx, &resp).await;
        let entry = rx.recv().await.expect("entry queued");
        assert_eq!(entry.prompt_tokens, Some(i64::MAX));
        assert_eq!(entry.completion_tokens, Some(i64::MAX));
        assert!(
            entry.total_tokens.unwrap_or(-1) >= 0,
            "must stay nonnegative"
        );
    }

    #[tokio::test]
    async fn queues_entry_with_null_cost_when_no_price_entry() {
        // (Some(usage), None) branch: usage present but no price-table entry
        // for either the serving or requested model; an entry with cost: None
        // must still be queued.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SpendEntry>(8);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        // Both the serving model (resp.model) and the requested model
        // (ctx.model) must be unpriced so neither branch resolves a cost.
        let ctx = CallContext::for_test("totally-unknown-requested-model");
        let resp = make_response("totally-unknown-model", 10, 5);
        hook.post_success(&ctx, &resp).await;
        let entry = rx.recv().await.expect("entry queued even without price");
        assert_eq!(entry.model, "totally-unknown-model");
        assert!(entry.cost.is_none(), "no price entry -> cost is None");
        assert_eq!(entry.prompt_tokens, Some(10));
        assert_eq!(entry.completion_tokens, Some(5));
        assert_eq!(entry.total_tokens, Some(15));
    }

    /// LOW: the `resp.cost == None` fallback must route through the
    /// provider-aware price lookup (`calculate_cost_with_override_for_provider`),
    /// matching the primary billing path (`orchestration/stream.rs`) and the
    /// budget hook. A tiered model whose price row is keyed with a provider
    /// prefix (e.g. `gemini/gemini-2.5-pro`) and whose response echoes the
    /// BARE name must resolve to the serving provider's canonical row via
    /// `lookup_pricing_with_provider`. Today the bare `gemini-2.5-pro` also
    /// resolves to a `vertex_ai` alias with identical rates, so the legacy
    /// provider-agnostic path happens to bill the same amount — but that is
    /// fragile (a future cross-provider alias with different tier rates would
    /// silently bill wrong). This test pins that the fallback uses the
    /// provider-aware path and bills the correct >200k tier rate.
    #[tokio::test]
    async fn cost_hook_fallback_uses_serving_provider_for_tiered_gemini() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SpendEntry>(8);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let ctx = CallContext::for_test("gemini-2.5-pro");
        // Bare response model name (the Gemini OpenAI-compat shape), provider
        // = "gemini", cost not pre-computed by the pipeline, >200k prompt so
        // the context tier fires.
        let prompt = 250_000u64;
        let completion = 1_000u64;
        let usage = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let resp = ResponseSummary {
            model: "gemini-2.5-pro".to_string(),
            cost: None,
            usage: Some(usage.clone()),
            body: None,
            provider: "gemini".to_string(),
        };
        hook.post_success(&ctx, &resp).await;
        let entry = rx.recv().await.expect("entry queued");

        // The cost must match the provider-aware tier billing exactly (the
        // gemini-2.5-pro >200k tier rates: input 2.5e-6, output 1.5e-5).
        let expected = oxllm_core::calculate_cost_with_provider("gemini-2.5-pro", &usage, "gemini")
            .expect("provider-aware tier cost resolves");
        let cost = entry
            .cost
            .expect("cost must resolve via provider-aware fallback");
        assert!(
            (cost - expected).abs() < 1e-9,
            "fallback must use the serving provider's tiered row: got {cost}, expected {expected}"
        );
        // And that expected cost is the >200k tier rate, not the base rate.
        assert_eq!(expected, 250_000.0 * 2.5e-6 + 1_000.0 * 1.5e-5);
    }

    #[test]
    fn to_i64_tokens_saturates_at_max() {
        assert_eq!(to_i64_tokens(0), 0);
        assert_eq!(to_i64_tokens(42), 42);
        assert_eq!(to_i64_tokens(i64::MAX as u64), i64::MAX);
        assert_eq!(to_i64_tokens(u64::MAX), i64::MAX);
    }

    #[test]
    fn derived_total_tokens_prefers_consistent_provider_value() {
        let usage = UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        assert_eq!(derived_total_tokens(&usage), 30);
    }

    /// Coverage gap (C16): `queue_spend` — the only code path that queues
    /// real spend rows in production — must count dropped entries under
    /// `reason="full"` when the channel is at capacity. This exercises the
    /// real hook (not the dead `SpendWriter::queue` wrapper), driving it
    /// with a capacity-1 sender and verifying the overflow lands on the
    /// `full` label of `oxllm_spend_entries_dropped_total`. Mirrors
    /// `spend_writer::tests::queue_overflow_increments_drop_counter` for
    /// the production code path.
    #[tokio::test]
    #[serial_test::serial]
    async fn hook_queue_spend_counts_full_drops() {
        let _ = crate::metrics::install();
        let before = spend_dropped_count("full");

        // Capacity 1, no draining receiver: the first queue succeeds, every
        // further one overflows.
        let (tx, _rx) = tokio::sync::mpsc::channel::<SpendEntry>(1);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let ctx = make_ctx();
        let resp = make_response("gpt-4o", 10, 5);

        // Drive the real `post_success` path three times. The channel can
        // hold one entry, so two of the three must be counted as `full`.
        hook.post_success(&ctx, &resp).await;
        hook.post_success(&ctx, &resp).await;
        hook.post_success(&ctx, &resp).await;

        let after = spend_dropped_count("full");
        assert!(
            after.saturating_sub(before) >= 2,
            "expected at least 2 'full' drops from the real hook path, \
             got before={before} after={after}"
        );
    }

    /// Coverage gap (C16): mirrors the test above for the `Closed` branch.
    /// When the channel receiver is dropped, every `post_success` call must
    /// increment `reason="closed"` so the loss is observable from the real
    /// production path.
    #[tokio::test]
    #[serial_test::serial]
    async fn hook_queue_spend_counts_closed_drops() {
        let _ = crate::metrics::install();
        let before = spend_dropped_count("closed");

        let (tx, rx) = tokio::sync::mpsc::channel::<SpendEntry>(2);
        drop(rx);
        let hook = CostTrackingHook::with_spend_writer(Some(tx));
        let ctx = make_ctx();
        let resp = make_response("gpt-4o", 10, 5);

        hook.post_success(&ctx, &resp).await;
        hook.post_success(&ctx, &resp).await;

        let after = spend_dropped_count("closed");
        assert_eq!(
            after.saturating_sub(before),
            2,
            "expected 2 'closed' drops from the real hook path, \
             got before={before} after={after}"
        );
    }

    /// Parse the current value of
    /// `oxllm_spend_entries_dropped_total{reason=...}` out of the rendered
    /// Prometheus exposition text (0 when the series is absent). Mirrors
    /// the same helper in `spend_writer::tests`.
    fn spend_dropped_count(reason: &str) -> u64 {
        let needle = format!("reason=\"{reason}\"");
        for line in crate::metrics::render().lines() {
            if line.starts_with("oxllm_spend_entries_dropped_total{") && line.contains(&needle) {
                if let Some(v) = line.rsplit(' ').next() {
                    if let Ok(n) = v.parse::<u64>() {
                        return n;
                    }
                }
            }
        }
        0
    }
}
