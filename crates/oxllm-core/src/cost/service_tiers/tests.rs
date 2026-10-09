use super::*;
use crate::cost::{DeploymentPricing, calculate_cost_for_service_tier, parse_provider_prices};
use crate::{CallContext, UsageSummary};
use serde_json::json;
use std::sync::Arc;

#[test]
fn served_tier_overrides_request_and_unknown_metadata_falls_back() {
    for (requested, served, expected) in [
        (Some("priority"), Some("default"), None),
        (Some("flex"), Some("STANDARD"), None),
        (Some("default"), Some("priority"), Some("priority")),
        (Some("priority"), Some("flex"), Some("flex")),
        (Some("auto"), Some("ultrafast"), Some("ultrafast")),
        (Some("priority"), Some("unknown"), Some("priority")),
        (Some("flex"), Some(""), Some("flex")),
        (Some("auto"), None, None),
        (None, Some("default"), None),
        (Some("BATCH"), None, None),
        (Some("batches"), None, None),
        (Some("priority"), Some("batch"), Some("batches")),
        (Some("flex"), Some("BATCHES"), Some("batches")),
        (Some("batches"), Some("standard"), None),
    ] {
        assert_eq!(resolve_service_tier(requested, served), expected);
    }
}

#[test]
fn response_metadata_is_detected_without_token_counts() {
    for (body, tier) in [
        (json!({"service_tier":"priority","choices":[]}), "priority"),
        (json!({"response":{"service_tier":"flex"}}), "flex"),
        (json!({"speed":"fast"}), "fast"),
        (json!({"speed":"standard"}), "default"),
        (json!({"message":{"speed":"fast"}}), "fast"),
        (json!({"message":{"speed":"standard"}}), "default"),
        (json!({"message":{"service_tier":"priority"}}), "priority"),
        (json!({"message":{"service_tier":"standard"}}), "default"),
        (
            json!({"message":{"usage":{"service_tier":"priority"}}}),
            "priority",
        ),
        (json!({"usage":{"service_tier":"ultrafast"}}), "ultrafast"),
        (json!({"message":{"usage":{"speed":"fast"}}}), "fast"),
        (
            json!({"usageMetadata":{"trafficType":"ON_DEMAND_PRIORITY"}}),
            "priority",
        ),
        (
            json!({"usageMetadata":{"trafficType":"ON_DEMAND"}}),
            "default",
        ),
        (json!({"usageMetadata":{"trafficType":"BATCH"}}), "batches"),
        (json!({"usage":{"traffic_type":"batch"}}), "batches"),
        (json!({"usageMetadata":{"serviceTier":"FLEX"}}), "flex"),
        (
            json!({"usageMetadata":{"serviceTier":"unspecified"}}),
            "default",
        ),
    ] {
        assert_eq!(response_service_tier(&body), Some(tier));
    }
    assert_eq!(response_service_tier(&json!({"service_tier":42})), None);
}

#[test]
fn native_standard_speed_downgrades_premium_requests_without_usage() {
    for request in [json!({"speed":"fast"}), json!({"service_tier":"priority"})] {
        let ctx = CallContext {
            request_body: Arc::new(request),
            served_service_tier: Some("fast".into()),
            ..CallContext::for_test("claude-opus-4-6")
        };
        for body in [
            json!({"speed":"standard","choices":[]}),
            json!({"type":"message_start","message":{"speed":"standard"}}),
            json!({"message":{"service_tier":"standard"}}),
            json!({"service_tier":null,"message":{"speed":"standard"}}),
        ] {
            assert_eq!(ctx.billable_service_tier(Some(&body)), None, "{body}");
        }
    }
}

#[test]
fn unpriced_anthropic_fast_mode_does_not_invent_premium_rates() {
    let usage = UsageSummary {
        prompt_tokens: 1_000,
        completion_tokens: 100,
        cache_read_input_tokens: Some(400),
        ..Default::default()
    };
    for model in ["claude-opus-4-6", "claude-sonnet-4-20250514"] {
        let standard = calculate_cost_for_service_tier(model, &usage, None, "anthropic", None)
            .expect("standard Anthropic price");
        let fast = calculate_cost_for_service_tier(model, &usage, None, "anthropic", Some("fast"))
            .expect("fallback Anthropic price");
        assert_eq!(
            fast, standard,
            "{model} has no published fast/priority overlay"
        );
    }
}

#[test]
fn fireworks_flash_priority_cost_uses_each_models_published_rates() {
    // LiteLLM f824d11a22, model_prices_and_context_window.json (2026-10-08).
    let usage = UsageSummary {
        prompt_tokens: 1_000,
        completion_tokens: 100,
        cache_read_input_tokens: Some(400),
        reasoning_tokens: Some(40),
        ..Default::default()
    };
    for (native, input, output, cache_read) in [
        ("deepseek-v4p1-flash", 3.75e-7, 1.5e-6, 7.5e-9),
        ("deepseek-v4-flash-0731", 2.75e-7, 8.25e-7, 8.75e-9),
        ("deepseek-v4-flash-vision-exp", 2.75e-7, 8.25e-7, 8.75e-9),
    ] {
        let expected = 600.0 * input + 400.0 * cache_read + 100.0 * output;
        for prefix in ["fireworks_ai/", "fireworks_ai/accounts/fireworks/models/"] {
            let model = format!("{prefix}{native}");
            let cost = calculate_cost_for_service_tier(
                &model,
                &usage,
                None,
                "fireworks_ai",
                Some("priority"),
            )
            .expect("published Fireworks priority cost");
            assert!(
                (cost - expected).abs() < 1e-12,
                "{model}: got {cost}, expected {expected}"
            );
        }
    }
}

#[test]
fn sonnet_55_batch_cost_covers_every_published_bucket() {
    // LiteLLM f824d11a22 publishes these four batch rates for all three rows.
    // No batch 1h write rate is published, so that bucket retains its own rate.
    let cases = [
        (
            UsageSummary {
                prompt_tokens: 1_000,
                ..Default::default()
            },
            1_000.0 * 1e-6,
        ),
        (
            UsageSummary {
                completion_tokens: 100,
                reasoning_tokens: Some(40),
                ..Default::default()
            },
            100.0 * 5e-6,
        ),
        (
            UsageSummary {
                prompt_tokens: 1_000,
                cache_read_input_tokens: Some(1_000),
                ..Default::default()
            },
            1_000.0 * 5e-8,
        ),
        (
            UsageSummary {
                cache_creation_input_tokens: Some(200),
                cache_creation_1h_input_tokens: Some(50),
                ..Default::default()
            },
            150.0 * 1.25e-6 + 50.0 * 4e-6,
        ),
    ];
    for (provider, model) in [
        ("anthropic", "claude-sonnet-5-5"),
        ("vertex_ai", "vertex_ai/claude-sonnet-5-5"),
        ("vertex_ai", "vertex_ai/claude-sonnet-5-5@default"),
    ] {
        for (usage, expected) in &cases {
            let cost =
                calculate_cost_for_service_tier(model, usage, None, provider, Some("batches"))
                    .expect("published Sonnet batch cost");
            assert!(
                (cost - expected).abs() < 1e-12,
                "{model}/{usage:?}: got {cost}, expected {expected}"
            );
        }
    }
}

#[test]
fn haiku_55_batch_cost_uses_provider_specific_context_and_cache_prices() {
    // LiteLLM f824d11a22: only Anthropic publishes >100k batch rates; Vertex
    // publishes flat batch rates. Both retain standard 1h write context prices.
    for (provider, models) in [
        ("anthropic", &["claude-haiku-5-5"][..]),
        (
            "vertex_ai",
            &[
                "vertex_ai/claude-haiku-5-5",
                "vertex_ai/claude-haiku-5-5@default",
            ][..],
        ),
    ] {
        for context in [1_000, 100_000, 100_001] {
            let usage = UsageSummary {
                prompt_tokens: context - 200,
                completion_tokens: 100,
                cache_read_input_tokens: Some(400),
                cache_creation_input_tokens: Some(200),
                cache_creation_1h_input_tokens: Some(50),
                reasoning_tokens: Some(40),
                ..Default::default()
            };
            let (input, output, cache_read, cache_write) =
                if provider == "anthropic" && context > 100_000 {
                    (2.5e-7, 1.25e-6, 2.5e-8, 3.125e-7)
                } else {
                    (5e-8, 2.5e-7, 5e-9, 6.25e-8)
                };
            let cache_write_1h = if context > 100_000 { 1e-6 } else { 2e-7 };
            let expected = (context - 600) as f64 * input
                + 400.0 * cache_read
                + 150.0 * cache_write
                + 50.0 * cache_write_1h
                + 100.0 * output;
            for model in models {
                let cost =
                    calculate_cost_for_service_tier(model, &usage, None, provider, Some("batches"))
                        .expect("published Haiku batch cost");
                assert!(
                    (cost - expected).abs() < 1e-12,
                    "{model}/{context}: got {cost}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn batch_traffic_is_served_metadata_and_not_a_request_service_tier() {
    for requested in ["batch", "batches", "BATCH"] {
        let ctx = CallContext {
            request_body: Arc::new(json!({"service_tier":requested})),
            ..CallContext::for_test("gemini-2.5-flash")
        };
        assert_eq!(ctx.billable_service_tier(None), None);
        assert_eq!(response_service_tier(&ctx.request_body), None);
        assert_eq!(
            ctx.billable_service_tier(Some(&json!({"usage":{"traffic_type":"BATCH"}}))),
            Some("batches")
        );
    }
    assert_eq!(
        response_service_tier(&json!({
            "usageMetadata":{"trafficType":null},
            "usage":{"traffic_type":"BATCH"}
        })),
        Some("batches")
    );
    assert_eq!(
        response_service_tier(&json!({"usageMetadata":{"trafficType":"UNSPECIFIED"}})),
        None
    );
    let ctx = CallContext {
        request_body: Arc::new(json!({"service_tier":"priority"})),
        served_service_tier: Some("batches".into()),
        ..CallContext::for_test("gemini-2.5-flash")
    };
    assert_eq!(ctx.billable_service_tier(None), Some("batches"));
    assert_eq!(
        ctx.billable_service_tier(Some(&json!({"usageMetadata":{"trafficType":"ON_DEMAND"}}))),
        None
    );
    assert_eq!(
        ctx.billable_service_tier(Some(
            &json!({"usageMetadata":{"serviceTier":"unspecified"}})
        )),
        None
    );
}

#[test]
fn native_batch_traffic_overrides_native_processing_service_tiers() {
    let ctx = CallContext {
        request_body: Arc::new(json!({"service_tier":"priority"})),
        served_service_tier: Some("default".into()),
        ..CallContext::for_test("gemini-2.5-flash")
    };
    for tier in ["unspecified", "default", "standard", "FLEX", "priority"] {
        let body = json!({"usageMetadata":{"serviceTier":tier,"trafficType":"BATCH"}});
        assert_eq!(response_service_tier(&body), Some("batches"), "{body}");
        assert_eq!(ctx.billable_service_tier(Some(&body)), Some("batches"));

        // A metadata-only stream event must retain the same classification.
        let streamed = CallContext {
            served_service_tier: response_service_tier(&body).map(str::to_owned),
            ..ctx.clone()
        };
        assert_eq!(streamed.billable_service_tier(None), Some("batches"));
    }
    let translated = json!({
        "usageMetadata":{"serviceTier":"unspecified","trafficType":null},
        "usage":{"traffic_type":"BATCH"}
    });
    assert_eq!(response_service_tier(&translated), Some("batches"));
}

#[test]
fn explicit_generic_served_fields_still_precede_native_batch_traffic() {
    let ctx = CallContext {
        request_body: Arc::new(json!({"service_tier":"priority"})),
        ..CallContext::for_test("gemini-2.5-flash")
    };
    for (explicit, expected) in [
        (json!({"service_tier":"default"}), "default"),
        (json!({"service_tier":"standard"}), "default"),
        (json!({"service_tier":"priority"}), "priority"),
        (json!({"service_tier":"flex"}), "flex"),
        (json!({"response":{"service_tier":"default"}}), "default"),
        (json!({"usage":{"service_tier":"standard"}}), "default"),
        (json!({"message":{"speed":"standard"}}), "default"),
        (json!({"speed":"fast"}), "fast"),
    ] {
        let mut body = explicit;
        body["usageMetadata"] = json!({"serviceTier":"unspecified","trafficType":"BATCH"});
        assert_eq!(response_service_tier(&body), Some(expected), "{body}");
        let billable = (expected != "default").then_some(expected);
        assert_eq!(ctx.billable_service_tier(Some(&body)), billable, "{body}");
    }
    for invalid in [Value::Null, json!(42), json!("auto"), json!("batch")] {
        let body = json!({
            "service_tier":invalid,
            "usageMetadata":{"serviceTier":"standard","trafficType":"BATCH"}
        });
        assert_eq!(response_service_tier(&body), Some("batches"), "{body}");
    }
}

#[test]
fn nonbatch_native_service_tiers_retain_precedence_over_traffic_quota() {
    for (native, traffic, expected) in [
        ("flex", "ON_DEMAND", "flex"),
        ("priority", "ON_DEMAND", "priority"),
        ("standard", "ON_DEMAND", "default"),
        ("unspecified", "UNSPECIFIED", "default"),
        ("unknown", "ON_DEMAND_FLEX", "flex"),
    ] {
        let body = json!({"usageMetadata":{"serviceTier":native,"trafficType":traffic}});
        assert_eq!(response_service_tier(&body), Some(expected), "{body}");
    }
}

#[test]
fn native_batch_with_default_service_tier_calculates_published_cost_once() {
    let ctx = CallContext {
        request_body: Arc::new(json!({"service_tier":"priority"})),
        served_service_tier: Some("default".into()),
        ..CallContext::for_test("gemini-2.5-flash")
    };
    let usage = UsageSummary {
        prompt_tokens: 1_000,
        completion_tokens: 100,
        total_tokens: 1_100,
        cache_read_input_tokens: Some(400),
        audio_input_tokens: Some(100),
        reasoning_tokens: Some(40),
        ..Default::default()
    };
    // Published Batch/Flex rates; cached reads retain the standard cache rate.
    let expected = 500.0 * 1.5e-7 + 400.0 * 3e-8 + 100.0 * 5e-7 + 100.0 * 1.25e-6;
    for provider in ["gemini", "vertex_ai"] {
        let standard = calculate_cost_for_service_tier(&ctx.model, &usage, None, provider, None)
            .expect("standard Google cost");
        let flex =
            calculate_cost_for_service_tier(&ctx.model, &usage, None, provider, Some("flex"))
                .expect("published Google flex cost");
        for native in ["unspecified", "default", "standard"] {
            let mut body = json!({"usageMetadata":{
                "serviceTier":native,
                "trafficType":"BATCH",
                "promptTokenCount":1_000,
                "cachedContentTokenCount":400,
                "candidatesTokenCount":60,
                "thoughtsTokenCount":40,
                "totalTokenCount":1_100,
                "promptTokensDetails":[{"modality":"AUDIO","tokenCount":100}]
            }});
            let batch = calculate_cost_for_service_tier(
                &ctx.model,
                &usage,
                None,
                provider,
                ctx.billable_service_tier(Some(&body)),
            )
            .expect("batch cost with both native metadata markers");
            assert!(
                (batch - expected).abs() < 1e-12,
                "{provider}/{native}: {batch}"
            );
            assert_eq!(batch, flex, "published rates must not be discounted twice");
            assert!(
                (batch - standard).abs() > 1e-12,
                "native default must not price the job as standard"
            );
            assert!(
                batch > standard * 0.5,
                "cache reads must not receive the batch discount"
            );

            body["service_tier"] = json!("default");
            let explicit_standard = calculate_cost_for_service_tier(
                &ctx.model,
                &usage,
                None,
                provider,
                ctx.billable_service_tier(Some(&body)),
            )
            .expect("explicit generic standard cost");
            assert_eq!(explicit_standard, standard);
        }
    }
}

#[test]
fn streamed_context_retains_served_default_and_unary_body_takes_precedence() {
    let ctx = CallContext {
        request_body: Arc::new(json!({"service_tier":"priority"})),
        served_service_tier: Some("default".into()),
        ..CallContext::for_test("test")
    };
    assert_eq!(ctx.billable_service_tier(None), None);
    assert_eq!(
        ctx.billable_service_tier(Some(&json!({"service_tier":"flex"}))),
        Some("flex")
    );
}

#[test]
fn null_service_tier_does_not_hide_native_speed() {
    for service_tier in [Value::Null, json!(42)] {
        let ctx = CallContext {
            request_body: Arc::new(json!({"service_tier":service_tier,"speed":"fast"})),
            ..CallContext::for_test("test")
        };
        assert_eq!(ctx.requested_service_tier(), Some("fast"));
        assert_eq!(ctx.billable_service_tier(None), Some("fast"));
        assert_eq!(
            ctx.billable_service_tier(Some(&json!({"usage":{"speed":"standard"}}))),
            None
        );
    }
}

#[test]
fn service_rates_cover_all_buckets_and_context_boundary() {
    let rows = parse_provider_prices(
        "test",
        r#"
["service-test"]
input_cost_per_token = 1.0
output_cost_per_token = 2.0
cache_read_input_token_cost = 0.1
cache_creation_input_token_cost = 1.25
cache_creation_input_token_cost_above_1hr = 2.0
input_cost_per_audio_token = 3.0
input_cost_per_token_above_100_tokens = 2.0
output_cost_per_token_above_100_tokens = 4.0
input_cost_per_token_priority = 5.0
output_cost_per_token_priority = 10.0
cache_read_input_token_cost_priority = 0.5
cache_creation_input_token_cost_priority = 6.25
cache_creation_input_token_cost_above_1hr_priority = 8.0
input_cost_per_audio_token_priority = 9.0
output_cost_per_reasoning_token_priority = 12.0
input_cost_per_token_above_100_tokens_priority = 7.0
output_cost_per_token_priority_above_100_tokens = 14.0
cache_read_input_token_cost_above_100_tokens_priority = 0.7
cache_creation_input_token_cost_above_100_tokens_priority = 8.75
cache_creation_input_token_cost_above_1hr_above_100_tokens_priority = 11.0
"#,
    );
    let pricing = &rows["service-test"];
    let base = pricing.rates_for_service_tier(80, 20, Some("priority"));
    assert_eq!(
        (
            base.input,
            base.output,
            base.cache_read,
            base.cache_creation,
            base.cache_creation_1h
        ),
        (5.0, 10.0, 0.5, 6.25, 8.0)
    );
    assert_eq!((base.audio, base.reasoning), (9.0, 12.0));
    let above = pricing.rates_for_service_tier(81, 20, Some("priority"));
    assert_eq!(
        (
            above.input,
            above.output,
            above.cache_read,
            above.cache_creation,
            above.cache_creation_1h
        ),
        (7.0, 14.0, 0.7, 8.75, 11.0)
    );
    let usage = UsageSummary {
        prompt_tokens: 81,
        completion_tokens: 10,
        cache_read_input_tokens: Some(30),
        cache_creation_input_tokens: Some(20),
        cache_creation_1h_input_tokens: Some(5),
        audio_input_tokens: Some(1),
        reasoning_tokens: Some(4),
        ..Default::default()
    };
    let cost = super::super::cost_from_pricing(pricing_for_tier(pricing, Some("priority")), &usage)
        .unwrap();
    let expected =
        50.0 * 7.0 + 30.0 * 0.7 + 15.0 * 8.75 + 5.0 * 11.0 + 9.0 + 6.0 * 14.0 + 4.0 * 12.0;
    assert!((cost - expected).abs() < 1e-9);
    assert_eq!(
        pricing
            .rates_for_service_tier(101, 0, Some("unpriced"))
            .input,
        2.0
    );
    assert_eq!(
        pricing.rates_for_service_tier(101, 0, Some("fast")).input,
        7.0
    );
}

#[test]
fn sparse_service_prices_inherit_standard_context_fields_and_keep_explicit_zero() {
    let rows = parse_provider_prices(
        "test",
        r#"
["sparse-service-test"]
input_cost_per_token = 1.0
output_cost_per_token = 2.0
cache_read_input_token_cost = 0.1
input_cost_per_token_above_100_tokens = 3.0
output_cost_per_token_above_100_tokens = 4.0
input_cost_per_token_flex = 0.5
cache_read_input_token_cost_flex = 0.0
output_cost_per_token_above_100_tokens_flex = 2.0
"#,
    );
    let pricing = &rows["sparse-service-test"];
    let base = pricing.rates_for_service_tier(100, 0, Some("flex"));
    assert_eq!((base.input, base.output, base.cache_read), (0.5, 2.0, 0.0));
    let above = pricing.rates_for_service_tier(101, 0, Some("flex"));
    assert_eq!(
        (above.input, above.output, above.cache_read),
        (3.0, 2.0, 0.0)
    );
}

#[test]
fn batch_suffixes_price_each_bucket_at_its_own_context_threshold() {
    let raw = r#"
["batch-context-test"]
input_cost_per_token = 2.0
output_cost_per_token = 4.0
output_cost_per_reasoning_token = 4.0
input_cost_per_audio_token = 8.0
input_cost_per_token_above_100_tokens = 20.0
output_cost_per_token_above_100_tokens = 40.0
input_cost_per_token_above_200_tokens = 200.0
output_cost_per_token_above_200_tokens = 400.0
input_cost_per_token_batches = "0.5"
output_cost_per_token_batches = 1
input_cost_per_audio_token_batches = 2.0
cache_read_input_token_cost_batches = 0.0
cache_creation_input_token_cost_batches = 0.7
cache_creation_input_token_cost_above_1hr_batches = 0.9
input_cost_per_token_batches_above_100_tokens = 1.0
output_cost_per_token_above_200_tokens_batches = 2.0
cache_creation_input_token_cost_above_100_tokens_batches = 0.8
"#;
    let rows = parse_provider_prices("test", raw);
    let pricing = &rows["batch-context-test"];
    let at_boundary = pricing.rates_for_service_tier(100, 0, Some("batches"));
    assert_eq!((at_boundary.input, at_boundary.output), (0.5, 1.0));
    let first = pricing.rates_for_service_tier(101, 0, Some("batch"));
    assert_eq!((first.input, first.output), (1.0, 1.0));
    let second = pricing.rates_for_service_tier(201, 0, Some("batches"));
    assert_eq!(
        (second.input, second.output, second.reasoning),
        (1.0, 2.0, 2.0)
    );
    assert_eq!(
        (
            second.cache_read,
            second.cache_creation,
            second.cache_creation_1h,
            second.audio
        ),
        (0.0, 0.8, 0.9, 2.0)
    );
    assert_eq!(pricing.rates_for_prompt(201, 0).input, 200.0);
    let usage = UsageSummary {
        prompt_tokens: 201,
        completion_tokens: 10,
        cache_read_input_tokens: Some(50),
        cache_creation_input_tokens: Some(20),
        cache_creation_1h_input_tokens: Some(5),
        audio_input_tokens: Some(1),
        reasoning_tokens: Some(4),
        ..Default::default()
    };
    let cost = super::super::cost_from_pricing(pricing_for_tier(pricing, Some("batches")), &usage)
        .expect("batch cost");
    assert!((cost - (150.0 + 15.0 * 0.8 + 5.0 * 0.9 + 2.0 + 10.0 * 2.0)).abs() < 1e-9);

    let inclusive = parse_provider_prices("xai", raw);
    let at_boundary =
        inclusive["batch-context-test"].rates_for_service_tier(100, 0, Some("batches"));
    assert_eq!((at_boundary.input, at_boundary.output), (1.0, 1.0));
}

#[test]
fn sparse_batch_rates_preserve_table_cache_policy_and_unpriced_context_fields() {
    let raw = r#"
["sparse-batch-test"]
input_cost_per_token = 2.0
output_cost_per_token = 4.0
input_cost_per_token_above_100_tokens = 20.0
output_cost_per_token_above_100_tokens = 40.0
input_cost_per_token_batches = 1.0
"#;
    for (provider, cache_read) in [("test", 0.0), ("fireworks_ai", 0.5)] {
        let rows = parse_provider_prices(provider, raw);
        let pricing = &rows["sparse-batch-test"];
        let rates = pricing.rates_for_service_tier(101, 0, Some("batches"));
        assert_eq!((rates.input, rates.output), (1.0, 40.0));
        assert_eq!(rates.cache_read, cache_read);
        assert_eq!((rates.cache_creation, rates.cache_creation_1h), (0.0, 0.0));
        assert!(
            pricing.service_tiers["batches"]
                .cache_read_input_token_cost
                .is_none()
        );
    }
}

#[test]
fn unsupported_batch_fields_are_not_whitelisted_as_supported_rates() {
    for key in [
        "input_cost_per_image_token_batches",
        "input_cost_per_token_above_200ktokens_batches",
        "input_cost_per_audio_token_above_200k_tokens_batches",
    ] {
        assert!(
            !is_service_price_key(key),
            "{key} must reach the warning path"
        );
    }
    let rows = parse_provider_prices(
        "test",
        r#"
["unsupported-batch-test"]
input_cost_per_token = 1.0
output_cost_per_token = 2.0
input_cost_per_image_token_batches = 3.0
input_cost_per_token_above_200ktokens_batches = 4.0
output_cost_per_token_batches = "not-a-rate"
"#,
    );
    assert!(
        !rows["unsupported-batch-test"]
            .service_tiers
            .contains_key("batches")
    );
}

#[test]
fn batch_metadata_uses_google_flex_rates_once_including_audio_cache_and_thinking() {
    let body = json!({"usageMetadata":{"trafficType":"BATCH"}});
    let ctx = CallContext {
        request_body: Arc::new(json!({"service_tier":"priority"})),
        ..CallContext::for_test("gemini-2.5-flash")
    };
    let usage = UsageSummary {
        prompt_tokens: 1_000,
        completion_tokens: 100,
        cache_read_input_tokens: Some(400),
        audio_input_tokens: Some(100),
        reasoning_tokens: Some(40),
        ..Default::default()
    };
    let expected = 500.0 * 1.5e-7 + 400.0 * 3e-8 + 100.0 * 5e-7 + 100.0 * 1.25e-6;
    for provider in ["gemini", "vertex_ai"] {
        let batch = calculate_cost_for_service_tier(
            &ctx.model,
            &usage,
            None,
            provider,
            ctx.billable_service_tier(Some(&body)),
        )
        .expect("Google batch cost");
        let flex =
            calculate_cost_for_service_tier(&ctx.model, &usage, None, provider, Some("flex"))
                .expect("Google flex cost");
        assert!((batch - expected).abs() < 1e-12, "{provider}: {batch}");
        assert_eq!(batch, flex, "batch rates must not get a second discount");
    }
}

#[test]
fn google_batch_long_context_and_cached_reads_use_published_rates() {
    for (model, base, long) in [
        (
            "gemini-2.5-pro",
            (6.25e-7, 5e-6, 1.25e-7),
            (1.25e-6, 7.5e-6, 2.5e-7),
        ),
        (
            "gemini-3.1-pro-preview",
            (1e-6, 6e-6, 2e-7),
            (2e-6, 9e-6, 4e-7),
        ),
    ] {
        for provider in ["gemini", "vertex_ai"] {
            let pricing =
                crate::cost::lookup_pricing_with_provider(model, provider).expect("Google model");
            let exact = pricing.rates_for_service_tier(200_000, 0, Some("batches"));
            assert_eq!((exact.input, exact.output, exact.cache_read), base);
            let above = pricing.rates_for_service_tier(200_001, 0, Some("batches"));
            assert_eq!((above.input, above.output, above.cache_read), long);
            let usage = UsageSummary {
                prompt_tokens: 200_001,
                completion_tokens: 100,
                cache_read_input_tokens: Some(100_000),
                reasoning_tokens: Some(40),
                ..Default::default()
            };
            let cost =
                calculate_cost_for_service_tier(model, &usage, None, provider, Some("batches"))
                    .expect("long-context batch cost");
            let expected =
                100_001.0 * above.input + 100_000.0 * above.cache_read + 100.0 * above.output;
            assert!(
                (cost - expected).abs() < 1e-12,
                "{provider}/{model}: {cost}"
            );
        }
    }
}

#[test]
fn published_openai_batch_prices_are_applied_without_an_extra_discount() {
    let usage = UsageSummary {
        prompt_tokens: 1_000,
        completion_tokens: 100,
        reasoning_tokens: Some(40),
        ..Default::default()
    };
    for (models, input, output) in [
        (
            &["gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"][..],
            1.25e-6,
            5e-6,
        ),
        (&["gpt-4o-2024-05-13"][..], 2.5e-6, 7.5e-6),
        (&["gpt-4o-mini", "gpt-4o-mini-2024-07-18"][..], 7.5e-8, 3e-7),
    ] {
        for model in models {
            let cost =
                calculate_cost_for_service_tier(model, &usage, None, "openai", Some("batches"))
                    .expect("published batch model");
            assert!(
                (cost - (1_000.0 * input + 100.0 * output)).abs() < 1e-12,
                "{model}"
            );
        }
    }
}

#[test]
fn batch_flex_fallback_is_google_only_and_explicit_batch_prices_win() {
    let raw = r#"
["flex-only-test"]
input_cost_per_token = 1.0
output_cost_per_token = 2.0
input_cost_per_token_flex = 0.5
["batch-and-flex-test"]
input_cost_per_token = 1.0
output_cost_per_token = 2.0
input_cost_per_token_flex = 0.5
input_cost_per_token_batches = 0.25
"#;
    for (provider, fallback) in [("openai", 1.0), ("gemini", 0.5), ("vertex_ai", 0.5)] {
        let rows = parse_provider_prices(provider, raw);
        assert_eq!(
            rows["flex-only-test"]
                .rates_for_service_tier(10, 0, Some("batches"))
                .input,
            fallback
        );
        assert_eq!(
            rows["batch-and-flex-test"]
                .rates_for_service_tier(10, 0, Some("batches"))
                .input,
            0.25
        );
    }
}

#[test]
fn distinct_or_service_specific_reasoning_prices_remain_explicit() {
    let rows = parse_provider_prices(
        "test",
        r#"
["distinct-reasoning-test"]
input_cost_per_token = 1.0
output_cost_per_token = 2.0
output_cost_per_reasoning_token = 3.0
output_cost_per_token_batches = 1.0
["batch-reasoning-test"]
input_cost_per_token = 1.0
output_cost_per_token = 2.0
output_cost_per_reasoning_token = 2.0
output_cost_per_token_batches = 1.0
output_cost_per_reasoning_token_batches = 0.25
"#,
    );
    let distinct = rows["distinct-reasoning-test"].rates_for_service_tier(10, 0, Some("batches"));
    assert_eq!((distinct.output, distinct.reasoning), (1.0, 3.0));
    let explicit = rows["batch-reasoning-test"].rates_for_service_tier(10, 0, Some("batches"));
    assert_eq!((explicit.output, explicit.reasoning), (1.0, 0.25));
}

#[test]
fn custom_token_prices_override_service_and_context_prices() {
    let override_prices = DeploymentPricing {
        input_cost_per_token: 0.001,
        output_cost_per_token: 0.002,
        ..Default::default()
    };
    let usage = UsageSummary {
        prompt_tokens: 300_000,
        completion_tokens: 10,
        ..Default::default()
    };
    let cost = calculate_cost_for_service_tier(
        "gpt-6.1-sol",
        &usage,
        Some(&override_prices),
        "openai",
        Some("priority"),
    )
    .unwrap();
    assert!((cost - (300_000.0 * 0.001 + 10.0 * 0.002)).abs() < 1e-9);
}

#[test]
fn duration_rate_precedence_bills_once_and_preserves_zero() {
    for (canonical, input, output, expected) in [
        (Some(0.2), Some(0.3), Some(0.4), 0.5),
        (None, Some(0.3), Some(0.3), 0.75),
        (None, None, Some(0.4), 1.0),
        (Some(0.0), Some(0.3), Some(0.4), 0.0),
    ] {
        let pricing = DeploymentPricing {
            cost_per_second_rate: canonical,
            input_cost_per_second: input,
            output_cost_per_second: output,
            ..Default::default()
        };
        assert_eq!(pricing.cost_per_second(2500.0), Some(expected));
    }
}

#[test]
fn token_override_precedence_also_holds_for_direct_pricing_structs() {
    let pricing = DeploymentPricing {
        input_cost_per_token: 0.001,
        output_cost_per_token: 0.002,
        cost_per_second_rate: Some(100.0),
        ..Default::default()
    };
    let usage = UsageSummary {
        prompt_tokens: 10,
        completion_tokens: 5,
        ..Default::default()
    };
    assert_eq!(pricing.cost_per_second(1000.0), None);
    assert_eq!(
        calculate_cost_for_service_tier("custom", &usage, Some(&pricing), "openai", None),
        Some(0.02)
    );
}

#[test]
fn duration_only_pricing_deserializes_canonical_and_legacy_field_names() {
    for field in ["cost_per_second", "cost_per_second_rate"] {
        let pricing: DeploymentPricing =
            serde_json::from_value(json!({(field):0.2})).expect("duration override");
        assert_eq!(pricing.input_cost_per_token, 0.0);
        assert_eq!(pricing.output_cost_per_token, 0.0);
        assert_eq!(pricing.cost_per_second_rate, Some(0.2));
        assert_eq!(pricing.cost_per_second(2_500.0), Some(0.5));
        let serialized = serde_json::to_value(&pricing).expect("serialize override");
        assert_eq!(serialized["cost_per_second"], json!(0.2));
        assert!(serialized.get("cost_per_second_rate").is_none());
        let roundtrip: DeploymentPricing = serde_json::from_value(serialized).expect("roundtrip");
        assert_eq!(roundtrip.cost_per_second(2_500.0), Some(0.5));
    }
}

#[test]
fn deserialized_token_overrides_keep_precedence_and_missing_cache_fallbacks() {
    let pricing: DeploymentPricing = serde_json::from_value(json!({
        "input_cost_per_token":0.01,
        "output_cost_per_token":0.02,
        "cost_per_second":100.0
    }))
    .expect("token and duration overrides");
    assert_eq!(pricing.cost_per_second(1_000.0), None);
    let usage = UsageSummary {
        prompt_tokens: 10,
        completion_tokens: 5,
        cache_read_input_tokens: Some(4),
        cache_creation_input_tokens: Some(3),
        cache_creation_1h_input_tokens: Some(1),
        reasoning_tokens: Some(2),
        ..Default::default()
    };
    let cost = calculate_cost_for_service_tier(
        "custom",
        &usage,
        Some(&pricing),
        "gemini",
        Some("batches"),
    )
    .expect("flat token override");
    assert!((cost - (13.0 * 0.01 + 5.0 * 0.02)).abs() < 1e-12);

    let zero_cache: DeploymentPricing = serde_json::from_value(json!({
        "cache_read_input_token_cost":0.0,
        "cost_per_second":100.0
    }))
    .expect("explicit zero token override");
    assert_eq!(zero_cache.cost_per_second(1_000.0), None);
}

#[test]
fn serialized_pricing_preserves_all_optional_token_and_duration_fields() {
    let pricing = DeploymentPricing {
        input_cost_per_token: 1.0,
        output_cost_per_token: 2.0,
        cache_read_input_token_cost: Some(0.0),
        cache_creation_input_token_cost: Some(1.25),
        cache_creation_input_token_cost_above_1hr: Some(2.0),
        output_cost_per_reasoning_token: Some(3.0),
        cost_per_second_rate: Some(4.0),
        input_cost_per_second: Some(5.0),
        output_cost_per_second: Some(6.0),
        ..Default::default()
    };
    let encoded = serde_json::to_value(&pricing).expect("serialize");
    let decoded: DeploymentPricing = serde_json::from_value(encoded.clone()).expect("deserialize");
    assert_eq!(serde_json::to_value(decoded).expect("reserialize"), encoded);
}

#[test]
fn token_pricing_and_audio_modes_do_not_bill_elapsed_time() {
    let rows = parse_provider_prices(
        "test",
        r#"
["elapsed-test"]
input_cost_per_token = 0.0
output_cost_per_token = 0.0
cost_per_second = 0.25
input_cost_per_second = 0.5
output_cost_per_second = 0.5
mode = "chat"
["token-and-duration-test"]
input_cost_per_token = 1.0
output_cost_per_token = 2.0
input_cost_per_second = 0.5
mode = "chat"
["audio-duration-test"]
input_cost_per_token = 0.0
output_cost_per_token = 0.0
input_cost_per_second = 0.5
mode = "audio_transcription"
"#,
    );
    assert_eq!(rows["elapsed-test"].cost_per_second(2000.0), Some(0.5));
    for model in ["token-and-duration-test", "audio-duration-test"] {
        assert!(!rows[model].is_per_second_priced());
        assert_eq!(rows[model].cost_per_second(2000.0), None);
    }
}

#[test]
fn every_vendored_service_rate_reaches_the_resolved_price_table() {
    for (model, overlays) in price_overlays() {
        let pricing = crate::cost::price_table()
            .get(model)
            .expect("service model is priced");
        for tier in TIERS {
            let Some(rates) = overlays.get(*tier).and_then(toml::Value::as_table) else {
                continue;
            };
            let resolved = pricing
                .service_tiers
                .get(*tier)
                .expect("service prices are loaded");
            for (key, value) in rates {
                let expected = tier_rate_value(value).expect("vendored service rate is numeric");
                let actual = if let Some((base, threshold)) = super::super::parse_threshold_key(key)
                {
                    let context = resolved
                        .tiers
                        .iter()
                        .find(|context| context.threshold_tokens == threshold)
                        .expect("service context threshold is loaded");
                    match base {
                        "input_cost_per_token" => context.input_cost_per_token,
                        "output_cost_per_token" => context.output_cost_per_token,
                        "cache_read_input_token_cost" => context.cache_read_input_token_cost,
                        "cache_creation_input_token_cost" => {
                            context.cache_creation_input_token_cost
                        }
                        "cache_creation_input_token_cost_above_1hr" => {
                            context.cache_creation_input_token_cost_above_1hr
                        }
                        _ => panic!("unhandled context rate {key}"),
                    }
                } else {
                    match key.as_str() {
                        "input_cost_per_token" => Some(resolved.input_cost_per_token),
                        "output_cost_per_token" => Some(resolved.output_cost_per_token),
                        "cache_read_input_token_cost" => resolved.cache_read_input_token_cost,
                        "cache_creation_input_token_cost" => {
                            resolved.cache_creation_input_token_cost
                        }
                        "cache_creation_input_token_cost_above_1hr" => {
                            resolved.cache_creation_input_token_cost_above_1hr
                        }
                        "input_cost_per_audio_token" => resolved.input_cost_per_audio_token,
                        "output_cost_per_reasoning_token" => {
                            resolved.output_cost_per_reasoning_token
                        }
                        _ => panic!("unhandled service rate {key}"),
                    }
                };
                assert_eq!(actual, Some(expected), "{model}/{tier}: {key}");
            }
        }
    }
}
