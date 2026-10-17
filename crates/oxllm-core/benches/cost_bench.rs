use criterion::{Criterion, criterion_group, criterion_main};
use oxllm_core::cost::{
    DeploymentPricing, calculate_cost, calculate_cost_with_override, calculate_cost_with_provider,
    lookup_pricing, lookup_pricing_with_provider,
};
use oxllm_core::hook::UsageSummary;

fn usage(prompt: u64, completion: u64) -> UsageSummary {
    UsageSummary {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        cache_read_input_tokens: None,
        cache_creation_input_tokens: None,
        cache_creation_1h_input_tokens: None,
        audio_input_tokens: None,
        reasoning_tokens: None,
        citation_tokens: None,
        web_search_requests: None,
        browser_open_requests: None,
    }
}

fn usage_with_cache(
    prompt: u64,
    completion: u64,
    cache_read: u64,
    cache_creation: u64,
) -> UsageSummary {
    UsageSummary {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        cache_read_input_tokens: Some(cache_read),
        cache_creation_input_tokens: Some(cache_creation),
        cache_creation_1h_input_tokens: None,
        audio_input_tokens: None,
        reasoning_tokens: None,
        citation_tokens: None,
        web_search_requests: None,
        browser_open_requests: None,
    }
}

fn usage_full(
    prompt: u64,
    completion: u64,
    cache_read: u64,
    cache_creation: u64,
    cache_creation_1h: u64,
    reasoning: u64,
) -> UsageSummary {
    UsageSummary {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        cache_read_input_tokens: Some(cache_read),
        cache_creation_input_tokens: Some(cache_creation),
        cache_creation_1h_input_tokens: Some(cache_creation_1h),
        audio_input_tokens: None,
        reasoning_tokens: Some(reasoning),
        citation_tokens: None,
        web_search_requests: None,
        browser_open_requests: None,
    }
}

fn lookup_pricing_benches(c: &mut Criterion) {
    c.bench_function("lookup_pricing_known", |b| {
        b.iter(|| lookup_pricing("gpt-4o"));
    });
    c.bench_function("lookup_pricing_with_prefix", |b| {
        b.iter(|| lookup_pricing("openai/gpt-4o"));
    });
    c.bench_function("lookup_pricing_unknown", |b| {
        b.iter(|| lookup_pricing("nonexistent-model-xyz"));
    });
    // Regression guard for the provider-aware billing path: must stay
    // allocation-free in the common case (stack-buffer concat).
    c.bench_function("lookup_pricing_with_provider", |b| {
        b.iter(|| lookup_pricing_with_provider("gemini-2.5-flash", "gemini"));
    });
}

fn calculate_cost_benches(c: &mut Criterion) {
    c.bench_function("calculate_cost_known_small", |b| {
        b.iter(|| calculate_cost("gpt-4o", &usage(100, 50)));
    });
    c.bench_function("calculate_cost_known_large", |b| {
        b.iter(|| calculate_cost("gpt-4o", &usage(100_000, 50_000)));
    });
    c.bench_function("calculate_cost_unknown", |b| {
        b.iter(|| calculate_cost("nonexistent-model-xyz", &usage(100, 50)));
    });
    c.bench_function("calculate_cost_with_prefix", |b| {
        b.iter(|| calculate_cost("openai/gpt-4o-mini", &usage(1000, 500)));
    });
}

fn calculate_cost_cache_benches(c: &mut Criterion) {
    c.bench_function("calculate_cost_cache_read_creation", |b| {
        b.iter(|| {
            calculate_cost(
                "claude-sonnet-4-5",
                &usage_with_cache(10_000, 5_000, 5_000, 2_000),
            )
        });
    });
    c.bench_function("calculate_cost_full_breakdown", |b| {
        b.iter(|| {
            calculate_cost(
                "claude-sonnet-4-5",
                &usage_full(50_000, 10_000, 20_000, 10_000, 5_000, 3_000),
            )
        });
    });
    c.bench_function("calculate_cost_tiered_above_200k", |b| {
        b.iter(|| calculate_cost("claude-sonnet-4-5", &usage(250_000, 50_000)));
    });
}

fn calculate_cost_override_benches(c: &mut Criterion) {
    let dp = DeploymentPricing {
        input_cost_per_token: 5e-6,
        output_cost_per_token: 1.5e-5,
        cache_read_input_token_cost: Some(5e-7),
        cache_creation_input_token_cost: Some(6.25e-6),
        output_cost_per_reasoning_token: None,
        ..Default::default()
    };
    c.bench_function("calculate_cost_with_override", |b| {
        b.iter(|| calculate_cost_with_override("custom-model", &usage(1000, 500), Some(&dp)));
    });
    c.bench_function("calculate_cost_with_override_cache", |b| {
        b.iter(|| {
            calculate_cost_with_override(
                "custom-model",
                &usage_with_cache(10_000, 5_000, 5_000, 2_000),
                Some(&dp),
            )
        });
    });
    c.bench_function("calculate_cost_with_provider", |b| {
        b.iter(|| {
            calculate_cost_with_provider("claude-sonnet-4-5", &usage(1000, 500), "anthropic")
        });
    });
}

criterion_group!(
    benches,
    lookup_pricing_benches,
    calculate_cost_benches,
    calculate_cost_cache_benches,
    calculate_cost_override_benches,
);
criterion_main!(benches);
