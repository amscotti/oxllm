use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::Value;

use super::{
    CostTier, ModelPricing, ModelPricingRow, TIER_BASE_FIELDS, extract_cost_tiers,
    parse_threshold_key, set_tier_rate, tier_rate_value,
};

const BATCH_TIER: &str = "batches";
const TIERS: &[&str] = &[
    "priority",
    "flex",
    "fast",
    "ultrafast",
    "balanced",
    BATCH_TIER,
];

/// Resolve the billable tier. A recognized served tier wins, including an
/// explicit standard/default response that overrides a premium request.
/// Batch pricing is accepted only from served metadata, never a request tier.
pub fn resolve_service_tier(requested: Option<&str>, served: Option<&str>) -> Option<&'static str> {
    let served = served.and_then(known_served_tier);
    let tier = served.or_else(|| requested.and_then(known_tier))?;
    if tier.eq_ignore_ascii_case("default") || tier.eq_ignore_ascii_case("standard") {
        None
    } else {
        Some(tier)
    }
}

fn known_tier(tier: &str) -> Option<&'static str> {
    let tier = tier.trim();
    if tier.eq_ignore_ascii_case("default") || tier.eq_ignore_ascii_case("standard") {
        return Some("default");
    }
    TIERS
        .iter()
        .copied()
        .filter(|known| *known != BATCH_TIER)
        .find(|known| tier.eq_ignore_ascii_case(known))
}

fn known_served_tier(tier: &str) -> Option<&'static str> {
    let tier = tier.trim();
    if tier.eq_ignore_ascii_case("batch") || tier.eq_ignore_ascii_case(BATCH_TIER) {
        Some(BATCH_TIER)
    } else {
        known_tier(tier)
    }
}

/// Read served-tier metadata from a response or stream event without requiring
/// token usage on the same event. Native Responses and Messages envelopes are
/// supported alongside Chat Completions and Gemini traffic metadata.
/// Google `BATCH` traffic yields the internal `"batches"` pricing selector;
/// a plain `service_tier: "batch"` is not evidence of a batch job.
/// Explicit generic served fields win. Batch traffic precedes Google's native
/// `serviceTier`, whose standard/unspecified processing tier does not exclude
/// an asynchronous batch job. Other traffic types remain native-tier fallbacks.
pub fn response_service_tier(response: &Value) -> Option<&'static str> {
    for pointer in [
        "/service_tier",
        "/speed",
        "/response/service_tier",
        "/response/speed",
        "/message/service_tier",
        "/message/speed",
        "/usage/service_tier",
        "/usage/speed",
        "/message/usage/service_tier",
        "/message/usage/speed",
    ] {
        if let Some(tier) = response
            .pointer(pointer)
            .and_then(Value::as_str)
            .and_then(known_tier)
        {
            return Some(tier);
        }
    }
    let traffic_tier = response_traffic_service_tier(response);
    if traffic_tier == Some(BATCH_TIER) {
        // Native serviceTier has no batch variant; its processing default
        // must not downgrade an explicitly classified batch job to on-demand.
        return traffic_tier;
    }
    response
        .pointer("/usageMetadata/serviceTier")
        .and_then(Value::as_str)
        .and_then(|tier| {
            if tier.trim().eq_ignore_ascii_case("unspecified") {
                Some("default")
            } else {
                known_tier(tier)
            }
        })
        .or(traffic_tier)
}

fn response_traffic_service_tier(response: &Value) -> Option<&'static str> {
    for pointer in ["/usageMetadata/trafficType", "/usage/traffic_type"] {
        let Some(traffic) = response.pointer(pointer).and_then(Value::as_str) else {
            continue;
        };
        for (native, tier) in [
            ("ON_DEMAND_PRIORITY", "priority"),
            ("PRIORITY", "priority"),
            ("ON_DEMAND_FLEX", "flex"),
            ("FLEX", "flex"),
            ("BATCH", BATCH_TIER),
            ("ON_DEMAND", "default"),
            ("STANDARD", "default"),
        ] {
            if traffic.trim().eq_ignore_ascii_case(native) {
                return Some(tier);
            }
        }
    }
    None
}

pub(super) fn pricing_for_tier<'a>(
    pricing: &'a ModelPricing,
    tier: Option<&str>,
) -> &'a ModelPricing {
    let Some(tier) = tier.and_then(known_served_tier) else {
        return pricing;
    };
    let resolved = pricing
        .service_tiers
        .get(tier)
        .or_else(|| {
            (tier == "fast")
                .then(|| pricing.service_tiers.get("priority"))
                .flatten()
        })
        .or_else(|| {
            // Google publishes the same token prices for Flex and Batch.
            // Prefer a separate batch entry when present and never halve an
            // already-discounted rate (cached reads keep their published rate).
            (tier == BATCH_TIER && matches!(pricing.provider.as_str(), "gemini" | "vertex_ai"))
                .then(|| pricing.service_tiers.get("flex"))
                .flatten()
        });
    if tier == BATCH_TIER && resolved.is_none() {
        tracing::warn!(
            provider = %pricing.provider,
            "batch pricing is unavailable for this model; using standard rates"
        );
    }
    resolved.unwrap_or(pricing)
}

static OVERLAYS: OnceLock<HashMap<String, toml::Value>> = OnceLock::new();

pub(super) fn price_overlays() -> &'static HashMap<String, toml::Value> {
    OVERLAYS.get_or_init(|| {
        let table: toml::Table = include_str!("../../data/prices/service_tiers.toml")
            .parse()
            .expect("embedded service price table must parse");
        let mut overlays = HashMap::new();
        for (model, values) in table {
            if let Some(aliases) = values.get("aliases").and_then(toml::Value::as_array) {
                for alias in aliases {
                    let alias = alias
                        .as_str()
                        .expect("service price alias must be a string");
                    assert!(
                        overlays.insert(alias.to_string(), values.clone()).is_none(),
                        "duplicate service price alias {alias}"
                    );
                }
            }
            assert!(
                overlays.insert(model.clone(), values).is_none(),
                "duplicate service price entry {model}"
            );
        }
        overlays
    })
}

fn normalized_price_key(key: &str, tier: &str) -> Option<String> {
    let suffix = format!("_{tier}");
    let normalized = if let Some(base) = key.strip_suffix(&suffix) {
        base.to_string()
    } else {
        key.replace(&format!("{suffix}_above_"), "_above_")
    };
    (normalized != key && price_field(&normalized)).then_some(normalized)
}

fn price_field(key: &str) -> bool {
    TIER_BASE_FIELDS.contains(&key)
        || parse_threshold_key(key).is_some_and(|(base, _)| TIER_BASE_FIELDS.contains(&base))
        || matches!(
            key,
            "input_cost_per_audio_token" | "output_cost_per_reasoning_token"
        )
}

pub(super) fn is_service_price_key(key: &str) -> bool {
    TIERS
        .iter()
        .any(|tier| normalized_price_key(key, tier).is_some())
}

fn context_rate(tier: &CostTier, field: &str) -> Option<f64> {
    match field {
        "input_cost_per_token" => tier.input_cost_per_token,
        "output_cost_per_token" => tier.output_cost_per_token,
        "cache_read_input_token_cost" => tier.cache_read_input_token_cost,
        "cache_creation_input_token_cost" => tier.cache_creation_input_token_cost,
        "cache_creation_input_token_cost_above_1hr" => {
            tier.cache_creation_input_token_cost_above_1hr
        }
        _ => None,
    }
}

/// Batch thresholds are per bucket: a published flat batch rate must not be
/// replaced by a standard long-context rate, nor may another bucket's higher
/// threshold hide a lower batch threshold. Unpriced buckets retain the normal
/// table fallback, including absent cache rates.
fn batch_context_tiers(base: &toml::Table, rates: &toml::Table) -> Vec<CostTier> {
    let standard = extract_cost_tiers(&toml::Value::Table(base.clone()));
    let batches = extract_cost_tiers(&toml::Value::Table(rates.clone()));
    let mut thresholds: Vec<_> = standard
        .iter()
        .chain(&batches)
        .map(|tier| tier.threshold_tokens)
        .collect();
    thresholds.sort_unstable_by_key(|threshold| std::cmp::Reverse(*threshold));
    thresholds.dedup();
    thresholds
        .into_iter()
        .map(|threshold_tokens| {
            let standard_tier = standard
                .iter()
                .find(|tier| tier.threshold_tokens <= threshold_tokens);
            let mut tier = CostTier {
                threshold_tokens,
                input_cost_per_token: None,
                output_cost_per_token: None,
                cache_read_input_token_cost: None,
                cache_creation_input_token_cost: None,
                cache_creation_input_token_cost_above_1hr: None,
            };
            for field in TIER_BASE_FIELDS {
                let rate = batches
                    .iter()
                    .filter(|tier| tier.threshold_tokens <= threshold_tokens)
                    .find_map(|tier| context_rate(tier, field))
                    .or_else(|| rates.get(*field).and_then(tier_rate_value))
                    .or_else(|| standard_tier.and_then(|tier| context_rate(tier, field)))
                    .or_else(|| base.get(*field).and_then(tier_rate_value));
                if let Some(rate) = rate {
                    set_tier_rate(&mut tier, field, rate);
                }
            }
            tier
        })
        .collect()
}

pub(super) fn parse_pricing(
    provider: &str,
    base: &toml::Value,
    overlays: Option<&toml::Value>,
) -> HashMap<String, ModelPricing> {
    let Some(base) = base.as_table() else {
        return HashMap::new();
    };
    for key in base.keys() {
        if key.contains("cost")
            && (key.ends_with("_batches") || key.contains("_batches_above_"))
            && normalized_price_key(key, BATCH_TIER).is_none()
        {
            tracing::warn!(provider, key, "skipping unsupported batch price key");
        }
    }
    let mut prices = HashMap::new();
    for tier in TIERS {
        let mut rates = toml::Table::new();
        for (key, value) in base {
            if let Some(key) = normalized_price_key(key, tier) {
                rates.insert(key, value.clone());
            }
        }
        if let Some(overlay) = overlays
            .and_then(|values| values.get(*tier))
            .and_then(toml::Value::as_table)
        {
            rates.extend(
                overlay
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
        }
        if rates.is_empty() {
            continue;
        }
        rates.retain(|key, value| {
            if !price_field(key) {
                tracing::warn!(
                    provider,
                    tier,
                    key,
                    "skipping unsupported service price key"
                );
                return false;
            }
            if let Some(rate) = tier_rate_value(value) {
                *value = toml::Value::Float(rate);
                true
            } else {
                tracing::warn!(
                    provider,
                    tier,
                    key,
                    "skipping non-numeric service price rate"
                );
                false
            }
        });
        if rates.is_empty() {
            continue;
        }
        let mut resolved = base.clone();
        // Google tiers and batch jobs bill thinking at the output rate unless
        // a distinct reasoning price is published. A copied standard output
        // rate must not defeat the selected tier's output price.
        if (*tier == BATCH_TIER || matches!(provider, "gemini" | "vertex_ai"))
            && rates.contains_key("output_cost_per_token")
            && !rates.contains_key("output_cost_per_reasoning_token")
            && base
                .get("output_cost_per_reasoning_token")
                .and_then(tier_rate_value)
                == base.get("output_cost_per_token").and_then(tier_rate_value)
        {
            resolved.remove("output_cost_per_reasoning_token");
        }
        resolved.extend(
            rates
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        let batch_tiers = (*tier == BATCH_TIER).then(|| batch_context_tiers(base, &rates));
        let value = toml::Value::Table(resolved);
        let tiers = batch_tiers.unwrap_or_else(|| extract_cost_tiers(&value));
        match value.try_into::<ModelPricingRow>() {
            Ok(row) => {
                prices.insert(
                    (*tier).to_string(),
                    ModelPricing::from_row(provider, row, tiers),
                );
            }
            Err(error) => {
                tracing::warn!(provider, tier, %error, "skipping malformed service price entry")
            }
        }
    }
    prices
}

#[cfg(test)]
mod tests;
