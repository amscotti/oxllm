//! Cost calculation for LLM requests.
//!
//! This module is **pure**: it reads the vendored per-provider price tables
//! (embedded at compile time via `include_str!` from `data/prices/*.toml`)
//! and computes the cost of a request from its [`UsageSummary`]. Pricing
//! overrides per-deployment should come from `Deployment` (the host consults
//! both).
//!
//! Each TOML file is one provider (`openai.toml`, `deepseek.toml`, …). The
//! provider slug is the filename stem and is injected at load time — model
//! rows do not carry a provider field.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::hook::UsageSummary;

mod service_tiers;
pub use service_tiers::{resolve_service_tier, response_service_tier};

/// Embedded price shards: `(provider_slug, toml_source)`.
///
/// Add a new provider by dropping `data/prices/<slug>.toml` and one line here.
const PRICE_SHARDS: &[(&str, &str)] = &[
    ("ai21", include_str!("../data/prices/ai21.toml")),
    ("anthropic", include_str!("../data/prices/anthropic.toml")),
    ("azure_ai", include_str!("../data/prices/azure_ai.toml")),
    ("baseten", include_str!("../data/prices/baseten.toml")),
    ("bedrock", include_str!("../data/prices/bedrock.toml")),
    (
        "bedrock_mantle",
        include_str!("../data/prices/bedrock_mantle.toml"),
    ),
    ("cerebras", include_str!("../data/prices/cerebras.toml")),
    ("cloudflare", include_str!("../data/prices/cloudflare.toml")),
    ("cohere", include_str!("../data/prices/cohere.toml")),
    ("dashscope", include_str!("../data/prices/dashscope.toml")),
    ("databricks", include_str!("../data/prices/databricks.toml")),
    ("deepinfra", include_str!("../data/prices/deepinfra.toml")),
    ("deepseek", include_str!("../data/prices/deepseek.toml")),
    (
        "fireworks_ai",
        include_str!("../data/prices/fireworks_ai.toml"),
    ),
    ("gemini", include_str!("../data/prices/gemini.toml")),
    ("groq", include_str!("../data/prices/groq.toml")),
    ("jina_ai", include_str!("../data/prices/jina_ai.toml")),
    ("meta", include_str!("../data/prices/meta.toml")),
    ("mistral", include_str!("../data/prices/mistral.toml")),
    ("nvidia_nim", include_str!("../data/prices/nvidia_nim.toml")),
    ("ollama", include_str!("../data/prices/ollama.toml")),
    ("openai", include_str!("../data/prices/openai.toml")),
    ("openrouter", include_str!("../data/prices/openrouter.toml")),
    ("perplexity", include_str!("../data/prices/perplexity.toml")),
    ("replicate", include_str!("../data/prices/replicate.toml")),
    ("sambanova", include_str!("../data/prices/sambanova.toml")),
    (
        "together_ai",
        include_str!("../data/prices/together_ai.toml"),
    ),
    ("vertex_ai", include_str!("../data/prices/vertex_ai.toml")),
    ("voyage", include_str!("../data/prices/voyage.toml")),
    ("watsonx", include_str!("../data/prices/watsonx.toml")),
    ("xai", include_str!("../data/prices/xai.toml")),
    #[cfg(test)]
    (
        "test_tiers",
        include_str!("../data/prices/cost_tiers_test.toml"),
    ),
];

/// Per-deployment pricing override. When present, used in place of the
/// embedded price-table entry for cost calculation. Cache rates fall back
/// to `input_cost_per_token` when `None` (after table inheritance — see
/// [`OxllmParams::pricing_override`](crate::deployment::OxllmParams::pricing_override)).
///
/// An override with an empty [`DeploymentPricing::tiers`] list is flat: it
/// bypasses the price table's context-length tiers. Tier keys parsed from
/// the deployment (`input_cost_per_token_above_272k_tokens` and the other
/// `*_above_*_tokens` spellings) fill `tiers`. Those rates apply to every
/// token type once the prompt is past the threshold. The boundary is
/// exclusive unless [`DeploymentPricing::tier_threshold_is_inclusive`] is
/// set: 272_000 stays on the base rate and 272_001 uses the tier. A `None`
/// tier rate falls back to the override's base rate.
///
/// Duration overrides (`cost_per_second`, `input_cost_per_second`, or
/// `output_cost_per_second`)
/// bill by request duration instead of tokens; when one is set the
/// token-based [`calculate_cost_with_override`] family returns `None` and
/// the billing path accrues spend through [`Self::cost_per_second`]
/// (per-token rates win when both are configured).
///
/// Serialization uses `cost_per_second`; deserialization also accepts the
/// legacy internal spelling `cost_per_second_rate`. Omitted fields default so
/// duration-only overrides need not supply token rates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DeploymentPricing {
    /// Cost per input (prompt) token.
    pub input_cost_per_token: f64,
    /// Cost per output (completion) token.
    pub output_cost_per_token: f64,
    /// Cost per cached-read input token (prompt cache hit). Falls back to
    /// `input_cost_per_token` when `None`.
    pub cache_read_input_token_cost: Option<f64>,
    /// Cost per cache-creation input token (first write to the cache).
    /// Falls back to `input_cost_per_token` when `None`.
    pub cache_creation_input_token_cost: Option<f64>,
    /// Cost per reasoning (thinking) output token. Falls back to
    /// `output_cost_per_token` when `None`. Reasoning tokens are a subset
    /// of `completion_tokens`.
    pub output_cost_per_reasoning_token: Option<f64>,
    /// Cost per cache-creation input token in the 1h ephemeral bucket.
    /// Populated by table inheritance (the
    /// `_inherit_builtin_cache_pricing` back-fills the 1h rate alongside
    /// the 5m rates); falls back to `cache_creation_input_token_cost` (then
    /// the input rate) when `None`.
    pub cache_creation_input_token_cost_above_1hr: Option<f64>,
    /// A single cost per second of request duration. Takes precedence over
    /// the legacy input/output duration-rate fields.
    #[serde(rename = "cost_per_second", alias = "cost_per_second_rate")]
    pub cost_per_second_rate: Option<f64>,
    /// Legacy cost per second of request duration. Used when
    /// the canonical duration rate is unset.
    pub input_cost_per_second: Option<f64>,
    /// Legacy duration rate used only when both the canonical and input
    /// duration rates are unset. The same elapsed time is billed once.
    pub output_cost_per_second: Option<f64>,
    /// Context-length tiers parsed from the deployment's `*_above_*_tokens`
    /// keys, highest threshold first. Empty means the override is flat and
    /// ignores table tiers.
    pub tiers: Vec<CostTier>,
    /// When true, a prompt whose size equals `threshold_tokens` uses the
    /// tier. Deployment overrides default to exclusive (`false`).
    pub tier_threshold_is_inclusive: bool,
}

impl DeploymentPricing {
    /// Whether this override prices by request duration rather than
    /// per-token. Callers on the billing path detect this via
    /// [`calculate_cost_with_override`] returning `None` and bill through
    /// [`Self::cost_per_second`] instead.
    pub fn is_per_second_priced(&self) -> bool {
        let duration_rate = self.cost_per_second_rate.is_some()
            || self.input_cost_per_second.is_some()
            || self.output_cost_per_second.is_some();
        duration_rate
            && self.tiers.is_empty()
            && self.input_cost_per_token == 0.0
            && self.output_cost_per_token == 0.0
            && self.cache_read_input_token_cost.is_none()
            && self.cache_creation_input_token_cost.is_none()
            && self.cache_creation_input_token_cost_above_1hr.is_none()
            && self.output_cost_per_reasoning_token.is_none()
    }

    /// Bill elapsed time once, using the canonical rate, then input, then
    /// output as fallbacks. Invalid rates or durations produce zero spend.
    pub fn cost_per_second(&self, response_time_ms: f64) -> Option<f64> {
        if !self.is_per_second_priced() {
            return None;
        }
        let rate = self
            .cost_per_second_rate
            .or(self.input_cost_per_second)
            .or(self.output_cost_per_second)?;
        Some(duration_cost(rate, response_time_ms))
    }
}

/// On-disk row shape (no provider — that comes from the filename).
#[derive(Debug, Clone, Deserialize)]
struct ModelPricingRow {
    input_cost_per_token: f64,
    output_cost_per_token: f64,
    #[serde(default)]
    input_cost_per_audio_token: Option<f64>,
    #[serde(default)]
    cache_read_input_token_cost: Option<f64>,
    #[serde(default)]
    cache_creation_input_token_cost: Option<f64>,
    /// Cost per cache-creation input token with 1h ephemeral TTL. Anthropic
    /// prices these separately from the default 5m-ephemeral bucket via
    /// `usage.cache_creation.ephemeral_1h_input_tokens`. A plain key, not a
    /// context-length tier.
    #[serde(default)]
    cache_creation_input_token_cost_above_1hr: Option<f64>,
    #[serde(default)]
    output_cost_per_reasoning_token: Option<f64>,
    /// Cost per citation token (Perplexity Sonar/online). Absent on
    /// non-citing models — those bill citation tokens at 0.0.
    #[serde(default)]
    citation_cost_per_token: Option<f64>,
    /// Flat cost per web-search query (Perplexity Sonar/online). Absent on
    /// non-search models — those bill search requests at 0.0.
    #[serde(default)]
    search_queries_cost_per_query: Option<f64>,
    #[serde(default)]
    max_input_tokens: Option<u32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    input_cost_per_second: Option<f64>,
    #[serde(default)]
    output_cost_per_second: Option<f64>,
    #[serde(default)]
    cost_per_second: Option<f64>,
    #[serde(default)]
    mode: Option<String>,
    /// Minimum prompt-prefix tokens the provider requires before it will
    /// engage prompt caching for this model (e.g. Anthropic requires 1024
    /// for most models, 4096 for Opus 4.5). Drives the prompt-caching
    /// deployment check (U34).
    #[serde(default)]
    prompt_cache_min_tokens: Option<u32>,
    #[serde(default)]
    supports_prompt_cache_breakpoint: Option<bool>,
}

/// One context-length pricing tier for a model, using the
/// `input_cost_per_token_above_<N>k_tokens` convention: when a request's
/// prompt exceeds `threshold_tokens`, the tier's rates apply for ALL token
/// types (input, output, and cache). Rates that are `None` in the tier fall
/// back to the model's base rates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CostTier {
    /// Prompt-token count above which this tier's rates apply (e.g. 200000).
    pub threshold_tokens: u32,
    /// Input-token rate that applies above the threshold. `None` falls back
    /// to the base rate.
    pub input_cost_per_token: Option<f64>,
    /// Output-token rate that applies above the threshold. `None` falls back
    /// to the base rate.
    pub output_cost_per_token: Option<f64>,
    /// Cache-read rate that applies above the threshold. `None` falls back
    /// to the base rate.
    pub cache_read_input_token_cost: Option<f64>,
    /// Cache-creation rate that applies above the threshold. `None` falls
    /// back to the base rate.
    pub cache_creation_input_token_cost: Option<f64>,
    /// 1h-ephemeral cache-creation rate that applies above the threshold
    /// (`cache_creation_input_token_cost_above_1hr_above_<N>k_tokens`).
    /// `None` falls back to the base 1h rate.
    pub cache_creation_input_token_cost_above_1hr: Option<f64>,
}

/// Per-model pricing information.
#[derive(Debug, Clone, Default)]
pub struct ModelPricing {
    /// Cost per input (prompt) token.
    pub input_cost_per_token: f64,
    /// Cost per output (completion) token.
    pub output_cost_per_token: f64,
    /// Cost per audio token, if the model is multimodal.
    pub input_cost_per_audio_token: Option<f64>,
    /// Cost per cached-read input token (prompt cache hit).
    pub cache_read_input_token_cost: Option<f64>,
    /// Cost per cache-creation input token (first write to the cache).
    pub cache_creation_input_token_cost: Option<f64>,
    /// Cost per cache-creation input token with 1h ephemeral TTL (Anthropic
    /// `usage.cache_creation.ephemeral_1h_input_tokens`). `None` means the
    /// 1h bucket bills at 0.0 (free), matching the catalog's
    /// `_get_cost_per_unit` default; only models that explicitly declare
    /// this rate bill 1h tokens.
    pub cache_creation_input_token_cost_above_1hr: Option<f64>,
    /// Cost per reasoning (thinking) output token. Falls back to
    /// `output_cost_per_token` when `None`. Reasoning tokens are a subset
    /// of `completion_tokens`.
    pub output_cost_per_reasoning_token: Option<f64>,
    /// Cost per citation token (Perplexity Sonar/online). `None` means
    /// citation tokens bill at 0.0 (the model does not cite).
    pub citation_cost_per_token: Option<f64>,
    /// Flat cost per web-search query (Perplexity Sonar/online). `None` means
    /// search requests bill at 0.0 (the model does not search).
    pub search_queries_cost_per_query: Option<f64>,
    /// Maximum input context window.
    pub max_input_tokens: Option<u32>,
    /// Maximum output tokens per request.
    pub max_output_tokens: Option<u32>,
    /// Provider slug (`"openai"`, `"deepseek"`, …), from the price-file name.
    pub provider: String,
    /// Canonical cost per second of elapsed request time.
    pub cost_per_second_rate: Option<f64>,
    /// Legacy input duration rate.
    pub input_cost_per_second: Option<f64>,
    /// Legacy output duration rate, used when no canonical/input rate exists.
    pub output_cost_per_second: Option<f64>,
    /// Endpoint mode. Audio duration is not elapsed request time.
    pub mode: Option<String>,
    /// Minimum prompt-prefix tokens the provider requires before it will
    /// engage prompt caching for this model. `None` means "not modelled";
    /// the caller falls back to a deployment/global/default threshold.
    pub prompt_cache_min_tokens: Option<u32>,
    /// Whether this exact hosted model accepts the OpenAI explicit cache dialect.
    pub supports_prompt_cache_breakpoint: Option<bool>,
    /// Context-length pricing tiers, sorted by threshold descending
    /// (highest threshold first). Empty for flat-priced models.
    pub tiers: Vec<CostTier>,
    /// Whether tier thresholds are INCLUSIVE (`>=`): xAI bills the tier rate
    /// once the context REACHES the threshold, while every other provider
    /// prices tiers as "more than N tokens" (exclusive `>`). Mirrors
    /// the reference `_INCLUSIVE_THRESHOLD_PROVIDERS = {"xai"}` /
    /// `_uses_inclusive_token_thresholds(custom_llm_provider)`.
    pub tier_threshold_is_inclusive: bool,
    /// Resolved service-specific prices. Missing fields inherit the standard
    /// entry, including its context-length tiers. Child entries have no
    /// further service-specific prices. Published batch rates use the
    /// `"batches"` entry and their own per-bucket context thresholds.
    pub service_tiers: HashMap<String, ModelPricing>,
}

impl ModelPricing {
    fn from_row(provider: &str, row: ModelPricingRow, tiers: Vec<CostTier>) -> Self {
        Self {
            input_cost_per_token: row.input_cost_per_token,
            output_cost_per_token: row.output_cost_per_token,
            input_cost_per_audio_token: row.input_cost_per_audio_token,
            cache_read_input_token_cost: row.cache_read_input_token_cost,
            cache_creation_input_token_cost: row.cache_creation_input_token_cost,
            cache_creation_input_token_cost_above_1hr: row
                .cache_creation_input_token_cost_above_1hr,
            output_cost_per_reasoning_token: row.output_cost_per_reasoning_token,
            citation_cost_per_token: row.citation_cost_per_token,
            search_queries_cost_per_query: row.search_queries_cost_per_query,
            max_input_tokens: row.max_input_tokens,
            max_output_tokens: row.max_output_tokens,
            provider: provider.to_string(),
            input_cost_per_second: row.input_cost_per_second,
            output_cost_per_second: row.output_cost_per_second,
            cost_per_second_rate: row.cost_per_second,
            mode: row.mode,
            prompt_cache_min_tokens: row.prompt_cache_min_tokens,
            supports_prompt_cache_breakpoint: row.supports_prompt_cache_breakpoint,
            tiers,
            tier_threshold_is_inclusive: provider == "xai",
            service_tiers: HashMap::new(),
        }
    }

    /// Resolve the effective per-token rates for a request, applying
    /// context-length tier selection.
    ///
    /// Tier selection follows `_get_token_base_cost` semantics: when the
    /// prompt exceeds a tier's threshold, that tier's rates replace the base
    /// rates for all token types, and the highest threshold exceeded wins
    /// (tiers are pre-sorted descending). Rates absent from the tier fall
    /// back to the base rates.
    ///
    /// The boundary is EXCLUSIVE (`>`) for every provider except xAI, whose
    /// rows set [`ModelPricing::tier_threshold_is_inclusive`] so the tier
    /// rate applies at `>=` the threshold — mirroring the reference
    /// `_uses_inclusive_token_thresholds` gating on
    /// `_INCLUSIVE_THRESHOLD_PROVIDERS = {"xai"}`. Providers price the
    /// boundary as "more than 200K tokens" (Gemini/OpenAI docs), so a
    /// request landing exactly on 200,000 / 272,000 context tokens bills at
    /// base rates.
    ///
    /// The threshold is evaluated against the *full context length*:
    /// `prompt_tokens + cache_creation_tokens`. oxllm's `UsageSummary`
    /// normalises cache-creation to be ADDITIVE (the extraction layer
    /// subtracts the OpenAI subset form out of `prompt_tokens` and leaves
    /// Anthropic's additive form outside it), so this sum is the billable
    /// context size — the same value the reference compares, whose Anthropic
    /// transformation folds BOTH `cache_creation_input_tokens` and
    /// `cache_read_input_tokens` onto `prompt_tokens` before cost lookup.
    /// Cache-read tokens are already inside `prompt_tokens` (subset
    /// invariant), so they are counted exactly once. Pass the caller's
    /// `cache_creation_input_tokens` (0 when unknown, e.g. the budget
    /// estimator sizing a request before the response arrives).
    ///
    /// Rate-fallback semantic (the *standard price-table path*): a model entry
    /// lacking `cache_read_input_token_cost`, `cache_creation_input_token_cost`,
    /// `cache_creation_input_token_cost_above_1hr`, or
    /// `input_cost_per_audio_token` bills those buckets at **0.0**. Fireworks
    /// is the exception: a missing cache-read rate bills at half the base
    /// input rate. This deliberately differs from the [`DeploymentPricing`]
    /// override path (which falls back to the input rate, matching the reference
    /// `_cost_per_token_custom_pricing_helper`). Reasoning tokens fall back
    /// to the output rate on both paths.
    ///
    /// This method uses standard pricing. Use
    /// [`Self::rates_for_service_tier`] for a request served on another tier.
    pub fn rates_for_prompt(&self, prompt_tokens: u64, cache_creation_tokens: u64) -> TokenRates {
        let context_tokens = prompt_tokens.saturating_add(cache_creation_tokens);
        let mut input = self.input_cost_per_token;
        let mut output = self.output_cost_per_token;
        let mut cache_read = self.cache_read_input_token_cost;
        let mut cache_creation = self.cache_creation_input_token_cost;
        let mut cache_creation_1h = self.cache_creation_input_token_cost_above_1hr;
        for tier in &self.tiers {
            let threshold = tier.threshold_tokens as u64;
            let exceeded = context_tokens > threshold
                || (self.tier_threshold_is_inclusive && context_tokens == threshold);
            if exceeded {
                input = tier.input_cost_per_token.unwrap_or(input);
                output = tier.output_cost_per_token.unwrap_or(output);
                cache_read = tier.cache_read_input_token_cost.or(cache_read);
                cache_creation = tier.cache_creation_input_token_cost.or(cache_creation);
                cache_creation_1h = tier
                    .cache_creation_input_token_cost_above_1hr
                    .or(cache_creation_1h);
                break;
            }
        }
        // Fireworks publishes a 50% cache-read discount when a model row has
        // no cache-read rate. The discount is taken off the base input rate
        // (a context tier overrides it only when that tier publishes its own
        // cache-read rate). Every other provider still bills a missing
        // cache-read rate at zero.
        let cache_read_rate = cache_read.unwrap_or(if self.provider == "fireworks_ai" {
            self.input_cost_per_token * 0.5
        } else {
            0.0
        });
        TokenRates {
            input,
            output,
            cache_read: cache_read_rate,
            cache_creation: cache_creation.unwrap_or(0.0),
            cache_creation_1h: cache_creation_1h.unwrap_or(0.0),
            audio: self.input_cost_per_audio_token.unwrap_or(0.0),
            reasoning: self.output_cost_per_reasoning_token.unwrap_or(output),
            citation: self.citation_cost_per_token.unwrap_or(0.0),
            search_query: self.search_queries_cost_per_query.unwrap_or(0.0),
        }
    }

    /// Resolve service-specific rates, falling back to standard pricing when
    /// the model has no entry for the requested tier.
    pub fn rates_for_service_tier(
        &self,
        prompt_tokens: u64,
        cache_creation_tokens: u64,
        service_tier: Option<&str>,
    ) -> TokenRates {
        service_tiers::pricing_for_tier(self, service_tier)
            .rates_for_prompt(prompt_tokens, cache_creation_tokens)
    }

    /// Compute cost from elapsed request time for a duration-priced model.
    /// Token pricing takes precedence, and audio/media duration rates never
    /// use wall-clock request latency.
    pub fn cost_per_second(&self, response_time_ms: f64) -> Option<f64> {
        if !self.is_per_second_priced() {
            return None;
        }
        let rate = self
            .cost_per_second_rate
            .or(self.input_cost_per_second)
            .or(self.output_cost_per_second)?;
        Some(duration_cost(rate, response_time_ms))
    }

    /// Whether this model is priced per-second (duration-based) rather than
    /// per-token. Use to detect the otherwise-silent [`calculate_cost`] skip
    /// path and bill via [`Self::cost_per_second`] instead.
    pub fn is_per_second_priced(&self) -> bool {
        let duration_rate = self.cost_per_second_rate.is_some()
            || self.input_cost_per_second.is_some()
            || self.output_cost_per_second.is_some();
        duration_rate
            && self.input_cost_per_token == 0.0
            && self.output_cost_per_token == 0.0
            && self.tiers.is_empty()
            && self.mode.as_deref().is_none_or(|mode| {
                matches!(
                    mode,
                    // Realtime voice sessions bill per wall-clock second of
                    // the live session (the reference's live-session
                    // cost_calculation); audio_transcription stays excluded
                    // because its seconds are media duration, not elapsed
                    // request time.
                    "chat" | "responses" | "completion" | "text_completion" | "realtime"
                )
            })
    }
}

/// Fully-resolved per-token rates for a request, after context-length tier
/// selection and rate-fallback resolution. Produced by
/// [`ModelPricing::rates_for_prompt`] for the price-table path and built
/// directly from [`DeploymentPricing`] for the override path.
#[derive(Debug, Clone, Copy)]
pub struct TokenRates {
    /// Cost per regular input (prompt) token.
    pub input: f64,
    /// Cost per regular output (completion) token.
    pub output: f64,
    /// Cost per cached-read input token (0.0 when the model declares none).
    pub cache_read: f64,
    /// Cost per cache-creation input token (5m ephemeral bucket).
    pub cache_creation: f64,
    /// Cost per cache-creation input token in the 1h ephemeral bucket.
    pub cache_creation_1h: f64,
    /// Cost per audio input token.
    pub audio: f64,
    /// Cost per reasoning (thinking) output token.
    pub reasoning: f64,
    /// Cost per citation token (0.0 when the model declares none).
    pub citation: f64,
    /// Flat cost per web-search query (0.0 when the model declares none).
    pub search_query: f64,
}

impl TokenRates {
    /// Resolve the rates for a deployment pricing override. Mirrors
    /// the custom-pricing helper: cache rates fall
    /// back to the input rate, and reasoning to the output rate (the 1h
    /// cache-creation rate first falls back to the 5m rate). Overrides
    /// carry no per-bucket audio rate, so audio bills at the input rate.
    fn from_override(dp: &DeploymentPricing) -> Self {
        let input = dp.input_cost_per_token;
        let output = dp.output_cost_per_token;
        let cache_creation = dp.cache_creation_input_token_cost.unwrap_or(input);
        Self {
            input,
            output,
            cache_read: dp.cache_read_input_token_cost.unwrap_or(input),
            cache_creation,
            cache_creation_1h: dp
                .cache_creation_input_token_cost_above_1hr
                .unwrap_or(cache_creation),
            audio: input,
            reasoning: dp.output_cost_per_reasoning_token.unwrap_or(output),
            citation: 0.0,
            search_query: 0.0,
        }
    }

    /// Like [`Self::from_override`], after applying the first deployment
    /// tier whose threshold the prompt has passed. Thresholds are exclusive
    /// unless `tier_threshold_is_inclusive` is set. A `None` tier rate keeps
    /// the override base, then the same cache-to-input fallback applies.
    fn from_override_for_usage(dp: &DeploymentPricing, usage: &UsageSummary) -> Self {
        if dp.tiers.is_empty() {
            return Self::from_override(dp);
        }
        let context = usage.prompt_tokens.saturating_add(
            usage
                .cache_creation_input_tokens
                .unwrap_or_else(|| usage.cache_creation_1h_input_tokens.unwrap_or(0)),
        );
        let mut priced = dp.clone();
        for tier in &dp.tiers {
            let threshold = tier.threshold_tokens as u64;
            let exceeded =
                context > threshold || (dp.tier_threshold_is_inclusive && context == threshold);
            if !exceeded {
                continue;
            }
            if let Some(rate) = tier.input_cost_per_token {
                priced.input_cost_per_token = rate;
            }
            if let Some(rate) = tier.output_cost_per_token {
                priced.output_cost_per_token = rate;
            }
            if tier.cache_read_input_token_cost.is_some() {
                priced.cache_read_input_token_cost = tier.cache_read_input_token_cost;
            }
            if tier.cache_creation_input_token_cost.is_some() {
                priced.cache_creation_input_token_cost = tier.cache_creation_input_token_cost;
            }
            if tier.cache_creation_input_token_cost_above_1hr.is_some() {
                priced.cache_creation_input_token_cost_above_1hr =
                    tier.cache_creation_input_token_cost_above_1hr;
            }
            break;
        }
        Self::from_override(&priced)
    }
}

/// Parse one provider's TOML shard into a map. Fault-tolerant per entry.
///
/// The shards are compile-time-vendored (`include_str!`) and therefore
/// trusted: a TOML syntax error is always a programmer error caught in CI,
/// never runtime input. Such a failure is promoted to a `panic!` so a broken
/// shard can never silently zero out an entire provider's pricing. A single
/// malformed model ROW, by contrast, only drops that row (handled below).
fn parse_provider_prices(provider: &str, raw: &str) -> HashMap<String, ModelPricing> {
    let table: toml::Table = match raw.parse() {
        Ok(t) => t,
        Err(e) => {
            panic!(
                "embedded price shard '{provider}' failed to parse (this is vendored, \
                 trusted data — a syntax error is a programmer error, not runtime input): {e}"
            );
        }
    };
    let mut out = HashMap::with_capacity(table.len());
    for (key, value) in table {
        let tiers = extract_cost_tiers(&value);
        let service_tiers = service_tiers::parse_pricing(
            provider,
            &value,
            service_tiers::price_overlays().get(&key),
        );
        match value.try_into::<ModelPricingRow>() {
            Ok(row) => {
                let mut pricing = ModelPricing::from_row(provider, row, tiers);
                pricing.service_tiers = service_tiers;
                out.insert(key, pricing);
            }
            Err(e) => {
                tracing::warn!(
                    provider = %provider,
                    model = %key,
                    error = %e,
                    "skipping pricing entry that does not match the current schema"
                );
            }
        }
    }
    out
}

/// Base rate fields that can carry a context-length tier, mirroring
/// the `input_cost_per_token_above_<N>k_tokens` key convention. The
/// `cache_creation_input_token_cost_above_1hr` entry covers the catalog's
/// combined `cache_creation_input_token_cost_above_1hr_above_<N>k_tokens`
/// shape (see [`parse_threshold_key`]).
const TIER_BASE_FIELDS: &[&str] = &[
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
];

/// Parse a tier rate that may be a TOML float (`1.5e-5`), an integer
/// (`2000000`), or a numeric string (`"1e-5"`). Serde coerces integers to
/// `f64` for the base row fields, so the tier path must accept both too —
/// otherwise a rate written without a decimal point silently drops the entire
/// tier. Accepting numeric strings mirrors the reference `_get_cost_per_unit`,
/// which coerces strings via `float(...)` (its config loader sometimes emits
/// rates as strings). A non-numeric string still returns `None` so that such
/// a key is skipped instead of poisoning the sibling tier rates.
fn tier_rate_value(v: &toml::Value) -> Option<f64> {
    match v {
        toml::Value::Float(f) => Some(*f),
        toml::Value::Integer(i) => Some(*i as f64),
        toml::Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Whether a raw price key that failed [`parse_threshold_key`] nevertheless
/// LOOKS like a context-tier key — the `_above_` marker plus a `tokens`
/// fragment anywhere (covers `_above_<N>k_tokens`, `_above_<N>_tokens`,
/// typo'd spellings like `_above_200ktokens`, and service-tier-suffixed
/// variants like `..._above_200k_tokens_priority`). Base keys such as
/// `cache_creation_input_token_cost_above_1hr` contain no `tokens` fragment
/// and stay silent.
fn looks_like_tier_key(key: &str) -> bool {
    key.contains("_above_") && key.contains("tokens")
}

/// Parse `_above_<N>k_tokens` (or `_above_<N>_tokens`) suffix keys out of a
/// model's raw TOML table into [`CostTier`]s. Tiers are returned sorted by
/// threshold descending (highest threshold first) so the selection loop can
/// take the first match, exactly like the reference `_get_token_base_cost`.
///
/// Deliberate divergence from the reference: it decides whether a model has
/// tiered pricing by scanning *only* `input_cost_per_token_above_*` keys
/// (its `threshold_keys` early-return in `_get_token_base_cost`); a model
/// declaring only `output_cost_per_token_above_*` never tiers there. oxllm
/// merges per-threshold rates across input, output, and cache fields, so a
/// tier carrying just an output rate still activates when the prompt exceeds
/// its threshold. This is strictly more general (it can only re-rate the
/// buckets the operator explicitly declared) and is pinned by the
/// `tiered_cost_highest_threshold_wins_with_fallback_to_base` test.
fn extract_cost_tiers(value: &toml::Value) -> Vec<CostTier> {
    let Some(table) = value.as_table() else {
        return Vec::new();
    };
    let mut tiers: Vec<CostTier> = Vec::new();
    for (key, v) in table {
        let Some((base, threshold)) = parse_threshold_key(key) else {
            if looks_like_tier_key(key) && !service_tiers::is_service_price_key(key) {
                tracing::warn!(
                    key = %key,
                    "dropping tier-shaped price key that does not parse — a hand-edited \
                     typo (e.g. '_above_200ktokens') bills long-context requests at base \
                     rates"
                );
            }
            continue;
        };
        if !TIER_BASE_FIELDS.contains(&base) {
            continue;
        }
        let Some(rate) = tier_rate_value(v) else {
            tracing::warn!(
                key = %key,
                "skipping context-tier key with a non-numeric rate"
            );
            continue;
        };
        if let Some(tier) = tiers.iter_mut().find(|t| t.threshold_tokens == threshold) {
            if set_tier_rate(tier, base, rate) {
                tracing::warn!(
                    key = %key,
                    threshold_tokens = threshold,
                    base_field = base,
                    "duplicate context-tier key spelling overwrites an already-set rate for this threshold"
                );
            }
        } else {
            let mut tier = CostTier {
                threshold_tokens: threshold,
                input_cost_per_token: None,
                output_cost_per_token: None,
                cache_read_input_token_cost: None,
                cache_creation_input_token_cost: None,
                cache_creation_input_token_cost_above_1hr: None,
            };
            set_tier_rate(&mut tier, base, rate);
            tiers.push(tier);
        }
    }
    tiers.sort_by_key(|t| std::cmp::Reverse(t.threshold_tokens));
    tiers
}

/// Set one tierable rate field by base-field name. Returns `true` when the
/// field was already `Some` — i.e. this call silently overwrote a prior value,
/// which can only happen when two key spellings parse to the same
/// `(base, threshold)` (e.g. `_above_100k_tokens` + `_above_100000_tokens`).
/// The caller logs that case so the overwrite cannot pass unnoticed.
fn set_tier_rate(tier: &mut CostTier, base: &str, rate: f64) -> bool {
    let slot: &mut Option<f64> = match base {
        "input_cost_per_token" => &mut tier.input_cost_per_token,
        "output_cost_per_token" => &mut tier.output_cost_per_token,
        "cache_read_input_token_cost" => &mut tier.cache_read_input_token_cost,
        "cache_creation_input_token_cost" => &mut tier.cache_creation_input_token_cost,
        "cache_creation_input_token_cost_above_1hr" => {
            &mut tier.cache_creation_input_token_cost_above_1hr
        }
        _ => return false,
    };
    let overwritten = slot.is_some();
    *slot = Some(rate);
    overwritten
}

/// Split a `_above_<N>k_tokens` / `_above_<N>_tokens` key into its base
/// field name and the parsed token threshold. Returns `None` for keys that
/// are not threshold keys (e.g. base rates or `_above_1hr` cache TTL keys).
///
/// The Anthropic 1h cache-creation base itself ends in `_above_1hr`
/// (`cache_creation_input_token_cost_above_1hr`), so its tiered form is a
/// *combined* key: `cache_creation_input_token_cost_above_1hr_above_<N>k_tokens`.
/// That shape is recognised and normalised to the `..._above_1hr` base.
fn parse_threshold_key(key: &str) -> Option<(&str, u32)> {
    let idx = key.find("_above_")?;
    let base = &key[..idx];
    let rest = &key[idx + "_above_".len()..];
    let (base, rest) = match rest.strip_prefix("1hr_above_") {
        Some(inner) if base == "cache_creation_input_token_cost" => {
            ("cache_creation_input_token_cost_above_1hr", inner)
        }
        _ => (base, rest),
    };
    let rest = rest.strip_suffix("_tokens")?;
    let (num, scale) = if let Some(n) = rest.strip_suffix('k') {
        (n, 1000u32)
    } else {
        (rest, 1u32)
    };
    let n: u32 = num.parse().ok()?;
    Some((base, n.saturating_mul(scale)))
}

/// How an `oxllm_params.extra` key relates to context-tier pricing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtraTierKeyClass {
    /// Not a context-tier key.
    NotTier,
    /// Parses as a known tier rate field.
    Parsed,
    /// Looks like a tier key but does not parse, or parses to an unknown field.
    Malformed,
}

/// Classify one flattened deployment key. Malformed keys are warned at load
/// and are not billed.
pub fn classify_extra_tier_key(key: &str) -> ExtraTierKeyClass {
    match parse_threshold_key(key) {
        Some((base, threshold))
            if TIER_BASE_FIELDS.contains(&base)
                && !(threshold == u32::MAX && key.ends_with("k_tokens")) =>
        {
            ExtraTierKeyClass::Parsed
        }
        Some(_) if looks_like_tier_key(key) => ExtraTierKeyClass::Malformed,
        None if looks_like_tier_key(key) => ExtraTierKeyClass::Malformed,
        _ => ExtraTierKeyClass::NotTier,
    }
}

fn json_tier_rate(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
        .filter(|rate| rate.is_finite())
}

/// Build deployment context tiers from flattened `*_above_*_tokens` keys.
/// Highest threshold first. Non-numeric rates are skipped; load-time
/// validation warns about those separately.
pub fn cost_tiers_from_extra(extra: &serde_json::Map<String, serde_json::Value>) -> Vec<CostTier> {
    let mut tiers: Vec<CostTier> = Vec::new();
    for (key, value) in extra {
        if classify_extra_tier_key(key) != ExtraTierKeyClass::Parsed {
            continue;
        }
        let Some((base, threshold)) = parse_threshold_key(key) else {
            continue;
        };
        if !TIER_BASE_FIELDS.contains(&base) {
            continue;
        }
        let Some(rate) = json_tier_rate(value) else {
            continue;
        };
        if let Some(tier) = tiers.iter_mut().find(|t| t.threshold_tokens == threshold) {
            set_tier_rate(tier, base, rate);
        } else {
            let mut tier = CostTier {
                threshold_tokens: threshold,
                input_cost_per_token: None,
                output_cost_per_token: None,
                cache_read_input_token_cost: None,
                cache_creation_input_token_cost: None,
                cache_creation_input_token_cost_above_1hr: None,
            };
            set_tier_rate(&mut tier, base, rate);
            tiers.push(tier);
        }
    }
    tiers.sort_by_key(|t| std::cmp::Reverse(t.threshold_tokens));
    tiers
}

/// True when a parsed tier key's JSON value is not a finite number.
pub fn extra_tier_rate_is_non_numeric(key: &str, value: &serde_json::Value) -> bool {
    classify_extra_tier_key(key) == ExtraTierKeyClass::Parsed && json_tier_rate(value).is_none()
}

/// Load and merge every embedded provider shard.
fn load_all_prices() -> HashMap<String, ModelPricing> {
    let mut out = HashMap::new();
    for &(provider, raw) in PRICE_SHARDS {
        let shard = parse_provider_prices(provider, raw);
        for (model, pricing) in shard {
            if out.insert(model.clone(), pricing).is_some() {
                tracing::warn!(
                    provider = %provider,
                    model = %model,
                    "duplicate pricing key across provider shards; later shard wins"
                );
            }
        }
    }
    out
}

static PRICES: OnceLock<HashMap<String, ModelPricing>> = OnceLock::new();

/// Returns the loaded price table (parsed lazily on first access).
pub fn price_table() -> &'static HashMap<String, ModelPricing> {
    PRICES.get_or_init(load_all_prices)
}

/// Strip a leading provider prefix (e.g. `"openai/"`) from a model id, since
/// the embedded price table is keyed by bare model names.
fn strip_provider_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some((_, rest)) if !rest.is_empty() => rest,
        _ => model,
    }
}

/// Strip a trailing dated snapshot suffix used by many providers.
///
/// Two formats are recognised:
/// - `-YYYY-MM-DD` (11 chars, ISO 8601 with inner dashes), e.g.
///   `gpt-4o-2024-08-06` → `gpt-4o`.
/// - `-YYYYMMDD` (9 chars, no inner dashes), e.g. Anthropic's compact
///   snapshot ids like `claude-3-5-sonnet-20241022` → `claude-3-5-sonnet`.
fn strip_date_version_suffix(model: &str) -> Option<&str> {
    let is_digit = |b: u8| b.is_ascii_digit();
    let bytes = model.as_bytes();

    // Pattern 1: ends with "-dddd-dd-dd" (11 chars including leading dash).
    if bytes.len() >= 12 {
        let s = &bytes[bytes.len() - 11..];
        let ok = s[0] == b'-'
            && is_digit(s[1])
            && is_digit(s[2])
            && is_digit(s[3])
            && is_digit(s[4])
            && s[5] == b'-'
            && is_digit(s[6])
            && is_digit(s[7])
            && s[8] == b'-'
            && is_digit(s[9])
            && is_digit(s[10]);
        if ok {
            return Some(&model[..model.len() - 11]);
        }
    }

    // Pattern 2: ends with "-dddddddd" (9 chars, no inner dashes — the
    // Anthropic compact snapshot format).
    if bytes.len() >= 10 {
        let s = &bytes[bytes.len() - 9..];
        let ok = s[0] == b'-'
            && is_digit(s[1])
            && is_digit(s[2])
            && is_digit(s[3])
            && is_digit(s[4])
            && is_digit(s[5])
            && is_digit(s[6])
            && is_digit(s[7])
            && is_digit(s[8]);
        if ok {
            return Some(&model[..model.len() - 9]);
        }
    }

    None
}

/// Look up pricing for `model`, tolerant of a leading provider prefix and
/// dated snapshot suffixes (`gpt-4o-2024-08-06` → `gpt-4o`).
///
/// Multi-segment provider prefixes (e.g. `openrouter/openai/gpt-4o`,
/// `together_ai/meta-llama/Llama-3.3-70B`) are unwound iteratively: after
/// the first strip fails to find a match, additional `/`-prefixed segments
/// are stripped one at a time (up to 3 levels deep).
///
/// ```rust
/// use oxllm_core::cost::lookup_pricing;
///
/// // Bare model id.
/// let bare = lookup_pricing("gpt-4o").expect("gpt-4o is priced");
/// // Provider-prefixed id resolves to the same record.
/// let prefixed = lookup_pricing("openai/gpt-4o").expect("prefixed lookup");
/// assert_eq!(bare.input_cost_per_token, prefixed.input_cost_per_token);
/// // Unknown models resolve to None.
/// assert!(lookup_pricing("not-a-real-model").is_none());
/// ```
pub fn lookup_pricing(model: &str) -> Option<&'static ModelPricing> {
    lookup_with_optional_provider(price_table(), model, None)
}

/// Try a single candidate name against the table, preferring the serving
/// provider's canonical `{provider}/{name}` key over a bare alias from a
/// different provider (which could otherwise mis-attribute spend). When no
/// provider hint is given, only the bare form is consulted — reproducing the
/// legacy provider-agnostic lookup exactly.
fn try_candidate<'a>(
    table: &'a HashMap<String, ModelPricing>,
    provider: Option<&str>,
    name: &str,
) -> Option<&'a ModelPricing> {
    if let Some(prov) = provider {
        // Configured pricing models may already be provider-scoped. Do not
        // turn `perplexity/sonar` into the distinct namespace card
        // `perplexity/perplexity/sonar` when previewing that configuration.
        if name
            .strip_prefix(prov)
            .is_some_and(|rest| rest.starts_with('/'))
        {
            if let Some(pricing) = table.get(name) {
                return Some(pricing);
            }
        }
        if let Some(p) = get_prefixed(table, prov, name) {
            return Some(p);
        }
    }
    table.get(name)
}

/// Core resolution shared by [`lookup_pricing`] and
/// [`lookup_pricing_with_provider`]. Each candidate name — the exact input,
/// the iterative provider-prefix-stripped forms, and the date-snapshot-stripped
/// form — is tried both bare and (when a provider hint is given) as
/// `{provider}/{name}`. Trying the prefixed form of the date-stripped
/// candidate is what lets a dated snapshot served by a prefixed-key provider
/// (e.g. `gemini-2.5-pro-2025-03-25` served by `gemini`, whose row is keyed
/// `gemini/gemini-2.5-pro`) resolve instead of falling through to unmetered
/// `None` spend.
fn lookup_with_optional_provider<'a>(
    table: &'a HashMap<String, ModelPricing>,
    model: &str,
    provider: Option<&str>,
) -> Option<&'a ModelPricing> {
    if let Some(p) = try_candidate(table, provider, model) {
        return Some(p);
    }
    // Iteratively strip provider prefixes: handles single-segment prefixes
    // (`openai/gpt-4o`) as well as multi-segment ones
    // (`openrouter/openai/gpt-4o`, `together_ai/meta-llama/Llama-3.3-70B`).
    //
    // The level check happens AFTER the table lookup so the value produced by
    // the final strip is still consulted before bailing out (the historical
    // bug: a 4-segment prefix like `a/b/c/d/gpt-4o` stripped `d/` to reach
    // `gpt-4o` but broke before ever looking it up).
    let mut stripped = strip_provider_prefix(model);
    let mut levels = 0u8;
    loop {
        if let Some(p) = try_candidate(table, provider, stripped) {
            return Some(p);
        }
        if levels >= 3 {
            break;
        }
        let next = strip_provider_prefix(stripped);
        if next == stripped {
            break;
        }
        stripped = next;
        levels += 1;
    }
    if let Some(base) = strip_date_version_suffix(stripped) {
        if let Some(p) = try_candidate(table, provider, base) {
            return Some(p);
        }
    }
    None
}

/// Look up pricing for `model`, preferring the canonical provider-prefixed
/// table key when the caller knows which provider served the request.
///
/// Some providers (notably Gemini) key their price rows with a `provider/model`
/// prefix (e.g. `gemini/gemini-2.5-flash`) while the upstream OpenAI-compat
/// response echoes the bare model back (`gemini-2.5-flash`). A bare
/// [`lookup_pricing`] call would then miss — or worse, silently resolve to a
/// *different* provider's bare alias (e.g. a Vertex AI entry with different
/// rates). This helper tries `{provider}/{model}` first so the serving
/// provider's canonical row wins, then falls back to [`lookup_pricing`].
///
/// ```rust
/// use oxllm_core::cost::lookup_pricing_with_provider;
///
/// // The bare name resolves to the gemini row, not a vertex_ai alias.
/// let p = lookup_pricing_with_provider("gemini-2.5-flash", "gemini")
///     .expect("gemini/gemini-2.5-flash is priced");
/// assert_eq!(p.provider, "gemini");
/// ```
pub fn lookup_pricing_with_provider(model: &str, provider: &str) -> Option<&'static ModelPricing> {
    let provider = if provider.is_empty() {
        None
    } else {
        Some(provider)
    };
    lookup_with_optional_provider(price_table(), model, provider)
}

/// Look up `{provider}/{model}` in the price table without heap-allocating on
/// the post-response billing path. Real model keys are short (the longest
/// shipped key is under 100 bytes), so a stack buffer covers them; a heap
/// fallback handles pathological inputs. Both `provider` and `model` are valid
/// UTF-8 (`&str`), so joining their bytes with a `/` separator is valid UTF-8.
fn get_prefixed<'a>(
    table: &'a HashMap<String, ModelPricing>,
    provider: &str,
    model: &str,
) -> Option<&'a ModelPricing> {
    let total = provider.len() + 1 + model.len();
    if total <= 256 {
        let mut buf = [0u8; 256];
        buf[..provider.len()].copy_from_slice(provider.as_bytes());
        buf[provider.len()] = b'/';
        buf[provider.len() + 1..total].copy_from_slice(model.as_bytes());
        let key = std::str::from_utf8(&buf[..total])
            .expect("concatenation of two valid &str bytes with '/' is valid UTF-8");
        table.get(key)
    } else {
        table.get(&format!("{provider}/{model}"))
    }
}

/// Compute cost from a resolved [`ModelPricing`] entry. Returns `None` for
/// per-second pricing (logged at `debug` and skipped): the token-based path
/// has no duration input, and the billing callers own the per-second
/// decision — they detect it via [`ModelPricing::is_per_second_priced`] and
/// bill through [`ModelPricing::cost_per_second`], so reaching this skip on
/// the billing path is expected flow, not a fault.
fn cost_from_pricing(pricing: &ModelPricing, usage: &UsageSummary) -> Option<f64> {
    if pricing.is_per_second_priced() {
        tracing::debug!(
            provider = %pricing.provider,
            "per-second pricing is not supported on the token-based cost path; cost omitted — \
             bill via ModelPricing::cost_per_second(response_time_ms) to accrue spend"
        );
        return None;
    }
    if (pricing.cost_per_second_rate.is_some()
        || pricing.input_cost_per_second.is_some()
        || pricing.output_cost_per_second.is_some())
        && pricing.input_cost_per_token == 0.0
        && pricing.output_cost_per_token == 0.0
        && pricing.tiers.is_empty()
    {
        // Media duration cannot be derived from token usage or request latency.
        return None;
    }
    // Tier threshold measure: the same effective cache-creation total
    // `cost_from_rates` bills — the aggregate when present, otherwise the 1h
    // detail treated as the effective total — so a provider reporting only
    // the 1h detail still counts those tokens toward the tier threshold
    // while an inconsistent (aggregate, 1h) pair can never push a request
    // whose billed context stays below the threshold into the tier.
    let cache_creation_effective = match usage.cache_creation_input_tokens {
        Some(aggregate) => aggregate,
        None => usage.cache_creation_1h_input_tokens.unwrap_or(0),
    };
    Some(cost_from_rates(
        usage,
        pricing.rates_for_prompt(usage.prompt_tokens, cache_creation_effective),
    ))
}

/// Compute the cost of a request given its model and token usage.
///
/// Returns `None` when:
/// - the model is not in the price table, or
/// - the model uses per-second pricing (logged at `debug` and skipped).
///
/// ```rust
/// use oxllm_core::cost::calculate_cost;
/// use oxllm_core::UsageSummary;
///
/// let usage = UsageSummary {
///     prompt_tokens: 1_000,
///     completion_tokens: 500,
///     total_tokens: 1_500,
///     cache_read_input_tokens: None,
///     cache_creation_input_tokens: None,
///     cache_creation_1h_input_tokens: None,
///     audio_input_tokens: None,
///     reasoning_tokens: None,
///     citation_tokens: None,
///     web_search_requests: None,
/// };
/// // gpt-4o is in the vendored price table.
/// let cost = calculate_cost("gpt-4o", &usage).expect("gpt-4o is priced");
/// assert!(cost > 0.0);
/// // An unknown model resolves to None.
/// assert!(calculate_cost("not-a-real-model", &usage).is_none());
/// ```
///
/// Token buckets applied (when present in `usage`):
/// - `cache_read_input_tokens` are priced at `cache_read_input_token_cost`
///   and subtracted from the regular input-token bucket (they are a subset
///   of `prompt_tokens` in the OpenAI convention; the extraction layer
///   normalises Anthropic's additive `cache_read_input_tokens` into this
///   subset form).
/// - `cache_creation_input_tokens` are priced at
///   `cache_creation_input_token_cost` (additive; not part of
///   `prompt_tokens`). When `cache_creation_1h_input_tokens` is present, that
///   many cache-creation tokens bill at the model's
///   `cache_creation_input_token_cost_above_1hr` (Anthropic 1h-ephemeral TTL
///   bucket) and the remainder at the 5m rate.
/// - `audio_input_tokens` are priced at `input_cost_per_audio_token`
///   and subtracted from the regular input-token bucket (OpenAI reports
///   them as a subset of `prompt_tokens`).
///
/// On the price-table path, a model entry that declares no cache-read,
/// cache-creation, or audio rate bills those buckets at 0.0 (the reference
/// `_get_cost_per_unit` default), so cached/audio tokens are free for such a
/// model.
pub fn calculate_cost(model: &str, usage: &UsageSummary) -> Option<f64> {
    cost_from_pricing(lookup_pricing(model)?, usage)
}

/// Like [`calculate_cost`] but prefers the serving provider's canonical
/// `provider/model` price-row key before falling back to the bare lookup.
///
/// Use this whenever the caller knows which provider served the request
/// (e.g. from `Deployment::provider_slug()`): it prevents a bare response
/// model name from missing the price table (zero spend accrues) or silently
/// resolving to a *different* provider's bare alias (wrong rate). See
/// [`lookup_pricing_with_provider`] for the resolution rule.
pub fn calculate_cost_with_provider(
    model: &str,
    usage: &UsageSummary,
    provider: &str,
) -> Option<f64> {
    cost_from_pricing(lookup_pricing_with_provider(model, provider)?, usage)
}

/// Calculate cost using either a deployment pricing override or the embedded
/// price table.
///
/// When `deployment_pricing` is `Some`, the cost is computed from the
/// override rates (cache rates fall back to the override's input rate when
/// `None`; audio tokens are billed at the override's input rate since
/// overrides have no audio field). When `None`, falls back to
/// [`calculate_cost`].
pub fn calculate_cost_with_override(
    model: &str,
    usage: &UsageSummary,
    deployment_pricing: Option<&DeploymentPricing>,
) -> Option<f64> {
    calculate_cost_with_override_for_provider(model, usage, deployment_pricing, "")
}

/// Provider-aware variant of [`calculate_cost_with_override`]: when no
/// deployment pricing override is set, the table lookup uses
/// [`lookup_pricing_with_provider`] so a bare response model resolves to the
/// serving provider's canonical row. Pass the deployment's provider slug
/// (e.g. `"gemini"`) from the billing path; an empty string reproduces the
/// legacy provider-agnostic behaviour exactly.
///
/// When an override is set and its `tiers` list is empty, it is applied
/// flat and the matched model's context-length tiers are ignored. Tiers
/// parsed onto the override are selected the same way as table tiers, with
/// an exclusive threshold unless the override says otherwise. See
/// [`DeploymentPricing`].
pub fn calculate_cost_with_override_for_provider(
    model: &str,
    usage: &UsageSummary,
    deployment_pricing: Option<&DeploymentPricing>,
    provider: &str,
) -> Option<f64> {
    calculate_cost_for_service_tier(model, usage, deployment_pricing, provider, None)
}

/// Provider-aware cost calculation with service-specific and context-length
/// pricing. A deployment override takes precedence over table prices. An
/// override with no tier keys stays flat. An absent or unpriced service
/// tier uses standard rates. The
/// `"batches"` selector uses published batch rates, or existing flex rates for
/// Gemini/Vertex when no separate batch entry exists; no extra discount is
/// applied. Use [`resolve_service_tier`] to distinguish served batch metadata
/// from an arbitrary request `service_tier`.
pub fn calculate_cost_for_service_tier(
    model: &str,
    usage: &UsageSummary,
    deployment_pricing: Option<&DeploymentPricing>,
    provider: &str,
    service_tier: Option<&str>,
) -> Option<f64> {
    if let Some(dp) = deployment_pricing {
        if dp.is_per_second_priced() {
            tracing::debug!(
                model = %model,
                "per-second deployment pricing override is not supported on the token-based \
                 cost path; cost omitted — bill via \
                 DeploymentPricing::cost_per_second(response_time_ms) to accrue spend"
            );
            return None;
        }
        Some(cost_from_rates(
            usage,
            TokenRates::from_override_for_usage(dp, usage),
        ))
    } else {
        let pricing = lookup_pricing_with_provider(model, provider)?;
        cost_from_pricing(
            service_tiers::pricing_for_tier(pricing, service_tier),
            usage,
        )
    }
}

fn duration_cost(rate: f64, response_time_ms: f64) -> f64 {
    let ms = if response_time_ms.is_finite() && response_time_ms >= 0.0 {
        response_time_ms
    } else {
        0.0
    };
    sanitize_rate(sanitize_rate(rate) * ms / 1000.0)
}

/// Clamp a pricing rate to a sane billing value. Negative or non-finite
/// (NaN / ±∞) rates collapse to `0.0`. The vendored TOML is clean; such
/// values can only arise from an operator config typo (e.g.
/// `input_cost_per_token: -2.5e-6`), and clamping here ensures a malformed
/// rate can never produce a negative or non-finite cost — which would corrupt
/// the spend counter and the reservation/settle (`actual − estimate`) math.
fn sanitize_rate(rate: f64) -> f64 {
    if rate >= 0.0 && rate.is_finite() {
        rate
    } else {
        0.0
    }
}

/// Shared inner cost formula used by both the price-table path and the
/// deployment-override path. The caller has already resolved all rates via
/// [`TokenRates`] (see [`ModelPricing::rates_for_prompt`] and
/// [`TokenRates::from_override`] for the two fallback semantics). All rates
/// are sanitised ([`sanitize_rate`]) so a malformed (negative / NaN) rate
/// from operator config cannot yield a negative or non-finite cost.
///
/// The cache-creation bucket is split by ephemeral TTL when the 1h detail is
/// present: `cache_creation_1h_input_tokens` bill at `cache_creation_1h` and
/// the remainder at `cache_creation`. Anthropic reports this split via
/// `usage.cache_creation.ephemeral_5m_input_tokens` /
/// `ephemeral_1h_input_tokens`.
fn cost_from_rates(usage: &UsageSummary, rates: TokenRates) -> f64 {
    let input_rate = sanitize_rate(rates.input);
    let output_rate = sanitize_rate(rates.output);
    let cache_read_raw = usage.cache_read_input_tokens.unwrap_or(0);
    let cache_creation_total_raw = usage.cache_creation_input_tokens.unwrap_or(0);
    let cache_creation_1h_raw = usage.cache_creation_1h_input_tokens.unwrap_or(0);
    // When the aggregate cache-creation count is present, clamp the 1h detail
    // to it so inconsistent provider data can never over-bill. When the
    // aggregate is ABSENT, bill the 1h detail directly — matching the reference
    // `calculate_cache_writing_cost`, which uses the detail fields directly
    // when `cache_creation_token_details` is present and ignores the
    // aggregate. Treating the 1h detail as the effective total in that case
    // makes the 5m remainder zero and avoids a `total - 1h` underflow.
    let cache_creation_total = if usage.cache_creation_input_tokens.is_some() {
        cache_creation_total_raw
    } else {
        cache_creation_total_raw.max(cache_creation_1h_raw)
    };
    let cache_creation_1h = cache_creation_1h_raw.min(cache_creation_total);
    let audio_raw = usage.audio_input_tokens.unwrap_or(0);

    let prompt_base = usage.prompt_tokens;
    // `cache_read` and `audio` are both subsets of `prompt_tokens`. Clamp
    // their sum so inconsistent provider data cannot make the regular-input
    // bucket (and thus the cost) negative.
    let cache_read = cache_read_raw.min(prompt_base);
    let remaining = prompt_base - cache_read;
    let audio = audio_raw.min(remaining);
    let regular_input = (prompt_base - cache_read - audio) as f64;
    let cache_read = cache_read as f64;
    let audio = audio as f64;

    let mut cost = regular_input * input_rate;

    // Reasoning tokens are a subset of `completion_tokens`. When present,
    // split the output bucket: text tokens at `output_rate`, reasoning
    // tokens at `reasoning_rate` (falling back to `output_rate`). Clamp to
    // guard against inconsistent provider data underflowing the text bucket.
    let reasoning_raw = usage.reasoning_tokens.unwrap_or(0);
    let reasoning = reasoning_raw.min(usage.completion_tokens);
    let text_tokens = usage.completion_tokens - reasoning;
    cost += text_tokens as f64 * output_rate;
    if reasoning > 0 {
        let rate = sanitize_rate(rates.reasoning);
        cost += reasoning as f64 * rate;
    }

    if cache_read > 0.0 {
        cost += cache_read * sanitize_rate(rates.cache_read);
    }
    let cache_creation_5m = (cache_creation_total - cache_creation_1h) as f64;
    if cache_creation_5m > 0.0 {
        cost += cache_creation_5m * sanitize_rate(rates.cache_creation);
    }
    if cache_creation_1h > 0 {
        cost += cache_creation_1h as f64 * sanitize_rate(rates.cache_creation_1h);
    }
    if audio > 0.0 {
        cost += audio * sanitize_rate(rates.audio);
    }

    // Citation tokens (Perplexity Sonar/online). Billed additively at the
    // model's citation rate when declared; absent on non-citing models
    // (rate defaults to 0.0).
    let citation = usage.citation_tokens.unwrap_or(0) as f64;
    if citation > 0.0 {
        cost += citation * sanitize_rate(rates.citation);
    }
    // Web-search requests (Perplexity Sonar/online, OpenAI/xAI web-search).
    // Billed as a flat per-query fee additively.
    let search_queries = usage.web_search_requests.unwrap_or(0) as f64;
    if search_queries > 0.0 {
        cost += search_queries * sanitize_rate(rates.search_query);
    }

    // Final guard: every rate is sanitised and token counts are non-negative,
    // but a hostile/huge token count cast to f64 (lossy above 2^53) multiplied
    // by a large operator rate can overflow the running sum to +∞. Collapse
    // any non-finite result to 0.0 so the spend counter / reservation-settle
    // math can never be corrupted by provider input.
    if cost.is_finite() { cost } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    #[test]
    fn perplexity_sonar_bills_search_queries() {
        let mut u = usage(100, 50);
        u.web_search_requests = Some(3);
        let cost = calculate_cost("perplexity/sonar", &u).expect("sonar priced");
        let token_cost = 100.0 * 1e-6 + 50.0 * 1e-6;
        let search_cost = 3.0 * 0.008;
        assert!(
            (cost - (token_cost + search_cost)).abs() < 1e-9,
            "got {cost}, expected {}",
            token_cost + search_cost
        );
    }

    #[test]
    fn realtime_voice_rows_bill_per_elapsed_second() {
        // xAI grok-voice (and OpenAI live) sessions bill per wall-clock
        // second of the live session; the rows must be per-second priced so
        // duration settlement reaches them and lowest-cost routing does not
        // read their zero token rates as free.
        for model in [
            "xai/grok-voice-latest",
            "xai/grok-voice-2.0",
            "xai/grok-voice-3.0",
            "gpt-live-1",
        ] {
            let pricing = lookup_pricing(model).unwrap_or_else(|| panic!("{model} must be priced"));
            assert!(
                pricing.is_per_second_priced(),
                "{model}: realtime rows must bill per elapsed second"
            );
            let rate = pricing
                .input_cost_per_second
                .or(pricing.cost_per_second_rate)
                .unwrap_or_else(|| panic!("{model} per-second rate"));
            assert!(rate > 0.0, "{model}");
        }
        // Audio transcription seconds are media duration, not elapsed
        // request time — those rows stay off the wall-clock path.
        let transcription = lookup_pricing("xai/grok-voice-transcribe-2.0").expect("priced");
        assert!(!transcription.is_per_second_priced());
    }

    #[test]
    fn claude_hosted_web_search_bills_per_query_across_surfaces() {
        // Anthropic-family hosted web search reports per-tool counts under
        // `usage.server_tool_use`; every shipping surface carries the same
        // $0.01/query rate on its card. Compare against the same usage
        // without search requests so per-surface token rates stay irrelevant.
        for model in [
            "claude-haiku-5-5",
            "anthropic/claude-haiku-5-5",
            "us.anthropic.claude-haiku-5-5",
            "bedrock/us.anthropic.claude-haiku-5-5",
            "bedrock_mantle/anthropic.claude-haiku-5-5",
            "azure_ai/claude-haiku-5-5",
            "vertex_ai/claude-haiku-5-5",
        ] {
            let mut u = usage(100, 50);
            u.web_search_requests = Some(3);
            let with_search = calculate_cost(model, &u).unwrap_or_else(|| panic!("{model} priced"));
            u.web_search_requests = None;
            let without_search =
                calculate_cost(model, &u).unwrap_or_else(|| panic!("{model} priced"));
            assert!(
                ((with_search - without_search) - 3.0 * 0.01).abs() < 1e-9,
                "{model}: search delta {}",
                with_search - without_search
            );
        }
        // A card without a rate keeps billing zero rather than guessing.
        let mut u = usage(100, 50);
        u.web_search_requests = Some(3);
        let with_search = calculate_cost("claude-3-7-sonnet-20250219", &u).expect("priced");
        u.web_search_requests = None;
        let without_search = calculate_cost("claude-3-7-sonnet-20250219", &u).expect("priced");
        assert!(
            (with_search - without_search).abs() < 1e-12,
            "unrated card must not bill search"
        );
    }

    #[test]
    fn perplexity_deep_research_bills_citation_and_search() {
        let mut u = usage(100, 50);
        u.citation_tokens = Some(200);
        u.web_search_requests = Some(2);
        let cost =
            calculate_cost("perplexity/sonar-deep-research", &u).expect("deep-research priced");
        let token_cost = 100.0 * 2e-6 + 50.0 * 8e-6;
        let citation_cost = 200.0 * 2e-6;
        let search_cost = 2.0 * 0.005;
        assert!(
            (cost - (token_cost + citation_cost + search_cost)).abs() < 1e-9,
            "got {cost}, expected {}",
            token_cost + citation_cost + search_cost
        );
    }

    #[test]
    fn non_citing_model_bills_citation_tokens_at_zero() {
        // gpt-4o has no citation_cost_per_token rate; citation tokens must
        // bill at 0.0 (the _get_cost_per_unit default), not error.
        let mut u = usage(10, 5);
        u.citation_tokens = Some(100);
        u.web_search_requests = Some(5);
        let cost = calculate_cost("gpt-4o", &u).expect("gpt-4o priced");
        let token_cost = 10.0 * 0.0000025 + 5.0 * 0.00001;
        assert!(
            (cost - token_cost).abs() < 1e-9,
            "citation/search must be free on non-citing model: got {cost}"
        );
    }

    #[test]
    fn known_model_returns_correct_cost() {
        let u = usage(1_000_000, 500_000);
        let cost = calculate_cost("gpt-4o", &u).expect("gpt-4o should be priced");
        let expected = 1_000_000.0 * 0.0000025 + 500_000.0 * 0.00001;
        assert!(
            (cost - expected).abs() < 1e-9,
            "got {cost}, expected {expected}"
        );
    }

    #[test]
    fn dated_mini_tts_uses_the_published_text_input_rate() {
        let pricing = lookup_pricing("gpt-4o-mini-tts-2025-03-20").expect("dated TTS model");
        assert_eq!(pricing.input_cost_per_token, 6e-7);
        assert_eq!(pricing.output_cost_per_token, 1e-5);
        let cost = calculate_cost_with_provider(
            "gpt-4o-mini-tts-2025-03-20",
            &usage(1_000_000, 100),
            "openai",
        )
        .expect("TTS cost");
        assert!((cost - (0.6 + 100.0 * 1e-5)).abs() < 1e-12);
    }

    #[test]
    fn fireworks_small_model_rate_and_missing_cache_discount_are_correct() {
        let pricing = lookup_pricing("fireworks-ai-up-to-4b").expect("small-model band");
        assert_eq!(
            (pricing.input_cost_per_token, pricing.output_cost_per_token),
            (1e-7, 1e-7)
        );
        assert!(pricing.cache_read_input_token_cost.is_none());
        let mut u = usage(1_000_000, 400_000);
        u.cache_read_input_tokens = Some(200_000);
        let cost = calculate_cost_with_provider("fireworks-ai-up-to-4b", &u, "fireworks_ai")
            .expect("Fireworks cost");
        assert!((cost - (800_000.0 * 1e-7 + 200_000.0 * 5e-8 + 400_000.0 * 1e-7)).abs() < 1e-12);
        let adjacent = lookup_pricing("fireworks-ai-4.1b-to-16b").expect("adjacent band");
        assert_eq!(
            (
                adjacent.input_cost_per_token,
                adjacent.output_cost_per_token
            ),
            (2e-7, 2e-7)
        );
    }

    #[test]
    fn unknown_model_returns_none() {
        let u = usage(10, 5);
        assert!(calculate_cost("made-up-model", &u).is_none());
    }

    #[test]
    fn strips_provider_prefix() {
        let u = usage(1_000_000, 0);
        let bare = calculate_cost("gpt-4o", &u).expect("bare lookup");
        let prefixed = calculate_cost("openai/gpt-4o", &u).expect("prefixed lookup");
        assert!((bare - prefixed).abs() < 1e-12);
    }

    #[test]
    fn embedding_models_have_zero_output_cost() {
        let u = usage(1_000_000, 0);
        let cost =
            calculate_cost("text-embedding-3-small", &u).expect("embedding model should be priced");
        let expected = 1_000_000.0 * 0.00000002;
        assert!((cost - expected).abs() < 1e-12);
    }

    #[test]
    fn price_table_loads_successfully() {
        let table = price_table();
        assert!(table.contains_key("gpt-4o"));
        assert!(table.contains_key("claude-sonnet-4-20250514"));
        assert!(table.contains_key("text-embedding-ada-002"));
    }

    #[test]
    fn strip_provider_prefix_handles_cases() {
        assert_eq!(strip_provider_prefix("openai/gpt-4o"), "gpt-4o");
        assert_eq!(
            strip_provider_prefix("anthropic/claude-3-opus"),
            "claude-3-opus"
        );
        assert_eq!(strip_provider_prefix("gpt-4o"), "gpt-4o");
        assert_eq!(strip_provider_prefix("foo/"), "foo/");
    }

    /// Nested `openai/openai/<model>` deployment ids should still resolve pricing.
    #[test]
    fn lookup_pricing_strips_leading_openai_prefix() {
        let bare = lookup_pricing("gpt-4o-mini").expect("bare model");
        let prefixed = lookup_pricing("openai/gpt-4o-mini").expect("one prefix");
        assert_eq!(
            bare.input_cost_per_token, prefixed.input_cost_per_token,
            "single openai/ prefix should strip to gpt-4o-mini"
        );
    }

    /// Bug #36: Multi-segment model ids (`openrouter/openai/gpt-4o`,
    /// `together_ai/meta-llama/Llama-3.3-70B`) must unwind multiple
    /// provider prefixes to find pricing keyed by the bare model name.
    #[test]
    fn lookup_pricing_strips_multiple_provider_prefixes() {
        let bare = lookup_pricing("gpt-4o").expect("bare gpt-4o priced");
        let one_prefix = lookup_pricing("openai/gpt-4o").expect("single prefix");
        assert_eq!(
            bare.input_cost_per_token, one_prefix.input_cost_per_token,
            "single prefix should resolve"
        );
        let two_prefix = lookup_pricing("openrouter/openai/gpt-4o").expect("double prefix");
        assert_eq!(
            bare.input_cost_per_token, two_prefix.input_cost_per_token,
            "openrouter/openai/gpt-4o should strip down to gpt-4o"
        );
        // Ensure the multi-segment lookup actually resolves (does not return
        // `openai/gpt-4o` pricing — there is no such key in the table).
        let cost = calculate_cost("openrouter/openai/gpt-4o", &usage(10, 5)).expect("priced");
        assert!(cost > 0.0);
    }

    /// Regression: the old loop broke out as soon as `levels >= 3`, which fired
    /// BEFORE the bare model name (reached after the final strip) was ever
    /// looked up. A 4-segment prefix like `a/b/c/d/gpt-4o` must still resolve.
    #[test]
    fn lookup_pricing_strips_four_level_prefix() {
        let bare = lookup_pricing("gpt-4o").expect("bare gpt-4o priced");
        let four_prefix = lookup_pricing("a/b/c/d/gpt-4o").expect("four-segment prefix");
        assert_eq!(
            bare.input_cost_per_token, four_prefix.input_cost_per_token,
            "a/b/c/d/gpt-4o should strip down to gpt-4o"
        );
        // The resolved entry must genuinely be gpt-4o's record, not a phantom
        // match on an intermediate prefix.
        assert_eq!(four_prefix.provider, bare.provider);
    }

    #[test]
    fn lookup_pricing_strips_dated_snapshot_suffix() {
        let bare = lookup_pricing("gpt-4o").expect("bare gpt-4o");
        let dated = lookup_pricing("gpt-4o-2024-08-06").expect("dated snapshot");
        assert_eq!(bare.input_cost_per_token, dated.input_cost_per_token);
        let cost = calculate_cost("gpt-4o-2024-08-06", &usage(10, 5)).expect("cost");
        assert!(cost > 0.0);
    }

    #[test]
    fn lookup_pricing_unknown_model_returns_none() {
        assert!(lookup_pricing("totally-unknown-model-xyz").is_none());
    }

    #[test]
    fn lookup_pricing_finds_vertex_ai_anthropic_models() {
        // Vertex Claude models keep the `@YYYYMMDD` snapshot id which the
        // shared date-version stripper does not recognise, so they must be
        // present as explicit entries.
        let p = lookup_pricing("vertex_ai/claude-3-5-sonnet@20240620").expect("found");
        assert_eq!(p.provider, "vertex_ai");
        assert!((p.input_cost_per_token - 0.000003).abs() < 1e-15);
        assert!((p.output_cost_per_token - 0.000015).abs() < 1e-15);

        let p = lookup_pricing("vertex_ai/claude-3-opus@20240229").expect("found");
        assert_eq!(p.provider, "vertex_ai");
        assert!((p.input_cost_per_token - 0.000015).abs() < 1e-15);
    }

    #[test]
    fn lookup_pricing_finds_vertex_ai_gemini_models() {
        let p = lookup_pricing("vertex_ai/gemini-3-pro-preview").expect("found");
        assert_eq!(p.provider, "vertex_ai");
        assert!((p.input_cost_per_token - 0.000002).abs() < 1e-15);
    }

    #[test]
    fn parse_provider_prices_is_per_entry_fault_tolerant() {
        // A single malformed entry must not poison the whole shard.
        let raw = r#"
[good-model]
input_cost_per_token = 0.0000025
output_cost_per_token = 0.00001

[metadata-only]
# missing required cost fields

[stringified-costs]
input_cost_per_token = "2.5e-6"
output_cost_per_token = 0.00001
"#;
        let map = parse_provider_prices("openai", raw);
        assert!(map.contains_key("good-model"));
        assert_eq!(map["good-model"].provider, "openai");
        assert!(!map.contains_key("metadata-only"));
        assert!(!map.contains_key("stringified-costs"));
    }

    #[test]
    fn lookup_pricing_returns_fields() {
        let p = lookup_pricing("gpt-4o-mini").expect("found");
        assert_eq!(p.provider, "openai");
        assert!(p.cache_read_input_token_cost.is_some());
    }

    /// U34: `prompt_cache_min_tokens` must round-trip from the TOML shard
    /// through `parse_provider_prices` onto `ModelPricing`.
    #[test]
    fn parse_provider_prices_reads_prompt_cache_min_tokens() {
        let raw = r#"
[cached-model]
input_cost_per_token = 0.0000025
output_cost_per_token = 0.00001
prompt_cache_min_tokens = 2048

[uncached-model]
input_cost_per_token = 0.0000025
output_cost_per_token = 0.00001
"#;
        let map = parse_provider_prices("openai", raw);
        assert_eq!(
            map["cached-model"].prompt_cache_min_tokens,
            Some(2048),
            "prompt_cache_min_tokens must round-trip from TOML"
        );
        assert_eq!(
            map["uncached-model"].prompt_cache_min_tokens, None,
            "absent field must default to None"
        );
    }

    /// Every embedded price shard must parse to a non-empty map. A broken or
    /// accidentally-empty shard would silently zero out an entire provider's
    /// pricing (every model resolves to `None` → no spend accrues).
    #[test]
    fn all_price_shards_parse_non_empty() {
        for &(provider, raw) in PRICE_SHARDS {
            let shard = parse_provider_prices(provider, raw);
            assert!(
                !shard.is_empty(),
                "price shard '{provider}' parsed empty — every model would be unpriced"
            );
        }
    }

    /// Every TOML entry in every shard must deserialize into
    /// [`ModelPricingRow`] — zero dropped rows. A row that fails schema
    /// validation is silently dropped by `parse_provider_prices` (a runtime
    /// `tracing::warn!` that CI never sees), leaving its model unmetered
    /// despite the operator's data file carrying rates. This is exactly how
    /// the six input-only rows (mistral embeddings, gpt-image models) slipped
    /// through: `output_cost_per_token` was a required field they did not
    /// carry. This guard makes any future vendoring that trips the schema
    /// fail CI at the first rejected provider/model key instead.
    #[test]
    fn all_price_shards_drop_zero_rows() {
        for &(provider, raw) in PRICE_SHARDS {
            let table: toml::Table = raw
                .parse()
                .unwrap_or_else(|e| panic!("price shard '{provider}' failed to parse: {e}"));
            for (model, value) in &table {
                if let Err(e) = value.clone().try_into::<ModelPricingRow>() {
                    panic!(
                        "price shard '{provider}' row '{model}' does not match the \
                         ModelPricingRow schema ({e}) — it would be silently dropped and \
                         the model would bill $0"
                    );
                }
            }
        }
    }

    /// The six rows that previously failed schema validation (missing the
    /// then-required `output_cost_per_token`) must resolve and accrue spend:
    /// the mistral embeddings bill input-only (the catalog carries no output rate
    /// for them) and the gpt-image models bill their input/cache-read rates.
    #[test]
    fn previously_dropped_embedding_and_image_rows_price_again() {
        for (key, input_rate) in [
            ("mistral/codestral-embed", 1.5e-7),
            ("mistral/codestral-embed-2505", 1.5e-7),
            ("mistral/mistral-embed", 1.0e-7),
        ] {
            let p = lookup_pricing(key).unwrap_or_else(|| panic!("{key} must be priced"));
            assert!(
                (p.input_cost_per_token - input_rate).abs() < 1e-15,
                "{key}: input rate {} != {input_rate}",
                p.input_cost_per_token
            );
            assert_eq!(p.output_cost_per_token, 0.0, "{key} bills no output");
        }
        for (key, input_rate, output_rate) in [
            ("gpt-image-1", 5.0e-6, 4.0e-5),
            ("gpt-image-1-mini", 2.0e-6, 8.0e-6),
            ("chatgpt-image-latest", 5.0e-6, 1.0e-5),
        ] {
            let p = lookup_pricing(key).unwrap_or_else(|| panic!("{key} must be priced"));
            assert!(
                (p.input_cost_per_token - input_rate).abs() < 1e-15,
                "{key}: input rate {} != {input_rate}",
                p.input_cost_per_token
            );
            assert!(
                (p.output_cost_per_token - output_rate).abs() < 1e-15,
                "{key}: output rate {} != {output_rate}",
                p.output_cost_per_token
            );
        }
        let cost = calculate_cost("mistral/codestral-embed", &usage(1_000_000, 0))
            .expect("codestral-embed priced");
        assert!(
            (cost - 1_000_000.0 * 1.5e-7).abs() < 1e-12,
            "embedding spend must accrue input-only: got {cost}"
        );
    }

    /// Cross-shard duplicate model keys silently shadow ("later shard wins").
    /// This test scans all shards and fails on any cross-provider duplicate,
    /// whether or not the rates differ. Provider-specific rates must be scoped.
    ///
    /// The dangerous class is a duplicate whose rates DIFFER (a reseller entry
    /// silently overriding the canonical provider's pricing); but even a
    /// same-rate duplicate silently picks its `provider` attribution from
    /// `PRICE_SHARDS` ordering rather than the serving provider, which can
    /// misattribute spend.
    #[test]
    fn no_unexpected_cross_shard_duplicate() {
        // model -> Vec<(provider, input_rate, output_rate)>
        let mut seen: HashMap<String, Vec<(&str, f64, f64)>> = HashMap::new();
        for &(provider, raw) in PRICE_SHARDS {
            for (model, pricing) in parse_provider_prices(provider, raw) {
                seen.entry(model).or_default().push((
                    provider,
                    pricing.input_cost_per_token,
                    pricing.output_cost_per_token,
                ));
            }
        }

        for (model, entries) in seen {
            if entries.len() < 2 {
                continue;
            }
            for i in 0..entries.len() {
                for j in (i + 1)..entries.len() {
                    let (pa, ia, oa) = entries[i];
                    let (pb, ib, ob) = entries[j];
                    assert!(
                        pa == pb,
                        "cross-shard duplicate '{model}': \
                         {pa}({ia}, {oa}) vs {pb}({ib}, {ob}) — scope serving-provider rates"
                    );
                }
            }
        }
    }

    #[test]
    fn safeguard_pricing_retains_serving_provider_and_context_limits() {
        for model in [
            "openai.gpt-oss-safeguard-120b",
            "openai.gpt-oss-safeguard-20b",
        ] {
            let bedrock = lookup_pricing_with_provider(model, "bedrock").expect("Bedrock rate");
            let mantle =
                lookup_pricing_with_provider(model, "bedrock_mantle").expect("Mantle rate");
            assert_eq!(bedrock.provider, "bedrock");
            assert_eq!(mantle.provider, "bedrock_mantle");
            assert_eq!(bedrock.max_input_tokens, Some(128_000));
            assert_eq!(mantle.max_input_tokens, Some(131_072));
            assert_eq!(bedrock.max_output_tokens, Some(16_000));
            assert_eq!(mantle.max_output_tokens, Some(16_000));
            assert_eq!(bedrock.input_cost_per_token, mantle.input_cost_per_token);
            assert_eq!(bedrock.output_cost_per_token, mantle.output_cost_per_token);
        }
    }

    #[test]
    fn provider_scoped_pricing_is_not_prefixed_twice() {
        let u = usage(5, 1);
        let bare = calculate_cost_with_provider("sonar", &u, "perplexity").expect("native rate");
        let scoped = calculate_cost_with_provider("perplexity/sonar", &u, "perplexity")
            .expect("already scoped rate");
        let namespaced =
            calculate_cost_with_provider("perplexity/perplexity/sonar", &u, "perplexity")
                .expect("explicit namespaced rate");
        assert!((bare - 6e-6).abs() < 1e-15);
        assert_eq!(scoped, bare, "adding the known provider must be idempotent");
        assert!((namespaced - 3.75e-6).abs() < 1e-15);
        assert_ne!(
            namespaced, scoped,
            "explicit upstream namespaces retain their own rates"
        );
    }

    /// When two shards list the same bare key at identical rates, "later shard
    /// wins" picks the provider deterministically from `PRICE_SHARDS` ordering.
    /// The resolved `provider` is therefore a function of declaration order,
    /// NOT the serving provider — pin the order so the attribution is stable
    /// and documented (see `PRICE_SHARDS` at the top of this file).
    #[test]
    fn same_rate_cross_shard_duplicate_resolves_deterministically() {
        let row = r#"
input_cost_per_token = 1e-6
output_cost_per_token = 2e-6
"#;
        let a = parse_provider_prices("alpha", &format!("[shared-dup-model]\n{row}"));
        let b = parse_provider_prices("beta", &format!("[shared-dup-model]\n{row}"));
        // Emulate `load_all_prices`: later shard wins on collision.
        let mut merged = a;
        for (k, v) in b {
            merged.insert(k, v);
        }
        let resolved = merged
            .get("shared-dup-model")
            .expect("duplicate key collapses to exactly one entry");
        // Identical rates: cost is unaffected, but the provider attribution is
        // determined purely by merge order (beta, the later shard).
        assert_eq!(
            resolved.provider, "beta",
            "later shard must win the provider attribution for an identical-rate duplicate"
        );
        assert_eq!(resolved.input_cost_per_token, 1e-6);
        assert_eq!(resolved.output_cost_per_token, 2e-6);
    }

    /// Gemini's OpenAI-compat endpoint echoes the bare model back in the
    /// response (`gemini-2.5-flash`), but `gemini.toml` keys its rows with a
    /// `gemini/` prefix. The billing path strips that prefix on the way out
    /// and uses the response model for cost lookup, so a bare name must still
    /// resolve to the canonical gemini row — neither missing entirely nor
    /// silently shadowed by a different provider's bare alias (e.g. a Vertex
    /// AI entry with different rates). [`lookup_pricing_with_provider`] owns
    /// that resolution; this test pins it for every prefixed gemini key.
    #[test]
    fn gemini_prefixed_models_resolve_via_provider_aware_lookup() {
        let gemini_raw = PRICE_SHARDS
            .iter()
            .find(|(slug, _)| *slug == "gemini")
            .map(|(_, raw)| *raw)
            .expect("gemini price shard present");

        let table = price_table();
        let mut checked = 0u32;
        for (key, pricing) in parse_provider_prices("gemini", gemini_raw) {
            let Some(stripped) = key.strip_prefix("gemini/") else {
                continue;
            };
            // The bare (stripped) name is what the upstream response carries.
            let resolved = lookup_pricing_with_provider(stripped, "gemini").unwrap_or_else(|| {
                panic!("provider-aware lookup missed gemini key '{key}' (bare '{stripped}')")
            });
            assert_eq!(
                resolved.provider, "gemini",
                "bare '{stripped}' resolved to wrong provider {}",
                resolved.provider
            );
            assert!(
                (resolved.input_cost_per_token - pricing.input_cost_per_token).abs()
                    <= f64::EPSILON
                    && (resolved.output_cost_per_token - pricing.output_cost_per_token).abs()
                        <= f64::EPSILON,
                "bare '{stripped}' resolved to a row with different rates than '{key}'"
            );
            // Sanity: the resolved row is the exact same table entry as the
            // prefixed key (not a coincidental rate-equal alias from another
            // shard).
            assert!(
                std::ptr::eq(resolved, table.get(&key).unwrap()),
                "bare '{stripped}' did not resolve to the exact '{key}' row"
            );
            checked += 1;
        }
        assert!(
            checked > 20,
            "expected to check many gemini keys, got {checked}"
        );
    }

    #[test]
    fn deepseek_v4_and_fireworks_glm_are_priced() {
        let flash = lookup_pricing("deepseek-v4-flash").expect("deepseek-v4-flash");
        assert_eq!(flash.provider, "deepseek");
        assert!((flash.input_cost_per_token - 3e-7).abs() < 1e-15);

        let glm =
            lookup_pricing("fireworks_ai/accounts/fireworks/models/glm-5p2").expect("glm-5p2");
        assert_eq!(glm.provider, "fireworks_ai");
        assert!((glm.input_cost_per_token - 1.4e-6).abs() < 1e-15);
    }

    #[test]
    fn zero_usage_costs_zero() {
        let u = usage(0, 0);
        let cost = calculate_cost("gpt-4o", &u).expect("priced");
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn cache_read_exceeding_prompt_does_not_underflow() {
        let mut u = usage(100, 0);
        u.cache_read_input_tokens = Some(500);
        let cost = calculate_cost("gpt-4o", &u).expect("priced");
        assert!(cost >= 0.0, "cost must not go negative: {cost}");
    }

    #[test]
    fn anthropic_models_are_present() {
        assert!(lookup_pricing("claude-sonnet-4-20250514").is_some());
        assert!(lookup_pricing("claude-haiku-4-5").is_some());
        assert!(lookup_pricing("claude-3-opus-20240229").is_some());
    }

    /// Haiku 5.5 ships across every routed provider with its over-100k tiers.
    #[test]
    fn haiku_5_5_models_are_priced_with_context_tiers() {
        for model in [
            "claude-haiku-5-5",
            "vertex_ai/claude-haiku-5-5",
            "vertex_ai/claude-haiku-5-5@default",
            "bedrock/global.anthropic.claude-haiku-5-5",
            "bedrock/us.anthropic.claude-haiku-5-5",
            "bedrock/us-gov.anthropic.claude-haiku-5-5",
            "bedrock_mantle/anthropic.claude-haiku-5-5",
            "azure_ai/claude-haiku-5-5",
            "openrouter/anthropic/claude-haiku-5.5",
        ] {
            let pricing = lookup_pricing(model).unwrap_or_else(|| panic!("{model} must be priced"));
            assert!(
                pricing
                    .tiers
                    .iter()
                    .any(|tier| tier.threshold_tokens == 100_000),
                "{model} must carry the 100k context tier"
            );
        }
        // Perplexity's map entry carries base rates only.
        let reseller =
            lookup_pricing("perplexity/anthropic/claude-haiku-5-5").expect("perplexity haiku 5.5");
        assert!((reseller.input_cost_per_token - 1e-7).abs() < 1e-15);
        let pricing = lookup_pricing("claude-haiku-5-5").expect("anthropic haiku 5.5");
        assert!((pricing.input_cost_per_token - 1e-7).abs() < 1e-15);
        assert!((pricing.output_cost_per_token - 5e-7).abs() < 1e-15);
        // Prompt sizes beyond 100k bill every bucket at the 100k tier rates.
        let above = calculate_cost("claude-haiku-5-5", &usage(100_001, 10)).expect("tiered cost");
        let expected = 100_001.0 * 5e-7 + 10.0 * 2.5e-6;
        assert!(
            (above - expected).abs() < 1e-6,
            "got {above}, expected {expected}"
        );
        let below = calculate_cost("claude-haiku-5-5", &usage(99_999, 10)).expect("base cost");
        let expected = 99_999.0 * 1e-7 + 10.0 * 5e-7;
        assert!(
            (below - expected).abs() < 1e-6,
            "got {below}, expected {expected}"
        );
    }

    /// Sonnet 5.5's corrected cache-read rates must match the refreshed upstream map.
    #[test]
    fn sonnet_5_5_cache_read_prices_use_corrected_rates() {
        for (model, expected) in [
            ("claude-sonnet-5-5", 1e-7),
            ("vertex_ai/claude-sonnet-5-5", 1e-7),
            ("vertex_ai/claude-sonnet-5-5@default", 1e-7),
            ("bedrock/global.anthropic.claude-sonnet-5-5", 1e-7),
            ("bedrock/us.anthropic.claude-sonnet-5-5", 1.1e-7),
            ("bedrock/us-gov.anthropic.claude-sonnet-5-5", 1.2e-7),
            ("bedrock_mantle/anthropic.claude-sonnet-5-5", 1.1e-7),
            (
                "bedrock_mantle/us-gov-west-1/anthropic.claude-sonnet-5-5",
                1.2e-7,
            ),
        ] {
            let pricing = lookup_pricing(model).unwrap_or_else(|| panic!("{model} must be priced"));
            assert_eq!(
                pricing.cache_read_input_token_cost,
                Some(expected),
                "{model} cache-read rate"
            );
        }
    }

    #[test]
    fn bedrock_models_are_present() {
        assert!(lookup_pricing("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0").is_some());
        assert!(lookup_pricing("bedrock/anthropic.claude-3-opus-20240229-v1:0").is_some());
        assert!(lookup_pricing("bedrock/anthropic.claude-3-haiku-20240307-v1:0").is_some());
        assert!(lookup_pricing("bedrock/meta.llama3-1-70b-instruct-v1:0").is_some());
        assert!(lookup_pricing("bedrock/meta.llama3-1-8b-instruct-v1:0").is_some());
    }

    #[test]
    fn sonnet_4_5_flat_prices_keep_all_cache_buckets_past_200k() {
        for model in ["claude-sonnet-4-5", "claude-sonnet-4-5-20250929"] {
            let pricing = lookup_pricing_with_provider(model, "anthropic").expect("Sonnet card");
            assert_eq!(pricing.max_input_tokens, Some(200_000));
            assert_eq!(pricing.max_output_tokens, Some(64_000));
            assert!(
                pricing.tiers.is_empty(),
                "{model} must not carry a premium tier"
            );
            for context in [200_000, 200_001] {
                let u = UsageSummary {
                    prompt_tokens: context - 200,
                    completion_tokens: 100,
                    cache_read_input_tokens: Some(400),
                    cache_creation_input_tokens: Some(200),
                    cache_creation_1h_input_tokens: Some(50),
                    reasoning_tokens: Some(40),
                    ..Default::default()
                };
                let expected = (context - 600) as f64 * 3e-6
                    + 400.0 * 3e-7
                    + 150.0 * 3.75e-6
                    + 50.0 * 6e-6
                    + 100.0 * 1.5e-5;
                let cost =
                    calculate_cost_with_provider(model, &u, "anthropic").expect("Sonnet cost");
                assert!((cost - expected).abs() < 1e-12, "{model}/{context}: {cost}");
            }
        }
        let hosted = lookup_pricing("vertex_ai/claude-sonnet-4-5").expect("hosted Sonnet card");
        assert!(
            !hosted.tiers.is_empty(),
            "hosted prices retain their own tiers"
        );
    }

    #[test]
    fn corrected_bedrock_rates_bill_regional_cache_and_token_prices() {
        for (model, input, output, cache) in [
            (
                "apac.amazon.nova-2-lite-v1:0",
                3.96e-7,
                3.311e-6,
                Some(9.9e-8),
            ),
            (
                "eu.amazon.nova-2-lite-v1:0",
                4.29e-7,
                3.597e-6,
                Some(1.0725e-7),
            ),
            ("amazon.nova-premier-v1:0", 2.5e-6, 1.25e-5, Some(6.25e-7)),
            ("bedrock/moonshotai.kimi-k2-thinking", 6e-7, 2.5e-6, None),
            ("bedrock/moonshotai.kimi-k2.5", 6e-7, 3e-6, None),
            ("amazon.titan-text-express-v1", 2e-7, 6e-7, None),
            ("amazon.titan-text-lite-v1", 1.5e-7, 2e-7, None),
            (
                "bedrock/us-gov-west-1/amazon.titan-text-express-v1",
                8e-7,
                1.6e-6,
                None,
            ),
            (
                "bedrock/us-gov-west-1/amazon.titan-embed-text-v2:0",
                1.1e-7,
                0.0,
                None,
            ),
        ] {
            let cached = if cache.is_some() { 400 } else { 0 };
            let u = UsageSummary {
                cache_read_input_tokens: Some(cached),
                ..usage(1_000, 100)
            };
            let cost = calculate_cost_with_provider(model, &u, "bedrock").expect("Bedrock cost");
            let expected = (1_000 - cached) as f64 * input
                + cached as f64 * cache.unwrap_or(0.0)
                + 100.0 * output;
            assert!((cost - expected).abs() < 1e-12, "{model}: {cost}");
        }
    }

    #[test]
    fn hosted_catalog_limits_keep_input_windows_separate_from_output_caps() {
        for (provider, model, input, output) in [
            ("bedrock", "amazon.nova-premier-v1:0", 1_000_000, 10_000),
            (
                "bedrock",
                "bedrock/ap-northeast-1/deepseek.v3.2",
                163_840,
                8_000,
            ),
            ("bedrock", "deepseek.v3-v1:0", 128_000, 8_000),
            (
                "bedrock",
                "bedrock/moonshotai.kimi-k2-thinking",
                262_144,
                16_000,
            ),
            (
                "bedrock",
                "bedrock/us-west-2/moonshotai.kimi-k2.5",
                262_144,
                16_000,
            ),
            (
                "bedrock",
                "bedrock/us-east-1/minimax.minimax-m2.5",
                196_000,
                8_192,
            ),
            ("bedrock", "mistral.voxtral-mini-3b-2507", 32_768, 8_192),
            ("bedrock", "mistral.voxtral-small-24b-2507", 32_768, 8_192),
            ("bedrock", "openai.gpt-oss-120b-1:0", 128_000, 16_384),
            (
                "bedrock",
                "bedrock/us-gov-west-1/openai.gpt-oss-20b-1:0",
                128_000,
                16_384,
            ),
            ("bedrock_mantle", "openai.gpt-oss-120b", 131_072, 16_384),
            (
                "bedrock_mantle",
                "us-gov-east-1/openai.gpt-oss-20b",
                131_072,
                16_384,
            ),
            (
                "bedrock_mantle",
                "openai.gpt-oss-safeguard-120b",
                131_072,
                16_000,
            ),
            (
                "bedrock_mantle",
                "openai.gpt-daybreak-blue-5.6-sol",
                1_000_000,
                128_000,
            ),
            ("bedrock", "global.openai.gpt-6.1-sol", 1_000_000, 131_072),
            ("bedrock", "us.openai.gpt-6.1-sol", 1_000_000, 131_072),
            ("bedrock_mantle", "openai.gpt-6.1-sol", 1_000_000, 131_072),
            ("openai", "gpt-6.1-sol", 922_000, 128_000),
        ] {
            let pricing = lookup_pricing_with_provider(model, provider).expect("hosted card");
            assert_eq!(pricing.max_input_tokens, Some(input), "{provider}/{model}");
            assert_eq!(
                pricing.max_output_tokens,
                Some(output),
                "{provider}/{model}"
            );
        }
        for region in ["global", "us"] {
            for model in [
                "gpt-5.4",
                "gpt-5.5",
                "gpt-5.6-luna",
                "gpt-5.6-sol",
                "gpt-5.6-terra",
            ] {
                let key = format!("{region}.openai.{model}");
                let pricing = lookup_pricing_with_provider(&key, "bedrock").expect("GPT card");
                assert_eq!(pricing.max_input_tokens, Some(1_050_000), "{key}");
                assert_eq!(pricing.max_output_tokens, Some(128_000), "{key}");
            }
        }
        for alias in ["openai.gpt-6.1-sol", "xai.grok-4.7"] {
            assert!(
                !price_table().contains_key(alias),
                "retired bare id {alias}"
            );
            assert!(lookup_pricing_with_provider(alias, "bedrock").is_none());
        }
        assert!(lookup_pricing("global.xai.grok-4.7").is_some());
    }

    #[test]
    fn gpt_6_1_ultrafast_bills_cache_and_output_at_exclusive_context_boundary() {
        for (provider, model, multiplier) in [
            ("openai", "gpt-6.1-sol", 1.0),
            ("bedrock", "global.openai.gpt-6.1-sol", 1.0),
            ("bedrock", "us.openai.gpt-6.1-sol", 1.1),
            ("bedrock_mantle", "openai.gpt-6.1-sol", 1.1),
        ] {
            for context in [272_000, 272_001] {
                let u = UsageSummary {
                    prompt_tokens: context - 100,
                    completion_tokens: 100,
                    cache_read_input_tokens: Some(400),
                    cache_creation_input_tokens: Some(100),
                    reasoning_tokens: Some(40),
                    ..Default::default()
                };
                let (input, output, read, write) = if context > 272_000 {
                    (2.4e-5, 9e-5, 1.2e-6, 3e-5)
                } else {
                    (1.2e-5, 6e-5, 6e-7, 1.5e-5)
                };
                let expected = ((context - 500) as f64 * input
                    + 400.0 * read
                    + 100.0 * write
                    + 100.0 * output)
                    * multiplier;
                let cost =
                    calculate_cost_for_service_tier(model, &u, None, provider, Some("ultrafast"))
                        .expect("UltraFast price");
                assert!((cost - expected).abs() < 1e-12, "{model}/{context}: {cost}");

                let override_prices = DeploymentPricing {
                    input_cost_per_token: 0.001,
                    output_cost_per_token: 0.002,
                    cache_read_input_token_cost: Some(0.0001),
                    cache_creation_input_token_cost: Some(0.0005),
                    ..Default::default()
                };
                let overridden = calculate_cost_for_service_tier(
                    model,
                    &u,
                    Some(&override_prices),
                    provider,
                    Some("ultrafast"),
                )
                .expect("explicit deployment price");
                let expected_override = (context - 500) as f64 * 0.001
                    + 400.0 * 0.0001
                    + 100.0 * 0.0005
                    + 100.0 * 0.002;
                assert!((overridden - expected_override).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn bedrock_flex_prices_use_published_buckets_instead_of_inferred_discounts() {
        for (provider, model, input, output, cache) in [
            ("bedrock", "google.gemma-3-12b-it", 5e-8, 1.5e-7, None),
            (
                "bedrock",
                "mistral.ministral-3-8b-instruct",
                7e-8,
                7e-8,
                None,
            ),
            ("bedrock", "nvidia.nemotron-nano-9b-v2", 3e-8, 1.2e-7, None),
            (
                "bedrock",
                "us.amazon.nova-2-lite-v1:0",
                1.65e-7,
                1.375e-6,
                Some(4.125e-8),
            ),
            (
                "bedrock",
                "bedrock/eu-west-2/qwen.qwen3-coder-next",
                3.9e-7,
                9.3e-7,
                None,
            ),
            ("bedrock_mantle", "google.gemma-4-31b", 7e-8, 2e-7, None),
            (
                "bedrock_mantle",
                "qwen.qwen3-vl-235b-a22b-instruct",
                2.6e-7,
                1.33e-6,
                None,
            ),
            (
                "bedrock_mantle",
                "openai.gpt-oss-safeguard-120b",
                7e-8,
                3e-7,
                None,
            ),
        ] {
            let cached = if cache.is_some() { 400 } else { 0 };
            let u = UsageSummary {
                cache_read_input_tokens: Some(cached),
                reasoning_tokens: Some(40),
                ..usage(1_000, 100)
            };
            let cost = calculate_cost_for_service_tier(model, &u, None, provider, Some("flex"))
                .expect("Flex price");
            let expected = (1_000 - cached) as f64 * input
                + cached as f64 * cache.unwrap_or(0.0)
                + 100.0 * output;
            assert!(
                (cost - expected).abs() < 1e-12,
                "{provider}/{model}: {cost}"
            );
        }
        for alias in [
            "google.gemma-4-26b-a4b",
            "google.gemma-4-31b",
            "google.gemma-4-e2b",
            "openai.gpt-oss-120b",
            "openai.gpt-oss-20b",
        ] {
            let bare = lookup_pricing(alias).expect("unscoped Mantle card");
            let scoped =
                lookup_pricing_with_provider(alias, "bedrock_mantle").expect("Mantle card");
            let a = bare.rates_for_service_tier(1_000, 0, Some("flex"));
            let b = scoped.rates_for_service_tier(1_000, 0, Some("flex"));
            assert_eq!((a.input, a.output), (b.input, b.output), "{alias}");
        }
    }

    #[test]
    fn new_catalog_cards_keep_serving_namespaces_and_unpublished_limits() {
        for (model, input, output, cache, write) in [
            ("openai/gpt-5-nano", 5e-8, 4e-7, 5e-9, None),
            ("openai/gpt-5.3-codex", 3.5e-6, 2.8e-5, 3.5e-7, None),
            ("openai/gpt-6-astra", 1e-5, 5e-5, 1e-6, Some(1.25e-5)),
            (
                "perplexity/deepseek-v4-pro-0813",
                1.32e-6,
                3.96e-6,
                4.4e-8,
                None,
            ),
        ] {
            let pricing = lookup_pricing_with_provider(model, "perplexity").expect("Agent card");
            assert_eq!(pricing.provider, "perplexity");
            assert_eq!(pricing.mode.as_deref(), Some("responses"));
            assert_eq!(pricing.max_input_tokens, None);
            assert_eq!(pricing.max_output_tokens, None);
            let u = UsageSummary {
                cache_read_input_tokens: Some(400),
                cache_creation_input_tokens: write.map(|_| 200),
                ..usage(1_000, 100)
            };
            let cost = calculate_cost_with_provider(model, &u, "perplexity").expect("Agent cost");
            let expected =
                600.0 * input + 400.0 * cache + 100.0 * output + 200.0 * write.unwrap_or(0.0);
            assert!((cost - expected).abs() < 1e-12, "{model}: {cost}");
        }
        let discovery = lookup_pricing("gpt-rosalind-discovery").expect("Discovery card");
        assert_eq!(discovery.provider, "openai");
        assert_eq!(
            (discovery.max_input_tokens, discovery.max_output_tokens),
            (None, None)
        );
        let u = UsageSummary {
            cache_read_input_tokens: Some(400),
            ..usage(1_000, 100)
        };
        let cost = calculate_cost("gpt-rosalind-discovery", &u).expect("Discovery cost");
        assert!((cost - (600.0 * 5e-6 + 400.0 * 5e-7 + 100.0 * 2.5e-5)).abs() < 1e-12);
        for (model, rate, mode, context) in [
            ("Cohere-Embed-V5-Fast", 8e-8, "embedding", Some(128_000)),
            ("Cohere-Embed-V5-Pro", 1.2e-7, "embedding", Some(128_000)),
        ] {
            let pricing = lookup_pricing_with_provider(model, "azure_ai").expect("Foundry card");
            assert_eq!(pricing.provider, "azure_ai");
            assert_eq!(pricing.mode.as_deref(), Some(mode));
            assert_eq!(pricing.max_input_tokens, context);
            assert_eq!(pricing.max_output_tokens, None);
            assert_eq!(pricing.output_cost_per_token, 0.0);
            let cost = calculate_cost_with_provider(model, &usage(1_000, 0), "azure_ai")
                .expect("Foundry input cost");
            assert!((cost - 1_000.0 * rate).abs() < 1e-12, "{model}: {cost}");
        }
    }

    #[test]
    fn mantle_bare_aliases_match_scoped_rates_limits_and_all_tiers() {
        let rows: toml::Table =
            toml::from_str(include_str!("../data/prices/bedrock_mantle.toml")).unwrap();
        let mut checked = 0;
        for (alias, row) in rows.iter().filter(|(name, _)| !name.contains('/')) {
            let canonical = format!("bedrock_mantle/{alias}");
            assert_eq!(
                row, &rows[&canonical],
                "bare Mantle alias {alias} must not drift from its serving card"
            );
            checked += 1;
        }
        assert_eq!(checked, 11, "retain every existing bare alias");
    }

    #[test]
    fn bedrock_claude_pricing_inherits_anthropic_cache_rates() {
        let p = lookup_pricing("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
            .expect("bedrock claude priced");
        assert_eq!(p.input_cost_per_token, 0.000003);
        assert_eq!(p.output_cost_per_token, 0.000015);
        assert!(p.cache_read_input_token_cost.is_some());
        assert!(p.cache_creation_input_token_cost.is_some());
        assert_eq!(p.provider, "bedrock");
    }

    #[test]
    fn meta_models_are_present() {
        let p = lookup_pricing("meta/muse-spark-1.1").expect("muse-spark-1.1 priced");
        assert_eq!(p.input_cost_per_token, 1.25e-6, "input $1.25/M");
        assert_eq!(p.output_cost_per_token, 4.25e-6, "output $4.25/M");
        assert!(p.output_cost_per_reasoning_token.is_none());
        let mut reasoned = usage(0, 100);
        reasoned.reasoning_tokens = Some(100);
        let cost = calculate_cost("meta/muse-spark-1.1", &reasoned).expect("cost");
        assert!(
            (cost - 100.0 * 4.25e-6).abs() < 1e-12,
            "reasoning still bills at the output rate: got {cost}"
        );
        assert_eq!(p.provider, "meta");
        let bare = lookup_pricing("muse-spark-1.1").expect("bare lookup");
        assert_eq!(bare.input_cost_per_token, 1.25e-6);
    }

    #[test]
    fn threshold_key_parsing() {
        assert_eq!(
            parse_threshold_key("input_cost_per_token_above_128k_tokens"),
            Some(("input_cost_per_token", 128_000))
        );
        assert_eq!(
            parse_threshold_key("input_cost_per_token_above_272k_tokens"),
            Some(("input_cost_per_token", 272_000))
        );
        assert_eq!(
            parse_threshold_key("output_cost_per_token_above_200k_tokens"),
            Some(("output_cost_per_token", 200_000))
        );
        assert_eq!(
            parse_threshold_key("cache_read_input_token_cost_above_128_tokens"),
            Some(("cache_read_input_token_cost", 128))
        );
        // Non-threshold keys are rejected.
        assert_eq!(parse_threshold_key("input_cost_per_token"), None);
        assert_eq!(
            parse_threshold_key("cache_creation_input_token_cost_above_1hr"),
            None
        );
        assert_eq!(
            parse_threshold_key("input_cost_per_token_above_128k_tokens_priority"),
            None
        );
        // The combined Anthropic 1h cache-creation tier key resolves to the
        // `cache_creation_input_token_cost_above_1hr` base.
        assert_eq!(
            parse_threshold_key("cache_creation_input_token_cost_above_1hr_above_200k_tokens"),
            Some(("cache_creation_input_token_cost_above_1hr", 200_000))
        );
        assert_eq!(
            parse_threshold_key(
                "cache_creation_input_token_cost_above_1hr_above_200k_tokens_priority"
            ),
            None
        );
    }

    #[test]
    fn tier_parsing_sorts_descending_and_merges_per_threshold() {
        let raw: toml::Table = r#"
input_cost_per_token = 1e-6
input_cost_per_token_above_100k_tokens = 2e-6
output_cost_per_token_above_200k_tokens = 1e-5
cache_read_input_token_cost_above_200k_tokens = 2e-7
"#
        .parse()
        .expect("parse");
        let tiers = extract_cost_tiers(&toml::Value::Table(raw));
        assert_eq!(tiers.len(), 2);
        assert_eq!(tiers[0].threshold_tokens, 200_000, "descending sort");
        assert_eq!(tiers[1].threshold_tokens, 100_000);
        assert_eq!(tiers[0].output_cost_per_token, Some(1e-5));
        assert_eq!(tiers[0].cache_read_input_token_cost, Some(2e-7));
        assert_eq!(
            tiers[0].input_cost_per_token, None,
            "merged row: input only in the 100k tier"
        );
        assert_eq!(tiers[1].input_cost_per_token, Some(2e-6));
    }

    #[test]
    fn tiered_cost_below_threshold_uses_base_rates() {
        let p = lookup_pricing("tiered-test-model").expect("test shard loaded");
        assert!(!p.tiers.is_empty());
        let u = usage(99_999, 1_000);
        let cost = calculate_cost("tiered-test-model", &u).expect("cost");
        assert_eq!(cost, 99_999.0 * 1e-6 + 1_000.0 * 5e-6);
    }

    #[test]
    fn tiered_cost_at_exact_threshold_stays_on_base_rates() {
        // Exclusive boundary (>) for every provider except xAI, mirroring
        // Inclusive-threshold semantics: at exactly 100k
        // prompt tokens (test shard is not xai), the tier does NOT fire and
        // the base rates apply. One token more activates the tier.
        let u = usage(100_000, 1_000);
        let cost = calculate_cost("tiered-test-model", &u).expect("cost");
        assert_eq!(cost, 100_000.0 * 1e-6 + 1_000.0 * 5e-6);
        let u = usage(100_001, 1_000);
        let cost = calculate_cost("tiered-test-model", &u).expect("cost");
        assert_eq!(cost, 100_001.0 * 2e-6 + 1_000.0 * 5e-6);
    }

    /// Shipped-provider boundary parity. Tier thresholds are exclusive (`>`)
    /// for every provider except xAI, so a request landing exactly on a
    /// shipped tier threshold bills at the base rates on Gemini and OpenAI.
    /// The old unconditional `>=` over-billed those requests 2x (gpt-5.6 at
    /// exactly 272k: 8e-6 vs 4e-6 per input token).
    #[test]
    fn exact_threshold_bills_base_rates_on_shipped_non_xai_models() {
        // gemini/gemini-2.5-pro: 200k tier input 2.5e-6 vs base 1.25e-6.
        let p = lookup_pricing("gemini/gemini-2.5-pro").expect("gemini-2.5-pro priced");
        assert!(!p.tier_threshold_is_inclusive);
        let u = usage(200_000, 0);
        let cost = calculate_cost_with_provider("gemini-2.5-pro", &u, "gemini").expect("cost");
        assert_eq!(cost, 200_000.0 * 1.25e-6);

        // gpt-5.6: 272k tier input 8e-6 vs base 4e-6.
        let p = lookup_pricing("gpt-5.6").expect("gpt-5.6 priced");
        assert!(!p.tier_threshold_is_inclusive);
        let u = usage(272_000, 0);
        let cost = calculate_cost("gpt-5.6", &u).expect("cost");
        assert_eq!(cost, 272_000.0 * 4e-6);
    }

    /// MEDIUM (C64 round 2): xAI is the catalog's only inclusive-threshold
    /// provider (`_INCLUSIVE_THRESHOLD_PROVIDERS = {"xai"}`) — the tier rate
    /// kicks in AT the threshold. A request on
    /// `xai/grok-4.3` landing exactly on 200k context tokens must bill at
    /// the tier rate (2.5e-6), not the base rate (1.25e-6).
    #[test]
    fn exact_threshold_activates_tier_on_xai() {
        let p = lookup_pricing("xai/grok-4.3").expect("xai/grok-4.3 priced");
        assert!(
            p.tier_threshold_is_inclusive,
            "xai rows must use inclusive tier thresholds"
        );
        let u = usage(200_000, 0);
        let cost = calculate_cost("xai/grok-4.3", &u).expect("cost");
        assert_eq!(cost, 200_000.0 * 2.5e-6);
        // One below the threshold still bills at the base rate.
        let u = usage(199_999, 0);
        let cost = calculate_cost("xai/grok-4.3", &u).expect("cost");
        assert_eq!(cost, 199_999.0 * 1.25e-6);
    }

    #[test]
    fn tiered_cost_above_threshold_applies_partial_tier_rates() {
        // 150k prompt exceeds only the 100k tier: input takes the tier
        // rate, output falls back to the base rate.
        let u = usage(150_000, 1_000);
        let cost = calculate_cost("tiered-test-model", &u).expect("cost");
        assert_eq!(cost, 150_000.0 * 2e-6 + 1_000.0 * 5e-6);
    }

    #[test]
    fn tiered_cost_highest_threshold_wins_with_fallback_to_base() {
        // 250k exceeds both tiers; the 200k tier wins. Its missing input
        // rate falls back to the BASE input rate (not the 100k tier).
        let u = usage(250_000, 1_000);
        let cost = calculate_cost("tiered-test-model", &u).expect("cost");
        assert_eq!(cost, 250_000.0 * 1e-6 + 1_000.0 * 1e-5);
    }

    #[test]
    fn tiered_cost_applies_tier_to_cache_tokens() {
        // 250k prompt with 100k cached: cache-read bucket at the 200k
        // tier's rate (2e-7), remaining 150k input at the fallback base
        // rate (1e-6).
        let mut u = usage(250_000, 500);
        u.cache_read_input_tokens = Some(100_000);
        let cost = calculate_cost("tiered-test-model", &u).expect("cost");
        assert_eq!(cost, 150_000.0 * 1e-6 + 100_000.0 * 2e-7 + 500.0 * 1e-5);
    }

    #[test]
    fn tiered_cost_with_provider_and_bare_lookup_agree() {
        let u = usage(150_000, 100);
        let bare = calculate_cost("tiered-test-model", &u).expect("bare");
        let with_provider = calculate_cost_with_provider("tiered-test-model", &u, "test_tiers")
            .expect("with provider");
        assert_eq!(bare, with_provider);
        assert_eq!(bare, 150_000.0 * 2e-6 + 100.0 * 5e-6);
    }

    #[test]
    fn gpt_56_real_tier_data() {
        let p = lookup_pricing("gpt-5.6").expect("gpt-5.6 priced");
        assert_eq!(p.tiers.len(), 1);
        assert_eq!(p.tiers[0].threshold_tokens, 272_000);
        let below = usage(200_000, 100);
        assert_eq!(
            calculate_cost("gpt-5.6", &below).expect("cost"),
            200_000.0 * 4e-6 + 100.0 * 2e-5
        );
        let above = usage(300_000, 100);
        assert_eq!(
            calculate_cost("gpt-5.6", &above).expect("cost"),
            300_000.0 * 8e-6 + 100.0 * 3e-5
        );
        let mut cached = usage(300_000, 0);
        cached.cache_read_input_tokens = Some(50_000);
        let cost = calculate_cost("gpt-5.6", &cached).expect("cost");
        assert_eq!(cost, 250_000.0 * 8e-6 + 50_000.0 * 8e-7);
    }

    #[test]
    fn gemini_25_pro_real_tier_data() {
        let p = lookup_pricing("gemini/gemini-2.5-pro").expect("gemini-2.5-pro priced");
        assert_eq!(p.tiers.len(), 1);
        assert_eq!(p.tiers[0].threshold_tokens, 200_000);
        let u = usage(250_000, 1_000);
        let cost = calculate_cost_with_provider("gemini-2.5-pro", &u, "gemini").expect("cost");
        assert_eq!(cost, 250_000.0 * 2.5e-6 + 1_000.0 * 1.5e-5);
    }

    /// U-hr: Anthropic prices 1h-ephemeral cache-creation tokens at a distinct
    /// (usually higher) rate via `cache_creation_input_token_cost_above_1hr`.
    /// The 1h bucket must bill at that rate and the remainder of the
    /// cache-creation bucket at the 5m rate.
    #[test]
    fn cache_creation_1h_tokens_bill_at_distinct_rate() {
        let pricing =
            lookup_pricing("claude-sonnet-4-20250514").expect("claude-sonnet-4-20250514 priced");
        let five_m = pricing.cache_creation_input_token_cost.expect("5m rate");
        let one_h = pricing
            .cache_creation_input_token_cost_above_1hr
            .expect("1h rate");
        assert!(
            one_h > five_m,
            "test premise: 1h rate must differ from the 5m rate"
        );
        let prompt = 100u64;
        let created = 500u64;
        let created_1h = 200u64;
        let completion = 50u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: Some(created),
            cache_creation_1h_input_tokens: Some(created_1h),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost = calculate_cost("claude-sonnet-4-20250514", &u).expect("priced");
        let expected = prompt as f64 * pricing.input_cost_per_token
            + (created - created_1h) as f64 * five_m
            + created_1h as f64 * one_h
            + completion as f64 * pricing.output_cost_per_token;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
        // Sanity: billing the 1h tokens at the cheaper 5m rate under-bills.
        let under_billed = prompt as f64 * pricing.input_cost_per_token
            + created as f64 * five_m
            + completion as f64 * pricing.output_cost_per_token;
        assert!(
            (cost - under_billed).abs() > 1e-9,
            "1h cache-creation tokens must bill at the 1h rate, got {cost}"
        );
    }

    /// A 1h bucket larger than the total cache-creation count (inconsistent
    /// provider data) must clamp to the total and never bill more than the
    /// reported cache-creation tokens.
    #[test]
    fn cache_creation_1h_bucket_clamped_to_total() {
        let pricing =
            lookup_pricing("claude-sonnet-4-20250514").expect("claude-sonnet-4-20250514 priced");
        let u = UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 0,
            total_tokens: 100,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: Some(300),
            cache_creation_1h_input_tokens: Some(1_000),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost = calculate_cost("claude-sonnet-4-20250514", &u).expect("priced");
        let one_h = pricing
            .cache_creation_input_token_cost_above_1hr
            .expect("1h rate");
        // 1h clamped to 300; the 5m remainder is 0.
        let expected = 100.0 * pricing.input_cost_per_token + 300.0 * one_h;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    /// Table-path rate fallback (`_get_cost_per_unit(..., default=0.0)` semantics):
    /// a model entry lacking cache-read, cache-creation, and audio rates bills
    /// those buckets at 0 — cached/audio tokens are FREE, and are only removed
    /// from the regular input bucket. This deliberately differs from the
    /// [`DeploymentPricing`] override path, which falls back to the input rate.
    #[test]
    fn missing_cache_and_audio_rates_bill_zero_on_table_path() {
        let pricing = ModelPricing {
            input_cost_per_token: 1e-6,
            output_cost_per_token: 2e-6,
            input_cost_per_audio_token: None,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            output_cost_per_reasoning_token: None,
            citation_cost_per_token: None,
            search_queries_cost_per_query: None,
            max_input_tokens: None,
            max_output_tokens: None,
            provider: "test".to_string(),
            input_cost_per_second: None,
            prompt_cache_min_tokens: None,
            tiers: Vec::new(),
            tier_threshold_is_inclusive: false,
            ..Default::default()
        };
        let rates = pricing.rates_for_prompt(1_000, 0);
        assert_eq!(rates.cache_read, 0.0);
        assert_eq!(rates.cache_creation, 0.0);
        assert_eq!(rates.cache_creation_1h, 0.0);
        assert_eq!(rates.audio, 0.0);

        let prompt = 1_000u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: prompt,
            cache_read_input_tokens: Some(400),
            cache_creation_input_tokens: Some(100),
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: Some(50),
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost = cost_from_rates(&u, rates);
        let expected = (prompt - 400 - 50) as f64 * 1e-6;
        assert!(
            (cost - expected).abs() < 1e-12,
            "missing cache/audio rates must bill those buckets free: got {cost}, expected {expected}"
        );

        // End-to-end on a shipped model with no audio or cache-creation rate:
        // gpt-4o audio tokens bill free (not at the 2.5e-6 input rate).
        let gpt4o = lookup_pricing("gpt-4o").expect("gpt-4o priced");
        assert!(gpt4o.input_cost_per_audio_token.is_none());
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: prompt,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: Some(200),
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost = calculate_cost("gpt-4o", &u).expect("priced");
        let expected = (prompt - 200) as f64 * gpt4o.input_cost_per_token;
        assert!(
            (cost - expected).abs() < 1e-12,
            "gpt-4o audio must bill free on the table path: got {cost}, expected {expected}"
        );
    }

    #[test]
    fn fireworks_missing_cache_read_rate_bills_half_the_base_input_rate() {
        let pricing = ModelPricing {
            input_cost_per_token: 2e-6,
            output_cost_per_token: 4e-6,
            input_cost_per_audio_token: None,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            output_cost_per_reasoning_token: None,
            citation_cost_per_token: None,
            search_queries_cost_per_query: None,
            max_input_tokens: None,
            max_output_tokens: None,
            provider: "fireworks_ai".to_string(),
            input_cost_per_second: None,
            prompt_cache_min_tokens: None,
            tiers: vec![CostTier {
                threshold_tokens: 200_000,
                input_cost_per_token: Some(4e-6),
                output_cost_per_token: None,
                cache_read_input_token_cost: None,
                cache_creation_input_token_cost: None,
                cache_creation_input_token_cost_above_1hr: None,
            }],
            tier_threshold_is_inclusive: false,
            ..Default::default()
        };
        let base = pricing.rates_for_prompt(1_000, 0);
        assert_eq!(base.cache_read, 1e-6);
        // A long-context tier that does not publish its own cache-read rate
        // keeps the base-input discount rather than half the tier input rate.
        let long = pricing.rates_for_prompt(250_000, 0);
        assert_eq!(long.input, 4e-6);
        assert_eq!(long.cache_read, 1e-6);

        let prompt = 1_000u64;
        let cached = 400u64;
        let mut u = usage(prompt, 0);
        u.cache_read_input_tokens = Some(cached);
        let cost = cost_from_pricing(&pricing, &u).expect("cost");
        let expected = (prompt - cached) as f64 * 2e-6 + cached as f64 * 1e-6;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    /// Tier thresholds are evaluated against the NORMALIZED prompt
    /// (`UsageSummary.prompt_tokens`), which for Anthropic includes the folded
    /// additive cache-read tokens — matching the reference, whose Anthropic
    /// transformation also adds `cache_read_input_tokens` onto `prompt_tokens`
    /// before cost lookup. A prompt that exceeds a tier threshold only once
    /// the cache-read fold is applied must trigger the tier.
    #[test]
    fn tier_threshold_fires_on_normalized_prompt_with_cache_read() {
        let p = lookup_pricing("tiered-test-model").expect("test shard loaded");
        assert!(
            p.tiers.iter().any(|t| t.threshold_tokens == 100_000),
            "test shard must declare a 100k tier"
        );
        // prompt_tokens = 150k is the NORMALIZED count: 100k fresh input +
        // 50k folded cache-read. The fold crosses the 100k tier threshold.
        let prompt = 150_000u64;
        let cache_read = 50_000u64;
        let completion = 100u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: Some(cache_read),
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost = calculate_cost("tiered-test-model", &u).expect("cost");
        // The 100k tier applies: fresh input at 2e-6, cached tokens at the
        // (untiered) 1e-7 cache rate, output at the base rate.
        let fresh = prompt - cache_read;
        let expected = fresh as f64 * 2e-6 + cache_read as f64 * 1e-7 + completion as f64 * 5e-6;
        assert!(
            (cost - expected).abs() < 1e-12,
            "tier must fire on the normalized prompt: got {cost}, expected {expected}"
        );
    }

    /// Data gap: claude-opus-5 was missing `cache_creation_input_token_cost`
    /// (catalog rate 6.25e-6), so its cache writes fell back to the input rate.
    /// Both TTL rates must now be present.
    #[test]
    fn claude_opus_5_cache_creation_rates_present() {
        let p = lookup_pricing("claude-opus-5").expect("claude-opus-5 priced");
        assert_eq!(p.cache_creation_input_token_cost, Some(6.25e-6));
        assert_eq!(p.cache_creation_input_token_cost_above_1hr, Some(1e-5));
    }

    /// gemini-2.5-pro publishes a >200k cache-read rate and does not publish
    /// a cache-creation tier. Long-context cache writes must not be billed
    /// at an invented creation rate.
    #[test]
    fn gemini_25_pro_has_cache_read_tier_and_no_cache_creation_tier() {
        let p = lookup_pricing("gemini/gemini-2.5-pro").expect("gemini-2.5-pro priced");
        let tier = p
            .tiers
            .iter()
            .find(|t| t.threshold_tokens == 200_000)
            .expect("200k tier");
        assert_eq!(tier.cache_read_input_token_cost, Some(2.5e-7));
        assert_eq!(tier.cache_creation_input_token_cost, None);
    }

    #[test]
    fn gpt6_and_grok_47_prices() {
        let luna = lookup_pricing("gpt-6-luna").expect("gpt-6-luna priced");
        assert_eq!(luna.input_cost_per_token, 1e-7);
        assert_eq!(luna.output_cost_per_token, 5e-7);
        assert_eq!(luna.max_input_tokens, Some(922_000));
        let sol = lookup_pricing("gpt-6-sol").expect("gpt-6-sol priced");
        assert_eq!(sol.input_cost_per_token, 2e-6);
        assert_eq!(sol.output_cost_per_token, 1e-5);
        let astra = lookup_pricing("gpt-6-astra").expect("gpt-6-astra priced");
        assert_eq!(astra.input_cost_per_token, 1e-5);
        assert_eq!(astra.output_cost_per_token, 5e-5);
        let long = calculate_cost("gpt-6-astra", &usage(300_000, 0)).expect("cost");
        assert_eq!(long, 300_000.0 * 2e-5);
        let grok = lookup_pricing("xai/grok-4.7").expect("grok-4.7 priced");
        assert_eq!(grok.input_cost_per_token, 2e-6);
        assert_eq!(grok.output_cost_per_token, 6e-6);
        assert!(grok.tier_threshold_is_inclusive);
        let at = calculate_cost("xai/grok-4.7", &usage(200_000, 0)).expect("cost");
        assert_eq!(at, 200_000.0 * 4e-6);
    }

    // =================================================================
    // C64: context-length pricing tiers.
    // =================================================================

    /// Coverage gap: with tiers `[200k, 100k]` and a prompt EXACTLY at the
    /// highest threshold, the 200k tier does NOT fire (exclusive `>` on this
    /// non-xAI shard) but 200k > 100k, so the LOWER 100k tier applies its
    /// rates (input 2e-6; output falls back to base 5e-6).
    #[test]
    fn tiered_cost_at_highest_threshold_fires_lower_tier() {
        let u = usage(200_000, 1_000);
        let cost = calculate_cost("tiered-test-model", &u).expect("cost");
        assert_eq!(cost, 200_000.0 * 2e-6 + 1_000.0 * 5e-6);
    }

    /// Coverage gap: the `_above_<N>_tokens` (non-`k`) spelling must fire
    /// end-to-end on a full model row, not just in `parse_threshold_key`.
    #[test]
    fn tiered_nonk_threshold_fires_end_to_end() {
        let p = lookup_pricing("tiered-nonk-model").expect("test shard loaded");
        assert_eq!(p.tiers.len(), 1);
        assert_eq!(p.tiers[0].threshold_tokens, 200_000);
        let below = usage(199_999, 100);
        assert_eq!(
            calculate_cost("tiered-nonk-model", &below).expect("cost"),
            199_999.0 * 1e-6 + 100.0 * 5e-6
        );
        let above = usage(250_000, 100);
        assert_eq!(
            calculate_cost("tiered-nonk-model", &above).expect("cost"),
            250_000.0 * 2e-6 + 100.0 * 1e-5
        );
    }

    /// Coverage gap: a tiered cache-creation rate must bill above the
    /// threshold even when the model declares NO base cache-creation rate
    /// (the gemini-2.5-pro shape): below the threshold cache writes are free
    /// (0.0 on the table path), above it they bill at the tier rate.
    #[test]
    fn gemini_25_pro_tiered_cache_creation_bills_above_threshold() {
        let p = lookup_pricing("gemini/gemini-2.5-pro").expect("gemini-2.5-pro priced");
        assert!(
            p.cache_creation_input_token_cost.is_none(),
            "test premise: no base cache-creation rate"
        );
        let created = 50_000u64;
        let below = UsageSummary {
            prompt_tokens: 149_000,
            completion_tokens: 0,
            total_tokens: 149_000,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: Some(created),
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        // Below the 200k threshold: no tier fires, cache writes bill free.
        let cost_below =
            calculate_cost_with_provider("gemini-2.5-pro", &below, "gemini").expect("cost");
        assert_eq!(cost_below, 149_000.0 * 1.25e-6);

        let above = UsageSummary {
            prompt_tokens: 250_000,
            cache_creation_input_tokens: Some(created),
            ..below.clone()
        };
        // Above the threshold the input tier fires. This model publishes no
        // cache-creation rate, so cache writes stay free.
        let cost_above =
            calculate_cost_with_provider("gemini-2.5-pro", &above, "gemini").expect("cost");
        let expected = 250_000.0 * 2.5e-6;
        assert!(
            (cost_above - expected).abs() < 1e-9,
            "got {cost_above}, expected {expected}"
        );
    }

    /// Coverage gap: audio and reasoning rates are NOT tiered. When a tier
    /// fires on a model declaring both, the audio and reasoning buckets must
    /// still bill at the base rates.
    #[test]
    fn tier_fires_but_audio_and_reasoning_stay_untiered() {
        let mut u = usage(250_000, 500);
        u.audio_input_tokens = Some(100);
        u.reasoning_tokens = Some(200);
        let cost = calculate_cost("tiered-test-model", &u).expect("cost");
        // 250k prompt fires the 200k tier: input falls back to the base rate
        // (1e-6), output is tiered (1e-5), cache_read is tiered (2e-7).
        // Audio (100) bills at the base audio rate and reasoning (200) at the
        // base reasoning rate — neither is tiered.
        let expected = (250_000 - 100) as f64 * 1e-6 + 100.0 * 7e-7 + 300.0 * 1e-5 + 200.0 * 3e-6;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    /// Coverage gap: for a tiered model with mixed usage, the resolved rates
    /// must be identical whether reached via `calculate_cost`,
    /// `calculate_cost_with_provider`, or a direct `rates_for_prompt` call.
    #[test]
    fn tiered_rates_identical_across_entry_points_with_mixed_usage() {
        let u = UsageSummary {
            prompt_tokens: 250_000,
            completion_tokens: 1_000,
            total_tokens: 251_000,
            cache_read_input_tokens: Some(50_000),
            cache_creation_input_tokens: Some(20_000),
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: Some(10_000),
            reasoning_tokens: Some(200),
            citation_tokens: None,
            web_search_requests: None,
        };
        let via_bare = calculate_cost("tiered-test-model", &u).expect("bare");
        let via_provider =
            calculate_cost_with_provider("tiered-test-model", &u, "test_tiers").expect("provider");
        assert_eq!(via_bare, via_provider);
        let pricing = lookup_pricing("tiered-test-model").expect("priced");
        // The direct call must use the same context measure: prompt + cache
        // creation (270k here, still above the 200k tier).
        let rates = pricing.rates_for_prompt(250_000, 20_000);
        let direct = cost_from_rates(&u, rates);
        assert_eq!(
            via_bare, direct,
            "direct rates_for_prompt must match the public cost entry points"
        );
    }

    /// HIGH: claude-sonnet-4-20250514 was missing its >200k tiers, so a
    /// 250k-prompt request billed at base input 3e-6 instead of the tier
    /// 6e-6 — a 2x under-bill on the input bucket.
    #[test]
    fn claude_sonnet_4_tier_bills_above_200k() {
        let p = lookup_pricing("claude-sonnet-4-20250514").expect("priced");
        let tier = p
            .tiers
            .iter()
            .find(|t| t.threshold_tokens == 200_000)
            .expect("200k tier");
        assert_eq!(tier.input_cost_per_token, Some(6e-6));
        let below = usage(150_000, 0);
        assert_eq!(
            calculate_cost("claude-sonnet-4-20250514", &below).expect("cost"),
            150_000.0 * 3e-6
        );
        let above = usage(250_000, 0);
        assert_eq!(
            calculate_cost("claude-sonnet-4-20250514", &above).expect("cost"),
            250_000.0 * 6e-6
        );
    }

    /// HIGH: `azure/gpt-5.6` (tier >272k input 1e-5) and `xai/grok-4.3`
    /// (tier >200k input 2.5e-6) must bill at the tier rates above their
    /// thresholds, not the base rates.
    #[test]
    fn azure_gpt_and_grok_tiers_bill_above_threshold() {
        let azure = calculate_cost("azure/gpt-5.6", &usage(300_000, 0)).expect("cost");
        assert_eq!(azure, 300_000.0 * 8e-6);
        let grok = calculate_cost("xai/grok-4.3", &usage(250_000, 0)).expect("cost");
        assert_eq!(grok, 250_000.0 * 2.5e-6);
        // Below the thresholds the base rates apply.
        let azure_below = calculate_cost("azure/gpt-5.6", &usage(200_000, 0)).expect("cost");
        assert_eq!(azure_below, 200_000.0 * 4e-6);
        let grok_below = calculate_cost("xai/grok-4.3", &usage(150_000, 0)).expect("cost");
        assert_eq!(grok_below, 150_000.0 * 1.25e-6);
    }

    /// OpenAI's gpt-5.6-luna / terra list-price cut. Base rates must match
    /// the new list (luna $0.20/$1.20 per 1M, terra $2/$12 per 1M). Mixing
    /// old base rates with the new >272k keys inverted luna so long-context
    /// was cheaper than short-context.
    #[test]
    fn gpt_5_6_luna_and_terra_use_post_cut_base_rates() {
        let luna = lookup_pricing("gpt-5.6-luna").expect("luna priced");
        assert_eq!(luna.input_cost_per_token, 2e-7);
        assert_eq!(luna.output_cost_per_token, 1.2e-6);
        let terra = lookup_pricing("gpt-5.6-terra").expect("terra priced");
        assert_eq!(terra.input_cost_per_token, 2e-6);
        assert_eq!(terra.output_cost_per_token, 1.2e-5);
        // Short-context luna must bill at the new base, not the leftover 272k
        // key (4e-7) which would still overcharge vs the cut.
        let luna_cost = calculate_cost("gpt-5.6-luna", &usage(10_000, 0)).expect("cost");
        assert_eq!(luna_cost, 10_000.0 * 2e-7);
        let luna_long = calculate_cost("gpt-5.6-luna", &usage(300_000, 0)).expect("cost");
        assert_eq!(luna_long, 300_000.0 * 4e-7);
        assert!(
            luna_long / 300_000.0 > luna_cost / 10_000.0,
            "long-context luna rate must be strictly above the short-context rate"
        );
    }

    /// Provider-aware Azure cost lookup must resolve `azure/<model>` when the
    /// caller passes `provider = "azure"`. `azure/gpt-4o-mini` now lists the
    /// same $0.15/$0.60 per 1M as OpenAI, so the rate alone no longer proves
    /// which row was chosen. The EU deployment is still a distinct row, and a
    /// bare fallback would bill the global $2.50 rate instead of $2.75.
    #[test]
    fn azure_provider_lookup_returns_azure_rate_not_openai() {
        let azure = lookup_pricing_with_provider("gpt-4o-mini", "azure").expect("azure priced");
        assert_eq!(
            azure.provider, "openai",
            "azure/* rows live in the openai shard; provider attribution is the shard slug"
        );
        assert!(
            (azure.input_cost_per_token - 1.5e-7).abs() < 1e-15,
            "azure/gpt-4o-mini input rate must be 1.5e-7, got {}",
            azure.input_cost_per_token
        );
        assert!(
            (azure.output_cost_per_token - 6e-7).abs() < 1e-15,
            "azure/gpt-4o-mini output rate must be 6e-7, got {}",
            azure.output_cost_per_token
        );
        let openai = lookup_pricing_with_provider("gpt-4o-mini", "openai").expect("openai priced");
        assert!(
            (openai.input_cost_per_token - 1.5e-7).abs() < 1e-15,
            "openai gpt-4o-mini input rate must be 1.5e-7"
        );
        let eu =
            lookup_pricing_with_provider("eu/gpt-4o-2024-08-06", "azure").expect("azure eu priced");
        let global = lookup_pricing("gpt-4o-2024-08-06").expect("global priced");
        assert!(
            (eu.input_cost_per_token - 2.75e-6).abs() < 1e-15,
            "azure/eu/gpt-4o-2024-08-06 input rate must be 2.75e-6"
        );
        assert!(
            (global.input_cost_per_token - 2.5e-6).abs() < 1e-15,
            "bare gpt-4o-2024-08-06 input rate must be 2.5e-6"
        );
        assert_ne!(eu.input_cost_per_token, global.input_cost_per_token);
    }

    /// MEDIUM: Azure-only deployment ids (no bare OpenAI equivalent) must
    /// resolve to `Some` — not unmetered `None` — via the provider-aware
    /// lookup. `azure/eu/gpt-4o-2024-08-06` and `azure/ada` are Azure-specific
    /// rows that have no bare equivalent in the price table.
    #[test]
    fn azure_only_deployment_ids_resolve_via_provider_aware_lookup() {
        let eu = lookup_pricing_with_provider("eu/gpt-4o-2024-08-06", "azure")
            .expect("azure/eu/gpt-4o-2024-08-06 must be priced");
        assert!(
            (eu.input_cost_per_token - 2.75e-6).abs() < 1e-15,
            "azure/eu/gpt-4o-2024-08-06 input rate must be 2.75e-6"
        );
        let ada = lookup_pricing("azure/ada").expect("azure/ada must be priced");
        assert!(
            (ada.input_cost_per_token - 1.1e-7).abs() < 1e-15,
            "azure/ada input rate must be 1.1e-7 (2026-10 Azure pricing sync)"
        );
    }

    /// LOW: the tier threshold must count Anthropic's additive cache-creation
    /// tokens, matching the reference, whose Anthropic transformation folds BOTH
    /// cache-creation and cache-read onto `prompt_tokens` before the cost
    /// lookup. A request whose context crosses 200k only once cache writes are
    /// counted must fire claude-sonnet-4's >200k tier.
    #[test]
    fn tier_fires_on_context_including_cache_creation() {
        let u = UsageSummary {
            prompt_tokens: 180_000,
            completion_tokens: 100,
            total_tokens: 180_100,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: Some(30_000),
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost = calculate_cost("claude-sonnet-4-20250514", &u).expect("cost");
        // Context = 180k + 30k = 210k > 200k: the tier fires. Fresh input at
        // the tier rate (6e-6), cache writes at the tier cache-creation rate
        // (7.5e-6), output at the tier rate (2.25e-5).
        let expected = 180_000.0 * 6e-6 + 30_000.0 * 7.5e-6 + 100.0 * 2.25e-5;
        assert!(
            (cost - expected).abs() < 1e-9,
            "got {cost}, expected {expected}"
        );
        // The pre-fix behaviour (threshold on prompt only, 180k < 200k) would
        // have stayed on base rates — a materially lower bill.
        let base_billed = 180_000.0 * 3e-6 + 30_000.0 * 3.75e-6 + 100.0 * 1.5e-5;
        assert!(
            (cost - base_billed).abs() > 1e-6,
            "cache-creation tokens must count toward the tier threshold"
        );
    }

    /// LOW: a provider reporting ONLY the 1h-ephemeral detail (no aggregate
    /// `cache_creation_input_tokens`) bills those tokens (the 1h detail is
    /// the effective total in `cost_from_rates`), so they must also count
    /// toward the context-length tier threshold — the same effective total,
    /// not `aggregate.unwrap_or(0)`. A synthetic/malformed usage of this
    /// shape must not slip under the tier.
    #[test]
    fn tier_threshold_counts_1h_only_cache_creation_detail() {
        let prompt = 150_000u64;
        let created_1h = 60_000u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: prompt,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: Some(created_1h),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        // Context = 150k + 60k = 210k ≥ 200k: the tiered-1h-model's 200k tier
        // fires and the 1h bucket bills at the tier rate (4e-6), not the
        // base 1h rate (2e-6).
        let cost = calculate_cost("tiered-1h-model", &u).expect("cost");
        let expected = prompt as f64 * 1e-6 + created_1h as f64 * 4e-6;
        assert!(
            (cost - expected).abs() < 1e-9,
            "1h-only cache creation must cross the tier threshold: got {cost}, expected {expected}"
        );
        // Sanity: the pre-fix threshold measure (aggregate only, 150k < 200k)
        // kept the base 1h rate — a strictly lower bill.
        let base_billed = prompt as f64 * 1e-6 + created_1h as f64 * 2e-6;
        assert!(
            (cost - base_billed).abs() > 1e-9,
            "the tier must actually fire for the 1h-only detail"
        );
    }

    /// LOW (C64 round 2): when a provider reports BOTH cache-creation fields
    /// inconsistently (aggregate 300 < 1h detail 1000), the billed effective
    /// total is the aggregate (300) — `cost_from_rates` clamps the 1h detail
    /// to it — so the tier threshold measure must also be 300, not
    /// `max(aggregate, 1h)`. The old `.max()` measure let the 1h detail push
    /// a request whose billed context stays below the threshold into the
    /// tier (over-bill on inconsistent payloads); the reference folds the raw
    /// aggregate only.
    #[test]
    fn tier_threshold_measure_matches_billed_cache_creation_total() {
        let prompt = 199_699u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: prompt,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: Some(300),
            cache_creation_1h_input_tokens: Some(1_000),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        // Measure = 199_699 + 300 = 199_999 < 200_000: no tier fires; the
        // 1h bucket is clamped to the aggregate (300) and bills at the base
        // 1h rate (6e-6).
        let cost = calculate_cost("claude-sonnet-4-20250514", &u).expect("cost");
        let expected = prompt as f64 * 3e-6 + 300.0 * 6e-6;
        assert!(
            (cost - expected).abs() < 1e-9,
            "the threshold measure must be the billed effective total (aggregate), \
             not max(aggregate, 1h): got {cost}, expected {expected}"
        );
        // Sanity: the pre-fix `.max()` measure (199_699 + 1_000 = 200_699)
        // fired the >200k tier and billed the input bucket at 6e-6 — a
        // strictly higher bill.
        let over_tiered = prompt as f64 * 6e-6 + 300.0 * 6e-6;
        assert!(
            (over_tiered - cost).abs() > 1e-6,
            "the inconsistent 1h detail must not push the request into the tier"
        );
    }

    /// Keep context-tier rates covered by the pinned catalog snapshot so
    /// long requests cannot silently fall back to base prices.
    #[test]
    fn context_tier_snapshot_covers_shipped_rates() {
        let snapshot: HashMap<String, HashMap<String, f64>> =
            serde_json::from_str(include_str!("../data/prices/context_tiers_snapshot.json"))
                .expect("pinned context-tier snapshot must parse");
        let shipped = price_table();
        let mut checked = 0u32;
        for (model, expected_tiers) in &snapshot {
            let Some(pricing) = shipped.get(model) else {
                continue;
            };
            for (key, expected) in expected_tiers {
                let Some((base, threshold)) = parse_threshold_key(key) else {
                    continue;
                };
                let tier = pricing
                    .tiers
                    .iter()
                    .find(|t| t.threshold_tokens == threshold)
                    .unwrap_or_else(|| panic!("{model} is missing context-tier key '{key}'"));
                let actual = match base {
                    "input_cost_per_token" => tier.input_cost_per_token,
                    "output_cost_per_token" => tier.output_cost_per_token,
                    "cache_read_input_token_cost" => tier.cache_read_input_token_cost,
                    "cache_creation_input_token_cost" => tier.cache_creation_input_token_cost,
                    "cache_creation_input_token_cost_above_1hr" => {
                        tier.cache_creation_input_token_cost_above_1hr
                    }
                    _ => panic!("snapshot key '{key}' has an unmodelable base field"),
                };
                assert_eq!(
                    actual,
                    Some(*expected),
                    "{model}: tier rate for '{key}' must match the pinned catalog value"
                );
                checked += 1;
            }
        }
        assert!(
            checked > 250,
            "expected broad context-tier coverage, got {checked}"
        );
    }

    /// MEDIUM: a tier rate written as a TOML integer (no decimal point) must
    /// parse into the tier instead of silently dropping it. Base row fields
    /// coerce integers through serde; the tier path must match.
    #[test]
    fn tier_parsing_accepts_integer_rates() {
        let raw: toml::Table = r#"
input_cost_per_token = 1e-6
input_cost_per_token_above_100k_tokens = 2000000
"#
        .parse()
        .expect("parse");
        let tiers = extract_cost_tiers(&toml::Value::Table(raw));
        assert_eq!(
            tiers.len(),
            1,
            "an integer tier rate must not drop the whole tier"
        );
        assert_eq!(tiers[0].threshold_tokens, 100_000);
        assert_eq!(tiers[0].input_cost_per_token, Some(2_000_000.0));
    }

    /// MEDIUM: a non-numeric (string) tier value must drop only that key —
    /// never crash and never poison the sibling tier rates. Numeric strings
    /// (e.g. `"1e-5"`) are now parsed via `tier_rate_value` for reference
    /// parity (`_get_cost_per_unit` coerces strings via `float(...)`); this
    /// test therefore uses a GENUINELY non-numeric string to pin the skip
    /// path. See `tier_parsing_accepts_string_rate` for the numeric-string
    /// acceptance path.
    #[test]
    fn tier_parsing_skips_non_numeric_value_without_poisoning() {
        let raw: toml::Table = r#"
input_cost_per_token = 1e-6
input_cost_per_token_above_100k_tokens = 2e-6
output_cost_per_token_above_200k_tokens = "not-a-number"
cache_read_input_token_cost_above_200k_tokens = 2e-7
"#
        .parse()
        .expect("parse");
        let tiers = extract_cost_tiers(&toml::Value::Table(raw));
        assert_eq!(tiers.len(), 2);
        let two_hundred = &tiers[0];
        assert_eq!(two_hundred.threshold_tokens, 200_000);
        assert_eq!(
            two_hundred.output_cost_per_token, None,
            "non-numeric string tier key must be skipped"
        );
        assert_eq!(two_hundred.cache_read_input_token_cost, Some(2e-7));
        assert_eq!(tiers[1].input_cost_per_token, Some(2e-6));
    }

    /// LOW parity: a tier rate written as a TOML numeric string (`"1e-5"`,
    /// `"2.5e-7"`, `"2000000"`) must parse into the tier instead of being
    /// silently dropped — matching the reference `_get_cost_per_unit`, which
    /// coerces strings via `float(...)`. Mirrors the integer-rate test
    /// (`tier_parsing_accepts_integer_rates`): without this, a rate vendored
    /// from the reference JSON (which sometimes carries string rates) would drop
    /// the whole tier and silently bill long-context requests at the base
    /// rate.
    #[test]
    fn tier_parsing_accepts_string_rate() {
        let raw: toml::Table = r#"
input_cost_per_token = 1e-6
input_cost_per_token_above_100k_tokens = "2e-6"
output_cost_per_token_above_200k_tokens = "1e-5"
cache_read_input_token_cost_above_200k_tokens = "2.5e-7"
"#
        .parse()
        .expect("parse");
        let tiers = extract_cost_tiers(&toml::Value::Table(raw));
        assert_eq!(
            tiers.len(),
            2,
            "numeric-string tier rates must not drop the whole tier"
        );
        assert_eq!(tiers[0].threshold_tokens, 200_000);
        assert_eq!(tiers[0].output_cost_per_token, Some(1e-5));
        assert_eq!(tiers[0].cache_read_input_token_cost, Some(2.5e-7));
        assert_eq!(tiers[1].threshold_tokens, 100_000);
        assert_eq!(tiers[1].input_cost_per_token, Some(2e-6));

        // An integer-valued string must round-trip identically to the
        // integer-typed value (mirrors `tier_parsing_accepts_integer_rates`).
        let raw_int_str: toml::Table = r#"
input_cost_per_token = 1e-6
input_cost_per_token_above_100k_tokens = "2000000"
"#
        .parse()
        .expect("parse");
        let tiers = extract_cost_tiers(&toml::Value::Table(raw_int_str));
        assert_eq!(tiers.len(), 1);
        assert_eq!(tiers[0].input_cost_per_token, Some(2_000_000.0));
    }

    /// MEDIUM: a tiered 1h cache-creation rate
    /// (`cache_creation_input_token_cost_above_1hr_above_<N>k_tokens`) must
    /// parse into the tier and bill 1h tokens at the tiered rate above the
    /// threshold and the base 1h rate below it — never silently dropped.
    #[test]
    fn tiered_1h_cache_creation_rate_parses_and_bills() {
        let p = lookup_pricing("tiered-1h-model").expect("test shard loaded");
        let tier = p
            .tiers
            .iter()
            .find(|t| t.threshold_tokens == 200_000)
            .expect("200k tier");
        assert_eq!(
            tier.cache_creation_input_token_cost_above_1hr,
            Some(4e-6),
            "the combined 1h tier key must parse into the tier"
        );

        let created = 100u64;
        let created_1h = 60u64;
        // Below the threshold: 1h tokens bill at the base 1h rate (2e-6).
        let below = UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 0,
            total_tokens: 100,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: Some(created),
            cache_creation_1h_input_tokens: Some(created_1h),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost_below = calculate_cost("tiered-1h-model", &below).expect("cost");
        let expected_below =
            100.0 * 1e-6 + (created - created_1h) as f64 * 1e-6 + created_1h as f64 * 2e-6;
        assert!(
            (cost_below - expected_below).abs() < 1e-12,
            "got {cost_below}, expected {expected_below}"
        );

        // Above the threshold: the tier fires and the 1h bucket bills at the
        // tiered rate (4e-6); the other buckets fall back to base.
        let above = UsageSummary {
            prompt_tokens: 250_000,
            ..below.clone()
        };
        let cost_above = calculate_cost("tiered-1h-model", &above).expect("cost");
        let expected_above =
            250_000.0 * 1e-6 + (created - created_1h) as f64 * 1e-6 + created_1h as f64 * 4e-6;
        assert!(
            (cost_above - expected_above).abs() < 1e-12,
            "got {cost_above}, expected {expected_above}"
        );
        assert!(
            cost_above > cost_below,
            "tiered 1h rate must exceed the base 1h rate above the threshold"
        );
    }

    /// Data gap: meta/muse-spark-1.1 (both bare and prefixed keys) was missing
    /// `cache_read_input_token_cost` (catalog rate 1.5e-7).
    #[test]
    fn muse_spark_11_cache_read_rate_present() {
        for key in ["muse-spark-1.1", "meta/muse-spark-1.1"] {
            let p = lookup_pricing(key).unwrap_or_else(|| panic!("{key} priced"));
            assert_eq!(p.cache_read_input_token_cost, Some(1.5e-7));
        }
    }

    /// The catalog publishes `max_output_tokens: 131072` for meta/muse-spark-1.1;
    /// both the bare and prefixed price rows must carry it so the pre-call
    /// context-window checker can reject an over-budget `max_tokens`.
    #[test]
    fn muse_spark_11_max_output_tokens_present() {
        for key in ["muse-spark-1.1", "meta/muse-spark-1.1"] {
            let p = lookup_pricing(key).unwrap_or_else(|| panic!("{key} priced"));
            assert_eq!(p.max_output_tokens, Some(131_072));
        }
    }

    /// Meta supports prompt caching: cached prompt tokens must bill at the
    /// discounted `cache_read_input_token_cost` rate and the remaining prompt
    /// tokens at the regular input rate, with no double counting. Reasoning
    /// tokens bill at the output (== reasoning) rate as a clamped subset of
    /// completion tokens.
    #[test]
    fn muse_spark_11_cache_read_tokens_billed_at_discounted_rate() {
        let pricing = lookup_pricing("muse-spark-1.1").expect("priced");
        let cache_rate = pricing
            .cache_read_input_token_cost
            .expect("muse-spark-1.1 has cache-read pricing");
        let prompt = 2_000u64;
        let cached = 800u64;
        let completion = 100u64;
        let reasoning = 100u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: Some(cached),
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: Some(reasoning),
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost("muse-spark-1.1", &u).expect("priced");
        let regular_input = (prompt - cached) as f64 * pricing.input_cost_per_token;
        let cached_cost = cached as f64 * cache_rate;
        let output = completion as f64 * pricing.output_cost_per_token;
        let expected = regular_input + cached_cost + output;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
        assert!(
            cache_rate < pricing.input_cost_per_token,
            "cache-read rate must be cheaper than the regular input rate"
        );
    }

    #[test]
    fn claude_pricing_includes_cache_creation_cost() {
        let p = lookup_pricing("claude-3-opus-20240229").expect("found");
        assert!(p.cache_creation_input_token_cost.is_some());
        assert!(p.cache_read_input_token_cost.is_some());
    }

    #[test]
    fn cache_read_tokens_priced_at_cache_rate() {
        let pricing = lookup_pricing("gpt-4o-mini").expect("priced");
        let cache_rate = pricing
            .cache_read_input_token_cost
            .expect("gpt-4o-mini has cache-read pricing");
        let prompt = 1_000u64;
        let cached = 400u64;
        let completion = 200u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: Some(cached),
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost("gpt-4o-mini", &u).expect("priced");
        let regular_input = (prompt - cached) as f64 * pricing.input_cost_per_token;
        let cached_cost = cached as f64 * cache_rate;
        let output = completion as f64 * pricing.output_cost_per_token;
        let expected = regular_input + cached_cost + output;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    #[test]
    fn cache_read_tokens_differ_from_regular_cost() {
        let pricing = lookup_pricing("gpt-4o-mini").expect("priced");
        let cache_rate = pricing
            .cache_read_input_token_cost
            .expect("cache-read pricing");
        let input_rate = pricing.input_cost_per_token;
        assert!(
            (cache_rate - input_rate).abs() > 1e-15,
            "cache-read rate must differ from regular rate for this test"
        );
        let prompt = 1_000u64;
        let completion = 0u64;
        let without_cache = usage(prompt, completion);
        let with_cache = UsageSummary {
            cache_read_input_tokens: Some(prompt),
            ..without_cache.clone()
        };
        let cost_plain = calculate_cost("gpt-4o-mini", &without_cache).expect("priced");
        let cost_cached = calculate_cost("gpt-4o-mini", &with_cache).expect("priced");
        assert!(
            (cost_plain - cost_cached).abs() > 1e-9,
            "cache-read pricing must change the total: plain={cost_plain}, cached={cost_cached}"
        );
        let expected_cached = prompt as f64 * cache_rate;
        assert!((cost_cached - expected_cached).abs() < 1e-12);
    }

    #[test]
    fn cache_creation_tokens_additive() {
        let pricing = lookup_pricing("claude-sonnet-4-20250514").expect("priced");
        let creation_rate = pricing
            .cache_creation_input_token_cost
            .expect("cache-creation pricing");
        let prompt = 500u64;
        let creation = 300u64;
        let completion = 100u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: Some(creation),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost("claude-sonnet-4-20250514", &u).expect("priced");
        let regular_input = prompt as f64 * pricing.input_cost_per_token;
        let creation_cost = creation as f64 * creation_rate;
        let output = completion as f64 * pricing.output_cost_per_token;
        let expected = regular_input + creation_cost + output;
        assert!(
            (cost - expected).abs() < 1e-9,
            "got {cost}, expected {expected}"
        );
    }

    #[test]
    fn no_optional_fields_matches_legacy_formula() {
        let u = usage(1_000_000, 500_000);
        let cost = calculate_cost("gpt-4o", &u).expect("priced");
        let pricing = lookup_pricing("gpt-4o").expect("priced");
        let expected =
            1_000_000.0 * pricing.input_cost_per_token + 500_000.0 * pricing.output_cost_per_token;
        assert!((cost - expected).abs() < 1e-12);
    }

    /// Bug #2: Anthropic reports `cache_read_input_tokens` as a top-level
    /// field ADDITIVE to `input_tokens`. The extraction layer folds the
    /// cache-read tokens into `prompt_tokens` so the `UsageSummary` obeys
    /// the "cache_read ⊆ prompt_tokens" invariant. This test pins the
    /// post-normalisation cost split: the 1_000_000 cache hits bill at the
    /// cache rate, and the 100 fresh input tokens bill at the input rate —
    /// not zero.
    ///
    /// The prompt (1_000_100 normalized tokens) crosses claude-sonnet-4's
    /// 200k context tier, so the tier rates apply; the expected cost uses
    /// the resolved [`TokenRates`] so the test stays correct if the model
    /// is repriced or re-tiered.
    #[test]
    fn anthropic_cache_read_not_double_subtracted() {
        let pricing =
            lookup_pricing("claude-sonnet-4-20250514").expect("claude-sonnet-4-20250514 priced");
        let cache_read = 1_000_000u64;
        let fresh_input = 100u64;
        // Post-normalisation form: prompt_tokens INCLUDES cache_read.
        let u = UsageSummary {
            prompt_tokens: fresh_input + cache_read,
            completion_tokens: 50,
            total_tokens: fresh_input + cache_read + 50,
            cache_read_input_tokens: Some(cache_read),
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost("claude-sonnet-4-20250514", &u).expect("priced");

        let rates = pricing.rates_for_prompt(fresh_input + cache_read, 0);
        let expected = fresh_input as f64 * rates.input
            + cache_read as f64 * rates.cache_read
            + 50.0 * rates.output;
        assert!(
            (cost - expected).abs() < 1e-9,
            "got {cost}, expected {expected}"
        );

        // The double-subtraction bug would zero out the fresh input tokens.
        let buggy = cache_read as f64 * rates.cache_read + 50.0 * rates.output;
        assert!(
            (cost - buggy).abs() > 1e-6,
            "fresh input tokens must still be billed: got {cost}, buggy {buggy}"
        );
    }

    /// Bug #3: Anthropic snapshot ids use a compact `-YYYYMMDD` suffix with
    /// no inner dashes; the stripper must reduce them to the base model.
    #[test]
    fn dated_snapshot_suffix_8_digit_stripped() {
        // gpt-4o-2024-08-06 is a real catalog key; the stripper reduces it
        // to the base name when the dated variant is not in the table.
        let bare = lookup_pricing("gpt-4o").expect("base model priced");
        let dated = lookup_pricing("gpt-4o-2024-08-06").expect("dated snapshot priced");
        assert_eq!(
            bare.input_cost_per_token, dated.input_cost_per_token,
            "dated snapshot suffix must resolve to base pricing"
        );
        let cost = calculate_cost("gpt-4o-2024-08-06", &usage(10, 5)).expect("cost");
        assert!(cost > 0.0);
    }

    /// Bug #21: OpenAI's `audio_tokens` is a subset of `prompt_tokens`, so
    /// audio tokens must be removed from the regular input bucket — not
    /// billed at both the audio rate and the text input rate.
    #[test]
    fn audio_tokens_not_double_charged() {
        let pricing = lookup_pricing("gpt-4o").expect("gpt-4o priced");
        let prompt = 1_000u64;
        let audio = 200u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: prompt,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            audio_input_tokens: Some(audio),
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost("gpt-4o", &u).expect("priced");

        // gpt-4o declares no `input_cost_per_audio_token`, so on the
        // price-table path the audio bucket bills at 0.0 (the reference
        // `_get_cost_per_unit(..., default=0.0)`) — audio tokens are FREE,
        // they are only removed from the regular input bucket so the total
        // input-charged tokens equal `prompt - audio`.
        assert!(
            pricing.input_cost_per_audio_token.is_none(),
            "this test pins the missing-audio-rate semantic; gpt-4o must have no audio rate"
        );
        let expected = (prompt - audio) as f64 * pricing.input_cost_per_token;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );

        // Sanity: the pre-fix fallback (audio billed at the text input rate,
        // i.e. the double-charge-era formula `prompt * input`) is strictly
        // larger — audio tokens must not re-enter the bill at the input rate.
        let fallback = prompt as f64 * pricing.input_cost_per_token
            + audio as f64 * pricing.input_cost_per_token;
        assert!(
            (fallback - cost).abs() > 1e-9,
            "audio tokens must bill free on the table path: got {cost}, fallback {fallback}"
        );
    }

    #[test]
    fn override_uses_override_rates_for_known_model() {
        // gpt-4o is in the price table; the override should win.
        let pricing = lookup_pricing("gpt-4o").expect("gpt-4o priced");
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.002,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            ..Default::default()
        };
        let u = usage(1_000, 500);
        let cost = calculate_cost_with_override("gpt-4o", &u, Some(&dp))
            .expect("override path returns Some");
        let expected = 1_000.0 * 0.001 + 500.0 * 0.002;
        assert!(
            (cost - expected).abs() < 1e-12,
            "override should win over price table: got {cost}, expected {expected}"
        );
        // Sanity: the price table would have given a different answer.
        let table_cost = calculate_cost("gpt-4o", &u).expect("priced");
        assert!(
            (table_cost - expected).abs() > 1e-9,
            "price-table cost ({table_cost}) should differ from override ({expected})"
        );
        let _ = pricing;
    }

    #[test]
    fn override_falls_back_to_price_table_when_none() {
        let u = usage(1_000_000, 500_000);
        let cost_override_none =
            calculate_cost_with_override("gpt-4o", &u, None).expect("priced via fallback");
        let cost_table = calculate_cost("gpt-4o", &u).expect("priced");
        assert!(
            (cost_override_none - cost_table).abs() < 1e-12,
            "None override should fall back to price table exactly"
        );
    }

    #[test]
    fn override_priced_for_unknown_model_returns_some() {
        // Unknown model would be `None` via the price table, but the override
        // should still produce a cost.
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.002,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            ..Default::default()
        };
        let u = usage(1_000, 500);
        let cost = calculate_cost_with_override("totally-unknown-model-xyz", &u, Some(&dp))
            .expect("override applies even for unknown model");
        let expected = 1_000.0 * 0.001 + 500.0 * 0.002;
        assert!((cost - expected).abs() < 1e-12);
    }

    #[test]
    fn override_cache_rates_fall_back_to_input_rate() {
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.002,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            ..Default::default()
        };
        let prompt = 1_000u64;
        let cached = 400u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: prompt,
            cache_read_input_tokens: Some(cached),
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost_with_override("any-model", &u, Some(&dp)).expect("override path");
        // All tokens (regular + cached) billed at the input rate when the
        // cache_read rate is None.
        let expected = prompt as f64 * 0.001;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    #[test]
    fn override_cache_rates_applied_when_set() {
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.002,
            cache_read_input_token_cost: Some(0.0001),
            cache_creation_input_token_cost: Some(0.0005),
            ..Default::default()
        };
        let prompt = 1_000u64;
        let cached = 400u64;
        let created = 100u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: 50,
            total_tokens: prompt + 50,
            cache_read_input_tokens: Some(cached),
            cache_creation_input_tokens: Some(created),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost_with_override("any-model", &u, Some(&dp)).expect("override path");
        let expected = (prompt - cached) as f64 * 0.001
            + cached as f64 * 0.0001
            + created as f64 * 0.0005
            + 50.0 * 0.002;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    // =================================================================
    // H3: reasoning-token-aware cost calculation.
    // =================================================================

    /// When the deployment override exposes a separate
    /// `output_cost_per_reasoning_token`, reasoning tokens must bill at
    /// that rate and the remaining `completion - reasoning` text tokens at
    /// the regular output rate.
    #[test]
    fn override_reasoning_tokens_use_separate_rate_when_available() {
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.01,
            output_cost_per_reasoning_token: Some(0.0001),
            ..Default::default()
        };
        let prompt = 1_000u64;
        let completion = 500u64;
        let reasoning = 200u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: Some(reasoning),
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost_with_override("any-model", &u, Some(&dp)).expect("priced");
        let expected = prompt as f64 * 0.001
            + (completion - reasoning) as f64 * 0.01
            + reasoning as f64 * 0.0001;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    /// When no separate reasoning rate exists, reasoning tokens bill at the
    /// regular output rate, preserving the legacy formula
    /// (`completion * output_rate`).
    #[test]
    fn override_reasoning_tokens_fall_back_to_output_rate_when_no_reasoning_rate() {
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.01,
            ..Default::default()
        };
        let prompt = 1_000u64;
        let completion = 500u64;
        let reasoning = 200u64;
        let u_with_reasoning = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: Some(reasoning),
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let u_without = UsageSummary {
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            ..u_with_reasoning.clone()
        };
        let cost_with = calculate_cost_with_override("any-model", &u_with_reasoning, Some(&dp))
            .expect("priced");
        let cost_without =
            calculate_cost_with_override("any-model", &u_without, Some(&dp)).expect("priced");
        // Without a reasoning rate, both forms reduce to the legacy formula.
        let legacy = prompt as f64 * 0.001 + completion as f64 * 0.01;
        assert!((cost_with - legacy).abs() < 1e-12);
        assert!((cost_without - legacy).abs() < 1e-12);
    }

    /// text_tokens = completion_tokens - reasoning_tokens, billed at the
    /// regular output rate; reasoning at the reasoning rate. Pins the split
    /// explicitly.
    #[test]
    fn override_text_tokens_are_completion_minus_reasoning_at_output_rate() {
        let dp = DeploymentPricing {
            input_cost_per_token: 0.0,
            output_cost_per_token: 0.02,
            output_cost_per_reasoning_token: Some(0.005),
            ..Default::default()
        };
        let completion = 1_000u64;
        let reasoning = 300u64;
        let u = UsageSummary {
            prompt_tokens: 0,
            completion_tokens: completion,
            total_tokens: completion,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: Some(reasoning),
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost_with_override("any-model", &u, Some(&dp)).expect("priced");
        let text_tokens = completion - reasoning;
        let expected = text_tokens as f64 * 0.02 + reasoning as f64 * 0.005;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    /// Inconsistent provider data: reasoning_tokens > completion_tokens.
    /// Must not underflow; clamp reasoning to completion (text bucket = 0).
    #[test]
    fn override_reasoning_exceeding_completion_does_not_underflow() {
        let dp = DeploymentPricing {
            input_cost_per_token: 0.0,
            output_cost_per_token: 0.01,
            output_cost_per_reasoning_token: Some(0.001),
            ..Default::default()
        };
        let u = UsageSummary {
            prompt_tokens: 0,
            completion_tokens: 100,
            total_tokens: 100,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: Some(1_000),
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost = calculate_cost_with_override("any-model", &u, Some(&dp)).expect("priced");
        assert!(cost >= 0.0, "cost must not go negative: {cost}");
        // reasoning clamped to 100, text_tokens clamped to 0.
        let expected = 100_f64 * 0.001;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    /// Negative or NaN operator-configured rates must not produce a negative
    /// or non-finite cost (which would corrupt the spend counter and the
    /// reservation/settle math). They collapse to 0.0.
    #[test]
    fn negative_or_nan_rates_collapse_to_zero_cost() {
        let u = usage(1_000, 500);
        // Negative input/output → zero cost, never negative.
        let dp_neg = DeploymentPricing {
            input_cost_per_token: -2.5e-6,
            output_cost_per_token: -1.0e-5,
            ..Default::default()
        };
        let cost_neg =
            calculate_cost_with_override("any-model", &u, Some(&dp_neg)).expect("priced");
        assert!(
            cost_neg.abs() < f64::EPSILON,
            "negative rates must yield zero cost, got {cost_neg}"
        );

        // NaN rates → zero cost, never NaN.
        let dp_nan = DeploymentPricing {
            input_cost_per_token: f64::NAN,
            output_cost_per_token: f64::NAN,
            ..Default::default()
        };
        let cost_nan =
            calculate_cost_with_override("any-model", &u, Some(&dp_nan)).expect("priced");
        assert!(
            !cost_nan.is_nan() && cost_nan.abs() < f64::EPSILON,
            "NaN rates must yield zero cost, got {cost_nan}"
        );

        // A negative CACHE rate on an otherwise-positive override must also be
        // neutralised (cache tokens billed at 0, not a negative rate).
        let dp_cache = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.0,
            cache_read_input_token_cost: Some(-0.5),
            ..Default::default()
        };
        let u_cache = UsageSummary {
            prompt_tokens: 1_000,
            completion_tokens: 0,
            total_tokens: 1_000,
            cache_read_input_tokens: Some(400),
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        let cost_cache =
            calculate_cost_with_override("any-model", &u_cache, Some(&dp_cache)).expect("priced");
        // 600 regular at 0.001 + 400 cached at 0 (negative clamped) = 0.6
        assert!(
            (cost_cache - 0.6).abs() < 1e-12,
            "negative cache rate must collapse to 0, got {cost_cache}"
        );
    }

    // =================================================================
    // C01 review follow-ups.
    // =================================================================

    /// MEDIUM: a per-second-priced model is un-metered on the token-based
    /// cost path. `calculate_cost` returns `None` for it (indistinguishable
    /// from "model not in the price table" today), and the only signal is a
    /// `tracing::warn!`. The decision must be OBSERVABLE: callers can detect
    /// per-second pricing via [`ModelPricing::is_per_second_priced`] and bill
    /// through [`ModelPricing::cost_per_second`], which mirrors the reference
    /// `input_cost_per_second * response_time_ms / 1000`.
    #[test]
    fn per_second_model_is_unmetered_on_token_path_but_billable_via_helper() {
        let pricing = ModelPricing {
            input_cost_per_token: 0.0,
            output_cost_per_token: 0.0,
            input_cost_per_audio_token: None,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            output_cost_per_reasoning_token: None,
            citation_cost_per_token: None,
            search_queries_cost_per_query: None,
            max_input_tokens: None,
            max_output_tokens: None,
            provider: "test".to_string(),
            input_cost_per_second: Some(0.0001),
            prompt_cache_min_tokens: None,
            tiers: Vec::new(),
            tier_threshold_is_inclusive: false,
            ..Default::default()
        };

        assert!(pricing.is_per_second_priced());

        // The token-based path skips per-second models (returns None).
        let u = usage(1_000, 500);
        assert!(
            cost_from_pricing(&pricing, &u).is_none(),
            "per-second model must return None on the token-based cost path"
        );

        // ...but the duration-based helper accrues spend.
        // 2500 ms * 0.0001 $/s / 1000 = 0.00025
        let cost = pricing
            .cost_per_second(2500.0)
            .expect("per-second model is billable via the helper");
        assert!(
            (cost - 0.00025).abs() < 1e-15,
            "got {cost}, expected 0.00025"
        );

        // Non-per-second models return None from the helper.
        let token_priced = ModelPricing {
            input_cost_per_second: None,
            ..pricing.clone()
        };
        assert!(!token_priced.is_per_second_priced());
        assert!(token_priced.cost_per_second(2500.0).is_none());

        // Negative / non-finite latency collapses to 0 spend (never negative).
        assert_eq!(pricing.cost_per_second(-100.0).unwrap(), 0.0);
        assert_eq!(pricing.cost_per_second(f64::NAN).unwrap(), 0.0);
        // +∞ latency is also sanitized to 0 spend — the silent-$0 direction,
        // but a corrupt duration measurement must never poison the spend
        // counter with a non-finite cost. Pins the sanitize-always contract.
        assert_eq!(
            pricing.cost_per_second(f64::INFINITY).unwrap(),
            0.0,
            "+inf latency must collapse to 0, not NaN/inf cost"
        );
    }

    /// LOW #3: a hostile/huge token count cast lossily to f64 and multiplied
    /// by a large operator rate can overflow the running cost sum to +∞. The
    /// final cost must be re-sanitised to a finite, non-negative value so the
    /// spend counter / reservation-settle math cannot be corrupted.
    #[test]
    fn huge_token_count_yields_finite_non_negative_cost() {
        let dp = DeploymentPricing {
            input_cost_per_token: 1e300,
            output_cost_per_token: 1e300,
            ..Default::default()
        };
        let u = UsageSummary {
            prompt_tokens: u64::MAX,
            completion_tokens: u64::MAX,
            total_tokens: u64::MAX,
            ..UsageSummary::default()
        };
        let cost = calculate_cost_with_override("any-model", &u, Some(&dp)).expect("priced");
        assert!(
            cost.is_finite(),
            "cost must be finite even for extreme token counts, got {cost}"
        );
        assert!(cost >= 0.0, "cost must never go negative, got {cost}");
    }

    /// LOW/INFO #4: a deployment override on a tiered model bills FLAT —
    /// context-length tiers are ignored, matching the reference
    /// `_cost_per_token_custom_pricing_helper`. Pins the intentional
    /// divergence: a >272k prompt on gpt-5.6 bills at the override rate, not
    /// the model's >272k tier rate.
    #[test]
    fn override_bills_flat_ignoring_context_tiers() {
        let p = lookup_pricing("gpt-5.6").expect("gpt-5.6 priced");
        assert!(
            p.tiers.iter().any(|t| t.threshold_tokens == 272_000),
            "test premise: gpt-5.6 must have a 272k tier"
        );
        let dp = DeploymentPricing {
            input_cost_per_token: 1e-6,
            output_cost_per_token: 1e-5,
            ..Default::default()
        };
        let above_tier = usage(300_000, 100);
        let cost =
            calculate_cost_with_override("gpt-5.6", &above_tier, Some(&dp)).expect("override path");
        // Flat override rate applied to ALL 300k tokens — NOT the tier rate.
        let expected = 300_000.0 * 1e-6 + 100.0 * 1e-5;
        assert!(
            (cost - expected).abs() < 1e-12,
            "override must bill flat ignoring the 272k tier: got {cost}, expected {expected}"
        );
        // Sanity: the table path would have applied the >272k tier rate.
        let table_cost = calculate_cost("gpt-5.6", &above_tier).expect("table path");
        assert!(
            (table_cost - expected).abs() > 1e-6,
            "table path ({table_cost}) must differ from the flat override ({expected})"
        );
    }

    /// Coverage: `sanitize_rate` must collapse `+∞` (not just NaN / negative)
    /// to 0.0. `is_finite()` handles it but it was previously unasserted.
    #[test]
    fn sanitize_rate_collapses_positive_infinity() {
        assert_eq!(sanitize_rate(f64::INFINITY), 0.0);
        assert_eq!(sanitize_rate(f64::NEG_INFINITY), 0.0);
        assert_eq!(sanitize_rate(f64::NAN), 0.0);
        assert_eq!(sanitize_rate(-1e-6), 0.0);
        assert_eq!(sanitize_rate(0.0), 0.0);
        assert!((sanitize_rate(2.5e-6) - 2.5e-6).abs() < 1e-15);
    }

    /// Coverage: the override path with `cache_creation_1h_input_tokens`.
    /// `DeploymentPricing` has no 1h field, so `TokenRates::from_override`
    /// sets the 1h rate to the override's cache-creation rate (or input
    /// fallback). 1h-ephemeral tokens must bill at that rate end-to-end.
    #[test]
    fn override_path_bills_1h_cache_creation_at_override_creation_rate() {
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.0,
            cache_creation_input_token_cost: Some(0.0005),
            ..Default::default()
        };
        let created = 100u64;
        let created_1h = 60u64;
        let u = UsageSummary {
            prompt_tokens: 50,
            completion_tokens: 0,
            total_tokens: 50,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: Some(created),
            cache_creation_1h_input_tokens: Some(created_1h),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost = calculate_cost_with_override("any-model", &u, Some(&dp)).expect("priced");
        // 1h bucket (60) bills at the override's cache-creation rate (0.0005);
        // the 5m remainder (40) also at 0.0005; fresh input (50) at 0.001.
        let expected =
            50.0 * 0.001 + (created - created_1h) as f64 * 0.0005 + created_1h as f64 * 0.0005;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    /// MEDIUM (C10): a per-second deployment pricing override must NOT bill
    /// token math (its per-token rates are unset/zero) — the token path
    /// returns `None` and spend accrues via
    /// [`DeploymentPricing::cost_per_second`], mirroring the reference
    /// per-second custom-pricing branch (`rate * response_time_ms / 1000`).
    #[test]
    fn override_per_second_skips_token_path_and_bills_via_helper() {
        let dp = DeploymentPricing {
            input_cost_per_second: Some(0.0001),
            ..Default::default()
        };
        assert!(dp.is_per_second_priced());
        let u = usage(1_000, 500);
        assert!(
            calculate_cost_with_override("any-model", &u, Some(&dp)).is_none(),
            "per-second override must return None on the token-based cost path"
        );
        // 2500 ms * 0.0001 $/s / 1000 = 0.00025
        let cost = dp
            .cost_per_second(2500.0)
            .expect("per-second override is billable via the helper");
        assert!((cost - 0.00025).abs() < 1e-15, "got {cost}");

        // Legacy rates describe the same elapsed time; input takes precedence.
        let both = DeploymentPricing {
            input_cost_per_second: Some(0.0001),
            output_cost_per_second: Some(0.0002),
            ..Default::default()
        };
        let cost = both.cost_per_second(1000.0).expect("billable");
        assert!((cost - 0.0001).abs() < 1e-15, "got {cost}");

        // A per-token override is not per-second priced: the helper returns
        // None and the token path computes cost as before.
        let token_dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.002,
            ..Default::default()
        };
        assert!(!token_dp.is_per_second_priced());
        assert!(token_dp.cost_per_second(2500.0).is_none());
        assert!(calculate_cost_with_override("any-model", &u, Some(&token_dp)).is_some());
    }

    /// MEDIUM (C10): malformed per-second override rates (negative / NaN /
    /// ±∞) and corrupt latency measurements must collapse to `0.0` spend —
    /// never negative, never non-finite (which would poison the spend
    /// counter and the reservation/settle math).
    #[test]
    fn override_per_second_sanitizes_rates_and_latency() {
        let dp = DeploymentPricing {
            input_cost_per_second: Some(-0.5),
            output_cost_per_second: Some(f64::NAN),
            ..Default::default()
        };
        assert_eq!(dp.cost_per_second(2500.0).unwrap(), 0.0);

        let inf = DeploymentPricing {
            input_cost_per_second: Some(f64::INFINITY),
            ..Default::default()
        };
        assert_eq!(inf.cost_per_second(2500.0).unwrap(), 0.0);
        assert_eq!(inf.cost_per_second(f64::NEG_INFINITY).unwrap(), 0.0);

        let ok = DeploymentPricing {
            input_cost_per_second: Some(0.0001),
            ..Default::default()
        };
        assert_eq!(ok.cost_per_second(-100.0).unwrap(), 0.0);
        assert_eq!(ok.cost_per_second(f64::NAN).unwrap(), 0.0);
    }

    /// MEDIUM (C10): when the override carries the inherited 1h
    /// cache-creation rate, 1h-ephemeral tokens bill at that rate (not the
    /// 5m rate); when the field is absent they fall back to the 5m rate
    /// (pinned by `override_path_bills_1h_cache_creation_at_override_creation_rate`).
    #[test]
    fn override_1h_rate_bills_when_inherited_field_set() {
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.0,
            cache_creation_input_token_cost: Some(0.0005),
            cache_creation_input_token_cost_above_1hr: Some(0.0008),
            ..Default::default()
        };
        let created = 100u64;
        let created_1h = 60u64;
        let u = UsageSummary {
            prompt_tokens: 50,
            completion_tokens: 0,
            total_tokens: 50,
            cache_creation_input_tokens: Some(created),
            cache_creation_1h_input_tokens: Some(created_1h),
            ..UsageSummary::default()
        };
        let cost = calculate_cost_with_override("any-model", &u, Some(&dp)).expect("priced");
        let expected =
            50.0 * 0.001 + (created - created_1h) as f64 * 0.0005 + created_1h as f64 * 0.0008;
        assert!(
            (cost - expected).abs() < 1e-12,
            "1h bucket must bill at the inherited 1h rate: got {cost}, expected {expected}"
        );
    }

    /// Coverage: `parse_threshold_key` saturates on overflow. A huge `k` value
    /// whose `k * 1000` exceeds `u32::MAX` clamps to `u32::MAX` rather than
    /// wrapping or dropping the tier. (Vendored data, so low impact.)
    #[test]
    fn parse_threshold_key_saturates_on_overflow() {
        let (base, threshold) = parse_threshold_key("input_cost_per_token_above_5000000k_tokens")
            .expect("huge-k tier key must parse");
        assert_eq!(base, "input_cost_per_token");
        assert_eq!(
            threshold,
            u32::MAX,
            "5_000_000 * 1000 overflows u32 and must saturate to u32::MAX"
        );
    }

    #[test]
    fn service_price_keys_do_not_overwrite_standard_context_rates() {
        let rows = parse_provider_prices(
            "test",
            r#"
["separate-service-test"]
input_cost_per_token = 1e-6
output_cost_per_token = 0.0
input_cost_per_token_above_200k_tokens = 2e-6
input_cost_per_token_above_200k_tokens_priority = 5e-6
"#,
        );
        let pricing = &rows["separate-service-test"];
        assert_eq!(pricing.rates_for_prompt(200_001, 0).input, 2e-6);
        assert_eq!(
            pricing
                .rates_for_service_tier(200_001, 0, Some("priority"))
                .input,
            5e-6
        );
    }

    /// LOW (C64 round 2): a tier-shaped key that fails `parse_threshold_key`
    /// (hand-vendored typos like `_above_200ktokens` / `_above_200K_tokens`,
    /// or wrong-order compounds like `..._above_1hr_above_200k_tokens_above_1hr`)
    /// must produce NO tier — and be recognised by `looks_like_tier_key` so
    /// the drop is surfaced by the load-time `tracing::warn!` instead of
    /// silently billing long-context requests at base rates. Non-tier keys
    /// (base rates, the `_above_1hr` TTL key) must stay silent.
    #[test]
    fn tier_shaped_unparsable_keys_are_skipped_and_warn_shaped() {
        let raw: toml::Table = r#"
input_cost_per_token = 1e-6
input_cost_per_token_above_200ktokens = 2e-6
"input_cost_per_token_above_200K_tokens" = 3e-6
"#
        .parse()
        .expect("parse");
        let tiers = extract_cost_tiers(&toml::Value::Table(raw));
        assert!(
            tiers.is_empty(),
            "typo'd tier spellings must not produce a tier (long-context \
             requests bill at base rates — now with a load-time warn)"
        );

        assert!(looks_like_tier_key("input_cost_per_token_above_200ktokens"));
        assert!(looks_like_tier_key(
            "input_cost_per_token_above_200K_tokens"
        ));
        assert!(looks_like_tier_key(
            "cache_creation_input_token_cost_above_1hr_above_200k_tokens_above_1hr"
        ));
        // Non-tier keys must NOT be flagged: base rates and the 1h TTL key
        // contain no `tokens` fragment alongside `_above_`.
        assert!(!looks_like_tier_key("input_cost_per_token"));
        assert!(!looks_like_tier_key("output_cost_per_token"));
        assert!(!looks_like_tier_key(
            "cache_creation_input_token_cost_above_1hr"
        ));
        assert!(!looks_like_tier_key("max_input_tokens"));
    }

    /// LOW (C64): a tier carrying ALL five tierable rate fields at the same
    /// threshold is the most complex combination of the selection logic and
    /// the one most likely to regress if a field is dropped from
    /// `set_tier_rate` or `rates_for_prompt`. Verify that every bucket bills
    /// at its distinct tier rate simultaneously above the threshold.
    #[test]
    fn tiered_cost_all_five_rate_fields_set_at_same_threshold() {
        let p = lookup_pricing("tiered-all-fields-model").expect("test shard loaded");
        assert_eq!(p.tiers.len(), 1);
        let tier = &p.tiers[0];
        assert_eq!(tier.threshold_tokens, 100_000);
        assert_eq!(tier.input_cost_per_token, Some(2e-6));
        assert_eq!(tier.output_cost_per_token, Some(1e-5));
        assert_eq!(tier.cache_read_input_token_cost, Some(2e-7));
        assert_eq!(tier.cache_creation_input_token_cost, Some(4e-6));
        assert_eq!(tier.cache_creation_input_token_cost_above_1hr, Some(8e-6));

        // Above the threshold: every tierable bucket resolves to the tier rate.
        let rates = p.rates_for_prompt(150_000, 0);
        assert_eq!(rates.input, 2e-6);
        assert_eq!(rates.output, 1e-5);
        assert_eq!(rates.cache_read, 2e-7);
        assert_eq!(rates.cache_creation, 4e-6);
        assert_eq!(rates.cache_creation_1h, 8e-6);

        // End-to-end: a mixed request bills every bucket at its distinct tier
        // rate. cache_read (50k) is a subset of prompt (150k) so fresh input is
        // 100k; cache_creation is split 40 (5m) + 60 (1h).
        let prompt = 150_000u64;
        let cache_read = 50_000u64;
        let created = 100u64;
        let created_1h = 60u64;
        let completion = 500u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: Some(cache_read),
            cache_creation_input_tokens: Some(created),
            cache_creation_1h_input_tokens: Some(created_1h),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost = calculate_cost("tiered-all-fields-model", &u).expect("cost");
        let expected = (prompt - cache_read) as f64 * 2e-6
            + cache_read as f64 * 2e-7
            + (created - created_1h) as f64 * 4e-6
            + created_1h as f64 * 8e-6
            + completion as f64 * 1e-5;
        assert!(
            (cost - expected).abs() < 1e-9,
            "got {cost}, expected {expected}"
        );

        // Sanity: below the threshold every bucket stays on the base rates.
        let below = p.rates_for_prompt(99_999, 0);
        assert_eq!(below.input, 1e-6);
        assert_eq!(below.output, 5e-6);
        assert_eq!(below.cache_read, 1e-7);
        assert_eq!(below.cache_creation, 1e-6);
        assert_eq!(below.cache_creation_1h, 2e-6);
    }

    /// LOW (C64): the descending-sort + first-match + break selection loop was
    /// only tested with two tiers. Pin a 3-tier chain (100k, 200k, 300k) so a
    /// future refactor that limits iteration depth or mishandles the break is
    /// caught. A prompt exceeding only the middle threshold must select it.
    #[test]
    fn tiered_cost_three_tiers_middle_selected() {
        let p = lookup_pricing("tiered-three-tier-model").expect("test shard loaded");
        assert_eq!(p.tiers.len(), 3, "the model must declare three tiers");
        // Sorted descending: [300k, 200k, 100k].
        assert_eq!(p.tiers[0].threshold_tokens, 300_000);
        assert_eq!(p.tiers[1].threshold_tokens, 200_000);
        assert_eq!(p.tiers[2].threshold_tokens, 100_000);

        // Exclusive `>` on this non-xAI shard: an exact hit on the lowest
        // threshold bills at the base rate; one token more activates it.
        assert_eq!(
            calculate_cost("tiered-three-tier-model", &usage(100_000, 0)).expect("cost"),
            100_000.0 * 1e-6,
            "exact 100k threshold stays on base rates (exclusive boundary)"
        );
        assert_eq!(
            calculate_cost("tiered-three-tier-model", &usage(100_001, 0)).expect("cost"),
            100_001.0 * 2e-6,
        );
        // Exceeds only the 100k tier.
        assert_eq!(
            calculate_cost("tiered-three-tier-model", &usage(150_000, 0)).expect("cost"),
            150_000.0 * 2e-6,
        );
        // Exceeds 100k and 200k; the highest exceeded is the MIDDLE (200k) tier.
        assert_eq!(
            calculate_cost("tiered-three-tier-model", &usage(250_000, 0)).expect("cost"),
            250_000.0 * 3e-6,
            "a prompt exceeding only the middle threshold must select it"
        );
        // Exceeds all three; the highest (300k) tier wins.
        assert_eq!(
            calculate_cost("tiered-three-tier-model", &usage(350_000, 0)).expect("cost"),
            350_000.0 * 4e-6,
        );
    }

    /// LOW (C64): two key spellings that parse to the same (base, threshold)
    /// (e.g. `_above_100k_tokens` + `_above_100000_tokens`) silently overwrite
    /// via `=`. The vendored JSON never mixes spellings, but a hand-edited
    /// TOML could. This pins the observable behaviour — the last key in
    /// BTreeMap-sorted order wins — and documents that `extract_cost_tiers`
    /// now emits a `tracing::warn!` on the overwrite so it cannot pass
    /// unnoticed.
    #[test]
    fn extract_cost_tiers_warns_on_duplicate_threshold_key_spellings() {
        let raw: toml::Table = r#"
input_cost_per_token = 1e-6
input_cost_per_token_above_100000_tokens = 2e-6
input_cost_per_token_above_100k_tokens = 5e-6
"#
        .parse()
        .expect("parse");
        let tiers = extract_cost_tiers(&toml::Value::Table(raw));
        // The two spellings collapse to ONE tier at threshold 100_000.
        assert_eq!(tiers.len(), 1);
        assert_eq!(tiers[0].threshold_tokens, 100_000);
        // BTreeMap iteration order: `..._above_100000_tokens` sorts before
        // `..._above_100k_tokens` ('0' < 'k'), so the `k` spelling is applied
        // second and wins via the overwrite.
        assert_eq!(
            tiers[0].input_cost_per_token,
            Some(5e-6),
            "the later (BTreeMap-sorted) duplicate spelling must win"
        );
    }

    /// Coverage: a model name that LEGITIMATELY ends in 8 digits (not a date
    /// snapshot) must resolve to its EXACT table entry, not be stripped to a
    /// different model. Guards the `strip_date_version_suffix` Pattern 2
    /// misfire: when both the 8-digit-suffixed name and its stripped form are
    /// priced, the exact key wins (the strip only runs after a direct lookup
    /// miss).
    #[test]
    fn legitimate_eight_digit_suffix_not_stripped() {
        let full =
            lookup_pricing("eight-digit-real-12345678").expect("exact 8-digit key must be priced");
        let stripped = lookup_pricing("eight-digit-real").expect("stripped form also priced");
        assert!(
            (full.input_cost_per_token - stripped.input_cost_per_token).abs() > 1e-12,
            "the two entries must have distinct rates for this guard to be meaningful"
        );
        assert_eq!(
            full.input_cost_per_token, 9e-6,
            "the exact 8-digit key must win, not the stripped alias"
        );
    }

    /// Coverage: `lookup_pricing_with_provider` with an empty provider must
    /// reproduce the legacy provider-agnostic bare lookup exactly (no prefixed
    /// key is ever constructed).
    #[test]
    fn lookup_pricing_with_provider_empty_reproduces_bare_lookup() {
        let via_empty = lookup_pricing_with_provider("gpt-4o", "");
        let bare = lookup_pricing("gpt-4o");
        assert!(
            std::ptr::eq(via_empty.unwrap(), bare.unwrap()),
            "empty provider must reproduce the bare lookup exactly"
        );
        // An unknown model with an empty provider still resolves to None.
        assert!(lookup_pricing_with_provider("not-a-real-model", "").is_none());
    }

    /// Coverage: when `cache_read + audio > prompt` (inconsistent provider
    /// data), `cache_read` is billed fully and the excess audio is dropped.
    /// This is a conservative divergence from the reference (which does not clamp
    /// and can over-bill on inconsistent data). Pin the clamp ordering so
    /// cache_read is always prioritised over audio in the subset budget.
    #[test]
    fn audio_dropped_when_cache_read_plus_audio_exceeds_prompt() {
        let pricing = ModelPricing {
            input_cost_per_token: 1e-6,
            output_cost_per_token: 0.0,
            input_cost_per_audio_token: Some(5e-6),
            cache_read_input_token_cost: Some(1e-7),
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            output_cost_per_reasoning_token: None,
            citation_cost_per_token: None,
            search_queries_cost_per_query: None,
            max_input_tokens: None,
            max_output_tokens: None,
            provider: "test".to_string(),
            input_cost_per_second: None,
            prompt_cache_min_tokens: None,
            tiers: Vec::new(),
            tier_threshold_is_inclusive: false,
            ..Default::default()
        };
        // prompt=100, cache_read=80, audio=50 (80+50=130 > 100). cache_read is
        // fully billed (80), audio is clamped to the remaining 20, and the
        // regular-input bucket is 0.
        let u = UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 0,
            total_tokens: 100,
            cache_read_input_tokens: Some(80),
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: Some(50),
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let rates = pricing.rates_for_prompt(100, 0);
        let cost = cost_from_rates(&u, rates);
        let expected = 80.0 * 1e-7 + 20.0 * 5e-6;
        assert!(
            (cost - expected).abs() < 1e-12,
            "cache_read must be prioritised over audio in the subset clamp: got {cost}, expected {expected}"
        );
    }

    // =================================================================
    // C01 review follow-ups (round 2).
    // =================================================================

    /// LOW #1: a model that declares a 5m cache-creation rate but NOT a 1h
    /// rate must bill the 1h bucket at 0.0, NOT at the 5m rate. The
    /// `cache_creation_input_token_cost_above_1hr` doc-comment previously
    /// promised a 5m fallback that the code never implemented; the code is
    /// reference-compatible (`_get_cost_per_unit(..., default=0.0)`), so
    /// the doc was wrong. This test pins the actual behaviour so the doc
    /// drift cannot re-occur and a future maintainer cannot "fix" the code
    /// toward the old (wrong) doc.
    #[test]
    fn cache_creation_1h_rate_bills_zero_when_absent() {
        let pricing = ModelPricing {
            input_cost_per_token: 1e-6,
            output_cost_per_token: 2e-6,
            input_cost_per_audio_token: None,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: Some(3e-6),
            cache_creation_input_token_cost_above_1hr: None,
            output_cost_per_reasoning_token: None,
            citation_cost_per_token: None,
            search_queries_cost_per_query: None,
            max_input_tokens: None,
            max_output_tokens: None,
            provider: "test".to_string(),
            input_cost_per_second: None,
            prompt_cache_min_tokens: None,
            tiers: Vec::new(),
            tier_threshold_is_inclusive: false,
            ..Default::default()
        };
        let rates = pricing.rates_for_prompt(1_000, 0);
        assert_eq!(rates.cache_creation, 3e-6, "5m rate must resolve");
        assert_eq!(
            rates.cache_creation_1h, 0.0,
            "absent 1h rate must bill at 0.0, NOT the 5m rate"
        );
    }

    /// LOW #2: `lookup_pricing_with_provider` previously tried exactly one
    /// prefixed key (`{provider}/{model}`) then delegated to the bare
    /// `lookup_pricing`, which only constructs BARE candidates. A dated
    /// snapshot served by a prefixed-key provider (e.g.
    /// `gemini-2.5-pro-2025-03-25` served by `gemini`, row keyed
    /// `gemini/gemini-2.5-pro`) therefore missed at every stage and fell
    /// through to unmetered `None` spend. The shared resolver now tries the
    /// prefixed form of the date-stripped candidate too. This test builds a
    /// minimal prefixed-key table and asserts the dated + prefixed path
    /// resolves, while the bare lookup (no provider hint) still misses
    /// because the only key is prefixed.
    #[test]
    fn lookup_with_provider_resolves_dated_prefixed_key() {
        let row = ModelPricing {
            input_cost_per_token: 7e-6,
            output_cost_per_token: 0.0,
            input_cost_per_audio_token: None,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            output_cost_per_reasoning_token: None,
            citation_cost_per_token: None,
            search_queries_cost_per_query: None,
            max_input_tokens: None,
            max_output_tokens: None,
            provider: "gemini".to_string(),
            input_cost_per_second: None,
            prompt_cache_min_tokens: None,
            tiers: Vec::new(),
            tier_threshold_is_inclusive: false,
            ..Default::default()
        };
        let mut table = HashMap::new();
        table.insert("gemini/gemini-test-dated".to_string(), row);

        // Dated model served by `gemini`: exact prefixed miss, bare miss,
        // then date-stripped prefixed hit.
        let resolved =
            lookup_with_optional_provider(&table, "gemini-test-dated-2024-01-01", Some("gemini"))
                .expect("dated + provider-prefixed key must resolve");
        assert_eq!(resolved.input_cost_per_token, 7e-6);
        assert_eq!(resolved.provider, "gemini");

        // The `-YYYYMMDD` compact form (Anthropic snapshot shape) must also
        // resolve via the same path.
        let resolved_compact =
            lookup_with_optional_provider(&table, "gemini-test-dated-20240101", Some("gemini"))
                .expect("compact-dated + provider-prefixed key must resolve");
        assert_eq!(resolved_compact.input_cost_per_token, 7e-6);

        // Without a provider hint, the bare date-stripped name still misses
        // because the only key in this table is the prefixed form — pinning
        // that the provider hint is what closes the gap.
        assert!(
            lookup_with_optional_provider(&table, "gemini-test-dated-2024-01-01", None).is_none(),
            "bare lookup must miss when the key is provider-prefixed"
        );
    }

    /// LOW #3: when a provider reports the 1h-ephemeral detail but OMITS the
    /// aggregate `cache_creation_input_tokens`, the old clamp
    /// (`1h.min(total = 0)`) zeroed the 1h bucket — billing those tokens
    /// free. The reference `calculate_cache_writing_cost` uses the detail fields
    /// directly when present, so it bills the 1h tokens at the 1h rate. The
    /// resolver now treats the 1h detail as the effective total when the
    /// aggregate is absent, so the 1h tokens accrue spend and the 5m
    /// remainder is zero.
    #[test]
    fn cache_creation_1h_bills_when_aggregate_absent() {
        let pricing =
            lookup_pricing("claude-sonnet-4-20250514").expect("claude-sonnet-4-20250514 priced");
        let one_h = pricing
            .cache_creation_input_token_cost_above_1hr
            .expect("1h rate");
        let prompt = 100u64;
        let created_1h = 100u64;
        let u = UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: 0,
            total_tokens: prompt,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: Some(created_1h),
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
        };
        let cost = calculate_cost("claude-sonnet-4-20250514", &u).expect("priced");
        let expected = prompt as f64 * pricing.input_cost_per_token + created_1h as f64 * one_h;
        assert!(
            (cost - expected).abs() < 1e-12,
            "1h detail must bill at the 1h rate when the aggregate is absent: got {cost}, expected {expected}"
        );

        // Sanity: the pre-fix behaviour billed the 1h tokens free (only the
        // prompt accrued spend), so the fixed cost must be strictly higher.
        let pre_fix = prompt as f64 * pricing.input_cost_per_token;
        assert!(
            cost > pre_fix,
            "1h tokens must accrue spend, not bill free: got {cost}, pre-fix {pre_fix}"
        );
    }

    #[test]
    fn deployment_override_without_tier_keys_stays_flat() {
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.002,
            ..Default::default()
        };
        let below = UsageSummary {
            prompt_tokens: 272_000,
            completion_tokens: 10,
            ..Default::default()
        };
        let above = UsageSummary {
            prompt_tokens: 272_001,
            completion_tokens: 10,
            ..Default::default()
        };
        let low = calculate_cost_with_override("gpt-6-luna", &below, Some(&dp)).unwrap();
        let high = calculate_cost_with_override("gpt-6-luna", &above, Some(&dp)).unwrap();
        let expected_low = 272_000.0 * 0.001 + 10.0 * 0.002;
        let expected_high = 272_001.0 * 0.001 + 10.0 * 0.002;
        assert!((low - expected_low).abs() < 1e-9);
        assert!((high - expected_high).abs() < 1e-9);
    }

    #[test]
    fn deployment_tier_boundary_is_exclusive_at_272k() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "input_cost_per_token_above_272k_tokens".into(),
            serde_json::json!(0.002),
        );
        extra.insert(
            "output_cost_per_token_above_272k_tokens".into(),
            serde_json::json!(0.004),
        );
        let tiers = cost_tiers_from_extra(&extra);
        assert_eq!(tiers.len(), 1);
        assert_eq!(tiers[0].threshold_tokens, 272_000);
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.002,
            tiers,
            ..Default::default()
        };
        let at = UsageSummary {
            prompt_tokens: 272_000,
            completion_tokens: 1,
            ..Default::default()
        };
        let past = UsageSummary {
            prompt_tokens: 272_001,
            completion_tokens: 1,
            ..Default::default()
        };
        let at_cost = calculate_cost_with_override("custom", &at, Some(&dp)).unwrap();
        let past_cost = calculate_cost_with_override("custom", &past, Some(&dp)).unwrap();
        assert!((at_cost - (272_000.0 * 0.001 + 0.002)).abs() < 1e-9);
        assert!((past_cost - (272_001.0 * 0.002 + 0.004)).abs() < 1e-9);
    }

    #[test]
    fn deployment_tier_none_rate_falls_back_to_override_base() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "input_cost_per_token_above_272k_tokens".into(),
            serde_json::json!(0.009),
        );
        let dp = DeploymentPricing {
            input_cost_per_token: 0.001,
            output_cost_per_token: 0.003,
            cache_read_input_token_cost: Some(0.0004),
            tiers: cost_tiers_from_extra(&extra),
            ..Default::default()
        };
        let usage = UsageSummary {
            prompt_tokens: 300_000,
            completion_tokens: 2,
            cache_read_input_tokens: Some(10),
            ..Default::default()
        };
        let cost = calculate_cost_with_override("custom", &usage, Some(&dp)).unwrap();
        // 10 cache-read tokens stay on the override cache rate. The rest of
        // the prompt uses the tier input rate. Output has no tier rate, so
        // it stays on the override base.
        let expected = (300_000.0 - 10.0) * 0.009 + 10.0 * 0.0004 + 2.0 * 0.003;
        assert!(
            (cost - expected).abs() < 1e-6,
            "got {cost}, expected {expected}"
        );
    }

    #[test]
    fn malformed_tier_key_is_classified_and_not_billed() {
        assert_eq!(
            classify_extra_tier_key("input_cost_per_token_above_272k_tokens"),
            ExtraTierKeyClass::Parsed
        );
        assert_eq!(
            classify_extra_tier_key("input_cost_per_token_above_200ktokens"),
            ExtraTierKeyClass::Malformed
        );
        assert_eq!(
            classify_extra_tier_key("cache_creation_input_token_cost_above_1hr"),
            ExtraTierKeyClass::NotTier
        );
        let mut extra = serde_json::Map::new();
        extra.insert(
            "input_cost_per_token_above_200ktokens".into(),
            serde_json::json!(0.5),
        );
        assert!(cost_tiers_from_extra(&extra).is_empty());
        assert!(extra_tier_rate_is_non_numeric(
            "input_cost_per_token_above_272k_tokens",
            &serde_json::json!("nope"),
        ));
    }

    #[test]
    fn deployment_tier_threshold_counts_standalone_one_hour_cache_creation() {
        let params: crate::deployment::OxllmParams = serde_json::from_value(serde_json::json!({
            "model": "custom", "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002, "input_cost_per_token_above_272k_tokens": 0.003,
        }))
        .unwrap();
        let usage = UsageSummary {
            prompt_tokens: 272_000,
            cache_creation_1h_input_tokens: Some(1),
            ..Default::default()
        };
        let pricing = params.pricing_override().unwrap();
        let cost = calculate_cost_with_override("custom", &usage, Some(&pricing)).unwrap();
        assert!((cost - 272_001.0 * 0.003).abs() < 1e-9);
    }

    #[test]
    fn deployment_tier_inclusive_boundary_can_be_configured() {
        let params: crate::deployment::OxllmParams = serde_json::from_value(serde_json::json!({
            "model": "custom", "input_cost_per_token": 0.001,
            "input_cost_per_token_above_272k_tokens": 0.003, "tier_threshold_is_inclusive": true,
        }))
        .unwrap();
        let usage = UsageSummary {
            prompt_tokens: 272_000,
            ..Default::default()
        };
        let pricing = params.pricing_override().unwrap();
        assert!(pricing.tier_threshold_is_inclusive);
        let cost = calculate_cost_with_override("custom", &usage, Some(&pricing)).unwrap();
        assert!((cost - 272_000.0 * 0.003).abs() < 1e-9);
        assert_eq!(
            classify_extra_tier_key("input_cost_per_token_above_4294968k_tokens"),
            ExtraTierKeyClass::Malformed
        );
    }
}
