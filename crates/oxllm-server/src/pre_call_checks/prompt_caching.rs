//! Sticky-by-prefix routing that helps provider-side prompt caching.
//!
//! When a request's prompt prefix is large enough for provider-side caching
//! (e.g. Anthropic `cache_control`, which requires a minimum of ~1024 tokens),
//! pin the request to the deployment that already served an identical prefix
//! so repeat traffic concentrates on one deployment. When the caller opts into
//! provider-side caching (by annotating `cache_control` breakpoints, which the
//! Anthropic converter passes through verbatim), this maximises cache-hit
//! rates; when the caller does not, the pin still acts as sticky routing,
//! which is harmless and occasionally helpful (e.g. it keeps a conversation's
//! follow-ups on a single warm backend).
//!
//! - Pre-call: hash the system prompt (or first user message) **and** the
//!   request's `tools` array with SHA-256 → look up
//!   `pcache:{api_key_hash}:{hash}` in the shared cache → when a mapping
//!   exists and the pinned deployment is available, prefer it over the
//!   router's pick.
//! - Post-call: write `pcache:{api_key_hash}:{hash}` → deployment id with a
//!   5-minute TTL (matching the reference).
//!
//! The pin value is the deployment's stable cooldown key
//! (`stable_key(deployment)`, see
//! [`oxllm_core::router::Router::cooldown_key`]) — *not* the simpler
//! `{model_name}:{provider_model}` spend-counter key (`deployment_key`), and
//! *not* the old positional `{model_name}:{index}` scheme. The cooldown key
//! is a pure hash of the deployment's full serialized `oxllm_params`, so it
//! distinguishes sibling deployments that share a model but differ in
//! endpoint/credentials and is stable across config hot-reloads (a reordered
//! list never reassigns a pin to a different deployment). The read side
//! (router `candidate_keys`) and the write side (`pin`) use the same key
//! space, so a pin written by one leg is observable by the next. The pin key
//! is scoped by the caller's API-key hash so pins never leak across tenants
//! sharing a cache.
//!
//! The check is opt-in via `oxllm_settings.prompt_caching_enabled` and
//! engages when the estimated prefix token count meets the threshold. The
//! pre-call *lookup* uses the **minimum** resolved `prompt_cache_min_tokens`
//! across the model group's deployments (so a high-floor member cannot
//! suppress a valid lookup for a low-floor member that may hold a warm cache,
//! matching the reference `_get_min_token_count_for_deployments`); the post-call
//! *write* uses the per-deployment resolved minimum (`engages`).
//!
//! # Cache-key caveats
//!
//! - The cache key is SHA-256 truncated to 128 bits — a stronger hash than
//!   FNV-1a (which the `SessionAffinity` strategy still uses for short-lived
//!   session ids) and large enough that birthday-paradox collisions are not a
//!   practical concern within a tenant's pin space.
//! - The hash is computed over the serialized message JSON **without Unicode
//!   normalization**: NFC and NFD variants of the same visible text produce
//!   different cache keys (this matches reference behavior).
//! - The cache key is sensitive to the full message JSON structure — role,
//!   content-part layout, and serialized field ordering — **and** to the
//!   request's `tools` array, not just the visible text. Same prefix with a
//!   different tool set produces a different pin (the provider-side cache key
//!   covers tool definitions, so this keeps the pin aligned with cache
//!   warmth).
//!
//! # Reference parity divergence (documented)
//!
//! The reference `get_prompt_caching_cache_key` returns `None` when **no** content
//! block in the request carries `cache_control: {type: "ephemeral"}`, and
//! neither the read nor the write touches the cache in that case. oxllm
//! intentionally diverges: it pins **every** large prompt regardless of
//! `cache_control` annotation, because oxllm does not auto-inject
//! `cache_control` breakpoints (see the future-feature note below) and the
//! pin doubles as sticky routing for conversations whose callers have not
//! opted into caching. The cost is that oxllm writes pins the reference would skip
//! (degenerating to sticky-by-prefix routing when the provider has nothing
//! cached); the benefit is simpler, uniformly sticky routing. Folding `tools`
//! into the key and hashing the first system/user message keeps collisions
//! aligned with provider cache prefixes where the caller *has* opted in.
//!
//! # Future feature: auto-inject `cache_control` breakpoints (U31)
//!
//! The reference adds *automatic* injection of Anthropic
//! `cache_control: {type: "ephemeral"}` breakpoints into the request body
//! (on the last tool definition, the last system block, and the last user
//! message) when the request is large enough to benefit from caching.
//!
//! oxllm does NOT currently implement this. What oxllm does today:
//!
//! - **Deployment pinning** (this module): the router-side half of prompt
//!   caching — concentrates repeat prefixes on one deployment so a caller
//!   who opts into `cache_control` sees higher hit rates.
//! - **Caller-supplied `cache_control` passthrough**: when a caller
//!   explicitly annotates a content block with `cache_control`, the
//!   Anthropic converter preserves it verbatim (see `handlers::convert::anthropic`
//!   and the `has_cache_control` branches in `anthropic::split_system_messages`).
//!   So callers who know they want caching can opt in explicitly today.
//!
//! What is missing for full reference parity: a hook that auto-injects
//! `cache_control` breakpoints on the host side, gated on the same
//! `prompt_caching_min_tokens` threshold, so callers get caching without
//! having to annotate their requests. Implementing this requires:
//!
//! 1. A new `PreCallCheck` (or extension of this one) that, after picking
//!    a deployment, rewrites the request body to add `cache_control`
//!    markers at the Anthropic-defined breakpoint positions.
//! 2. The rewrite must run AFTER provider-specific request transformation
//!    (so the markers land on the Anthropic-shaped body, not the OpenAI
//!    one), which is a new injection point — the current pre-call hook
//!    chain runs against the OpenAI-shaped body.
//! 3. The threshold logic should reuse [`PromptCachingCheck::engages`] so
//!    auto-injection only happens for prompts large enough to benefit.
//!
//! Tracked as a future feature; not a bug fix.

use std::sync::Arc;
use std::time::Duration;

use oxllm_core::cost::lookup_pricing;
use oxllm_core::deployment::Deployment;
use oxllm_core::{Cache, estimate_tokens};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Default minimum estimated prompt-prefix tokens for the check to engage
/// (Anthropic's prompt-cache minimum is 1024 tokens for most models).
pub const DEFAULT_MIN_TOKENS: u32 = 1024;

/// TTL for prompt → deployment pin entries (5 minutes, matching the reference
/// `PromptCachingCache` set calls, which pass `ttl=300`). Exposed to tests so
/// the value is pinned; a regression here silently changes how long stale
/// pins survive deployment rotation.
pub(crate) const PIN_TTL: Duration = Duration::from_secs(300);

/// Pins cacheable prompts to the deployment whose provider-side cache is
/// already warm for that prompt.
pub struct PromptCachingCheck {
    cache: Arc<dyn Cache>,
    default_min_tokens: Option<u32>,
}

impl PromptCachingCheck {
    /// Create a check over the shared cache. `default_min_tokens` is the
    /// global threshold from `oxllm_settings.prompt_caching_min_tokens`.
    pub fn new(cache: Arc<dyn Cache>, default_min_tokens: Option<u32>) -> Self {
        Self {
            cache,
            default_min_tokens,
        }
    }

    /// Resolve the effective minimum-token threshold for a deployment.
    ///
    /// Resolution order (first `Some` wins — U34):
    /// 1. **Model pricing entry** — `prompt_cache_min_tokens` on the model's
    ///    vendored price-table record (looked up via
    ///    [`oxllm_core::cost::lookup_pricing`] against the deployment's
    ///    provider model id). This carries the provider's true minimum (e.g.
    ///    1024 for most Anthropic models, 4096 for Opus 4.5).
    /// 2. **Per-deployment override** —
    ///    `oxllm_params.prompt_caching_min_tokens`, for models not modelled
    ///    in the price table.
    /// 3. **Global default** — `oxllm_settings.prompt_caching_min_tokens`.
    /// 4. [`DEFAULT_MIN_TOKENS`] (1024).
    ///
    /// The model pricing entry wins over the deployment override because it is
    /// the provider's intrinsic floor: a deployment override cannot lower a
    /// request below the threshold the provider would actually honour.
    pub fn min_tokens_for(&self, deployment: &Deployment) -> u32 {
        lookup_pricing(&deployment.oxllm_params.model)
            .and_then(|p| p.prompt_cache_min_tokens)
            .or(deployment.oxllm_params.prompt_caching_min_tokens)
            .or(self.default_min_tokens)
            .unwrap_or(DEFAULT_MIN_TOKENS)
    }

    /// Whether the check engages for a deployment: the estimated prefix
    /// tokens meet the deployment's effective threshold.
    ///
    /// # U35 — write gate semantics
    ///
    /// This is the **pin-write gate**: it decides whether a prompt that was
    /// served by `deployment` is large enough to *record* a pin so future
    /// requests route back here. The reference specifies that the write gate should
    /// use the **highest** minimum across the model group (so we only pin when
    /// the prompt is cacheable for *every* deployment in the group, avoiding
    /// pinning to a deployment whose cache the provider would reject).
    ///
    /// oxllm's check operates **per-deployment** (each deployment resolves its
    /// own threshold via [`Self::min_tokens_for`]), not per-group. Because the
    /// resolved threshold already incorporates the model's intrinsic minimum
    /// (the highest floor in the group for duplicate deployments of the same
    /// model), the per-deployment resolution naturally satisfies the
    /// "highest minimum" requirement — a pin is only written when this
    /// deployment's own threshold is met.
    pub fn engages(&self, deployment: &Deployment, est_tokens: u64) -> bool {
        est_tokens >= u64::from(self.min_tokens_for(deployment))
    }

    /// Whether the pre-call *pin lookup* engages: the estimated prefix
    /// tokens meet the **minimum** resolved `prompt_cache_min_tokens` across
    /// the model group's deployments, mirroring the reference
    /// `_get_min_token_count_for_deployments`.
    ///
    /// The read gate deliberately uses the group minimum (not the maximum,
    /// and not the bare global default): consulting the cache is cheap, and a
    /// higher-floor member cannot be hurt by a lookup that finds nothing,
    /// while a high global threshold must not suppress a valid lookup for a
    /// low-floor member that genuinely holds a warm cache. Concretely, an
    /// operator who sets `prompt_caching_min_tokens: 4096` to accommodate an
    /// Opus-4.5 deployment must NOT silently disable cache consultation for a
    /// Sonnet (1024) group — the group minimum resolves to 1024 and the
    /// lookup engages at 1024 tokens.
    ///
    /// The gate is independent of the router's eventual pick (which carries
    /// its own per-deployment threshold) by construction: it consults the
    /// whole group before any candidate is chosen. Per-deployment thresholds
    /// only gate the pin *write* ([`Self::engages`]). When `group` is empty
    /// (e.g. an unknown model group) the gate falls back to the global
    /// default (`Self::default_min_tokens`) or [`DEFAULT_MIN_TOKENS`], so
    /// the feature fails open rather than silently never engaging.
    ///
    /// # U35 — read gate semantics
    ///
    /// This is the **cache-read gate**: it decides whether to even *consult*
    /// the pin table for a request. The reference specifies that the read gate
    /// should use the **lowest** minimum across the model group (so any
    /// deployment that could have a warm cache is consulted). This method
    /// implements that directly, rather than approximating it with the global
    /// default.
    pub fn engages_globally<'a, I>(&self, est_tokens: u64, group: I) -> bool
    where
        I: IntoIterator<Item = &'a Deployment>,
    {
        let min_tokens = group
            .into_iter()
            .map(|d| self.min_tokens_for(d))
            .min()
            .unwrap_or_else(|| self.default_min_tokens.unwrap_or(DEFAULT_MIN_TOKENS));
        est_tokens >= u64::from(min_tokens)
    }

    /// Compute the SHA-256 (truncated to 128 bits) of the cacheable prompt
    /// prefix (the system prompt when present, otherwise the first user
    /// message) concatenated with the request's `tools` array, plus the
    /// estimated token count of the prefix text. Returns `None` when the
    /// request carries no usable prefix (no messages, or an empty prefix).
    ///
    /// The hash is keyed by both the message prefix and the tool definitions
    /// so that two requests sharing a system prompt but differing in tools
    /// get distinct pins — the provider-side cache key covers tool
    /// definitions, so this keeps the pin aligned with cache warmth.
    ///
    /// # Token-estimate approximation
    ///
    /// `model` here is the client-facing model *group* name (e.g.
    /// `"gpt-4o"`), not the deployment's provider model id — the router
    /// has not picked a deployment yet at this point in the pipeline. The
    /// cache *key* (the hash) is independent of this value, so the
    /// approximation only affects the threshold gate
    /// ([`Self::engages_globally`]). When the group name does not map to a
    /// known BPE tokenizer, [`estimate_tokens`] falls back to the ASCII
    /// `byte_length / 4` heuristic, which under-counts CJK / emoji-heavy
    /// prompts by roughly the ASCII-to-BPE ratio. The net effect is that
    /// `prompt_caching_min_tokens` can feel miscalibrated for non-ASCII
    /// prompts on unmodelled groups (the gate trips later than it should).
    /// This is a known limitation, not a correctness bug: a pin, once
    /// written, is keyed only by the hash.
    pub fn prompt_hash(&self, request_body: &Value, model: &str) -> Option<(u128, u64)> {
        let (canonical, text) = prompt_prefix(request_body)?;
        Some((hash128(&canonical), estimate_tokens(&text, model)))
    }

    /// Look up the pinned deployment id for a prompt hash, scoped to the
    /// caller's API-key hash (tenant isolation).
    pub async fn preferred_deployment(&self, hash: u128, api_key_hash: &str) -> Option<String> {
        self.cache.get(&cache_key(api_key_hash, hash)).await
    }

    /// Record the deployment that served this prompt so future requests with
    /// the same prefix pin to it. `deployment_id` is the deployment's stable
    /// cooldown key (`stable_key(deployment)`, see
    /// [`oxllm_core::router::Router::cooldown_key`]); the entry is scoped to
    /// the caller's API-key hash (tenant isolation).
    pub async fn pin(&self, hash: u128, api_key_hash: &str, deployment_id: &str) {
        self.cache
            .set(
                &cache_key(api_key_hash, hash),
                deployment_id.to_string(),
                Some(PIN_TTL),
            )
            .await;
    }
}

impl std::fmt::Debug for PromptCachingCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptCachingCheck")
            .field("default_min_tokens", &self.default_min_tokens)
            .finish()
    }
}

/// The config name of this check (`optional_pre_call_checks` entry).
pub const CHECK_NAME: &str = "prompt_caching";

/// Normalize a configured check name and compare it to [`CHECK_NAME`]
/// (accepts kebab-case spellings).
pub fn is_prompt_caching_name(name: &str) -> bool {
    oxllm_core::router::normalize_check_name(name) == CHECK_NAME
}

impl oxllm_core::router::PreCallCheck for PromptCachingCheck {
    /// Filter the candidate list to the pinned (cache-warm) deployment when
    /// the host pre-resolved a pin for this request
    /// ([`oxllm_core::router::PreCallData::pinned_deployment_key`]) and the
    /// pinned deployment is among the candidates. Candidates are already
    /// cooldown-filtered by the router, so a pinned candidate is never
    /// cooling. When there is no pin, or the pinned deployment is not a
    /// candidate, all candidates pass unchanged (the router's pick
    /// stands).
    fn filter_deployments<'a>(
        &self,
        candidates: &[&'a Deployment],
        ctx: &oxllm_core::RequestContext,
    ) -> Result<Vec<&'a Deployment>, oxllm_core::Error> {
        let Some(pinned) = ctx.pre_call.pinned_deployment_key.as_deref() else {
            return Ok(candidates.to_vec());
        };
        let keys = &ctx.pre_call.candidate_keys;
        if keys.len() != candidates.len() {
            return Ok(candidates.to_vec());
        }
        if let Some(pos) = keys.iter().position(|k| k == pinned) {
            tracing::debug!(
                event = "prompt_cache_pinned",
                model_id = %pinned,
                "prompt pinned to cache-warm deployment"
            );
            return Ok(vec![candidates[pos]]);
        }
        Ok(candidates.to_vec())
    }

    fn name(&self) -> &'static str {
        CHECK_NAME
    }
}

/// The cache key for a prompt-hash pin entry, scoped by the caller's
/// API-key hash so pins never leak across tenants sharing a cache.
fn cache_key(api_key_hash: &str, hash: u128) -> String {
    format!("pcache:{api_key_hash}:{hash:032x}")
}

/// Extract the cacheable prompt prefix: the system prompt when present,
/// otherwise the first user message, plus the request's `tools` array when
/// present. Returns the canonical string used for hashing (the serialized
/// message JSON concatenated with the serialized `tools` array, so multipart
/// content, roles, and tool definitions are captured exactly) and the visible
/// text used for token estimation.
///
/// `tools` is folded into the canonical string (with a separator that cannot
/// appear in JSON) so that two requests sharing a system prompt but differing
/// in their tool definitions produce distinct pins. Anthropic's provider-side
/// cache key covers tool definitions, so this keeps the pin aligned with
/// cache warmth (the reference cache key also includes `tools`).
fn prompt_prefix(body: &Value) -> Option<(String, String)> {
    let messages = body.get("messages")?.as_array()?;
    let find = |role: &str| {
        messages
            .iter()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some(role))
    };
    let msg = find("system").or_else(|| find("user"))?;
    let text = content_text(msg.get("content")?);
    if text.is_empty() {
        return None;
    }
    let mut canonical = serde_json::to_string(msg).ok()?;
    if let Some(tools) = body.get("tools") {
        if let Ok(tools_json) = serde_json::to_string(tools) {
            canonical.push('\u{0}');
            canonical.push_str(&tools_json);
        }
    }
    Some((canonical, text))
}

/// Extract a best-effort string from a chat message's `content` field.
/// `content` can be a string or an array of content parts (vision/tools), in
/// which case all `text` parts are concatenated.
fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                    p.get("text").and_then(|t| t.as_str()).map(String::from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// SHA-256 of the input, truncated to 128 bits (big-endian first 16 bytes of
/// the digest). Deterministic across processes and Rust versions so every
/// replica computes the same pin key for a given prompt prefix + tool set.
/// 128 bits puts the birthday-paradox 50% collision point at ~2^64 distinct
/// prefixes per tenant — comfortably beyond any single tenant's working set —
/// while keeping the key a fixed-width integer (vs FNV-1a's 64-bit width,
/// which collides at ~2^32 and was the wrong tradeoff for an accumulating
/// pin space). Replaces the previous FNV-1a 64-bit hash.
fn hash128(s: &str) -> u128 {
    let digest = Sha256::digest(s.as_bytes());
    let first16: [u8; 16] = digest[..16].try_into().expect("sha256 digest is 32 bytes");
    u128::from_be_bytes(first16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use oxllm_core::deployment::OxllmParams;
    use serde_json::json;

    fn make_dep(model_name: &str, provider_model: &str) -> Deployment {
        Deployment {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: provider_model.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn check() -> PromptCachingCheck {
        PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), None)
    }

    fn big_prompt_body(tokens_wanted: usize) -> Value {
        // ~4 chars per token under the heuristic/tokenizer; overshoot a bit.
        let text = "lorem ipsum dolor sit amet ".repeat(tokens_wanted / 6 + 8);
        json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": text},
                {"role": "user", "content": "hi"}
            ]
        })
    }

    // ------------------------------------------------------------------
    // prompt_hash / prefix extraction
    // ------------------------------------------------------------------

    #[test]
    fn prompt_hash_prefers_system_prompt_over_user() {
        let c = check();
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "first user message"},
                {"role": "system", "content": "the system prompt"},
            ]
        });
        let with_system = c.prompt_hash(&body, "gpt-4o");
        let system_only = c.prompt_hash(
            &json!({"messages": [{"role": "system", "content": "the system prompt"}]}),
            "gpt-4o",
        );
        assert_eq!(
            with_system.map(|(h, _)| h),
            system_only.map(|(h, _)| h),
            "hash must be computed over the system prompt even when listed second"
        );
    }

    #[test]
    fn prompt_hash_uses_first_user_message_when_no_system() {
        let c = check();
        let body = json!({
            "messages": [
                {"role": "user", "content": "hello there"},
                {"role": "user", "content": "a different second message"},
            ]
        });
        let first_only = c.prompt_hash(
            &json!({"messages": [{"role": "user", "content": "hello there"}]}),
            "gpt-4o",
        );
        assert_eq!(
            c.prompt_hash(&body, "gpt-4o").map(|(h, _)| h),
            first_only.map(|(h, _)| h),
            "hash must cover only the first user message"
        );
    }

    #[test]
    fn prompt_hash_none_without_messages_or_empty_prefix() {
        let c = check();
        assert!(
            c.prompt_hash(&json!({"model": "gpt-4o"}), "gpt-4o")
                .is_none()
        );
        assert!(c.prompt_hash(&json!({"messages": []}), "gpt-4o").is_none());
        assert!(
            c.prompt_hash(
                &json!({"messages": [{"role": "system", "content": ""}]}),
                "gpt-4o"
            )
            .is_none()
        );
    }

    #[test]
    fn prompt_hash_handles_multipart_content() {
        let c = check();
        let body = json!({
            "messages": [{
                "role": "system",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    {"type": "text", "text": "part two"}
                ]
            }]
        });
        let (hash, tokens) = c.prompt_hash(&body, "gpt-4o").expect("prefix present");
        assert!(hash != 0);
        assert!(tokens > 0, "text parts must contribute to the estimate");
    }

    /// C13/LOW: when the model *group* name does not map to a known BPE
    /// tokenizer, [`estimate_tokens`] falls back to `cl100k_base` (matching
    /// the reference token counter, which normalises unknown names to
    /// `gpt-3.5-turbo`). For a CJK prompt this is visibly larger — and closer
    /// to the real BPE token count — than the previous `byte_length / 4`
    /// ASCII heuristic. This locks in the documented approximation so a future
    /// change to the estimator is a conscious decision, and asserts the hash
    /// itself is stable (the cache key is independent of the estimate).
    #[test]
    fn prompt_hash_estimate_uses_cl100k_fallback_for_unmodelled_group() {
        let c = check();
        // CJK system prompt whose group name is not a known model.
        let text = "你好世界，这是一个用于测试的中文提示词。".repeat(4);
        let body = json!({
            "messages": [{"role": "system", "content": text.clone()}]
        });
        let group = "my-custom-group";
        let (hash, tokens) = c.prompt_hash(&body, group).expect("prefix present");
        // The hash must not depend on the model argument — same prefix,
        // same hash regardless of the group name.
        let (hash_again, _) = c.prompt_hash(&body, "other-group").expect("prefix present");
        assert_eq!(hash, hash_again, "hash is model-independent");
        // The estimate follows `cl100k_base` for an unmodelled group, which
        // counts CJK more accurately than the old byte/4 heuristic.
        let expected = oxllm_core::estimate_tokens(&text, group);
        assert_eq!(
            tokens, expected,
            "unmodelled group must use the cl100k fallback; got {tokens}, expected {expected}"
        );
        let byte_heuristic = (text.len() / 4) as u64;
        assert!(
            tokens > byte_heuristic,
            "cl100k must beat the old byte/4 heuristic for CJK: tokens={tokens}, byte_heuristic={byte_heuristic}"
        );
    }

    // ------------------------------------------------------------------
    // Thresholds
    // ------------------------------------------------------------------

    #[test]
    fn min_tokens_resolution_prefers_pricing_then_deployment_then_global_then_default() {
        // gpt-4o has no `prompt_cache_min_tokens` in the price table, so the
        // pricing entry is `None` and resolution falls through to the
        // deployment override → global default → DEFAULT_MIN_TOKENS.
        let pricing = lookup_pricing("gpt-4o").expect("gpt-4o priced");
        assert!(
            pricing.prompt_cache_min_tokens.is_none(),
            "gpt-4o must not carry a modelled cache minimum for this test"
        );

        let mut dep = make_dep("gpt-4o", "openai/gpt-4o");

        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), None);
        assert_eq!(c.min_tokens_for(&dep), DEFAULT_MIN_TOKENS);

        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), Some(2048));
        assert_eq!(c.min_tokens_for(&dep), 2048);

        dep.oxllm_params.prompt_caching_min_tokens = Some(512);
        assert_eq!(
            c.min_tokens_for(&dep),
            512,
            "per-deployment override beats the global default when pricing is absent"
        );
    }

    /// U34: a model whose price-table record carries `prompt_cache_min_tokens`
    /// must resolve to that value, and it wins over both the deployment
    /// override and the global default (provider's intrinsic floor).
    #[test]
    fn min_tokens_resolution_uses_pricing_entry_first() {
        // claude-sonnet-4-20250514 carries prompt_cache_min_tokens = 1024.
        let dep = make_dep(
            "claude-sonnet-4-20250514",
            "anthropic/claude-sonnet-4-20250514",
        );
        assert_eq!(
            lookup_pricing("anthropic/claude-sonnet-4-20250514")
                .expect("priced")
                .prompt_cache_min_tokens,
            Some(1024),
        );

        // No override / no global → pricing entry (1024).
        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), None);
        assert_eq!(c.min_tokens_for(&dep), 1024);

        // Global default set lower: pricing still wins.
        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), Some(512));
        assert_eq!(
            c.min_tokens_for(&dep),
            1024,
            "pricing entry must beat a lower global default"
        );
    }

    /// U34: the pricing entry wins even when the deployment tries to lower
    /// the threshold below the provider's floor.
    #[test]
    fn min_tokens_resolution_pricing_beats_deployment_override() {
        let mut dep = make_dep(
            "claude-sonnet-4-20250514",
            "anthropic/claude-sonnet-4-20250514",
        );
        dep.oxllm_params.prompt_caching_min_tokens = Some(128);
        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), Some(64));
        assert_eq!(
            c.min_tokens_for(&dep),
            1024,
            "pricing floor (1024) must win over the deployment override (128) and global (64)"
        );
    }

    /// U34: Opus 4.5 carries a higher floor (4096); the resolution must pick
    /// it up via the model id, including the dated snapshot variant.
    #[test]
    fn min_tokens_resolution_opus_4_5_floor_is_4096() {
        assert_eq!(
            lookup_pricing("claude-opus-4-5")
                .expect("priced")
                .prompt_cache_min_tokens,
            Some(4096),
        );
        let dep = make_dep("claude-opus-4-5", "anthropic/claude-opus-4-5");
        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), None);
        assert_eq!(c.min_tokens_for(&dep), 4096);

        // The dated snapshot variant resolves to the same floor.
        let dep_dated = make_dep("claude-opus-4-5", "anthropic/claude-opus-4-5-20251101");
        assert_eq!(c.min_tokens_for(&dep_dated), 4096);
    }

    /// U34: an unmodelled model with no override falls all the way through to
    /// the global default, then DEFAULT_MIN_TOKENS.
    #[test]
    fn min_tokens_resolution_unknown_model_uses_global_then_default() {
        let dep = make_dep("totally-unknown-model-xyz", "totally-unknown-model-xyz");
        assert!(lookup_pricing("totally-unknown-model-xyz").is_none());

        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), None);
        assert_eq!(c.min_tokens_for(&dep), DEFAULT_MIN_TOKENS);

        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), Some(8192));
        assert_eq!(c.min_tokens_for(&dep), 8192);
    }

    #[test]
    fn engages_above_threshold_and_not_below() {
        let c = check();
        let dep = make_dep("gpt-4o", "openai/gpt-4o");
        assert!(c.engages(&dep, u64::from(DEFAULT_MIN_TOKENS)));
        assert!(c.engages(&dep, u64::from(DEFAULT_MIN_TOKENS) + 500));
        assert!(!c.engages(&dep, u64::from(DEFAULT_MIN_TOKENS) - 1));
        assert!(!c.engages(&dep, 0));
    }

    #[test]
    fn engages_globally_empty_group_falls_back_to_global() {
        // With no deployments to consult, the gate fails open to the global
        // default (or DEFAULT_MIN_TOKENS when the global is unset). This is
        // the degenerate path (e.g. an unknown model group); real groups use
        // the per-deployment minimum via engages_globally_uses_group_minimum.
        let c = check();
        assert!(c.engages_globally(
            u64::from(DEFAULT_MIN_TOKENS),
            std::iter::empty::<&Deployment>()
        ),);
        assert!(!c.engages_globally(
            u64::from(DEFAULT_MIN_TOKENS) - 1,
            std::iter::empty::<&Deployment>(),
        ));

        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), Some(2048));
        assert!(
            !c.engages_globally(1024, std::iter::empty::<&Deployment>()),
            "empty group falls back to the global default (2048)"
        );
        assert!(c.engages_globally(2048, std::iter::empty::<&Deployment>()),);
    }

    /// C13/MEDIUM: the read gate must use the **minimum** resolved
    /// `prompt_cache_min_tokens` across the model group's deployments, not the
    /// bare global default. An operator who sets `prompt_caching_min_tokens:
    /// 4096` to accommodate an Opus-4.5 deployment must NOT silently disable
    /// cache consultation for a Sonnet (1024) group: the group minimum
    /// resolves to 1024 and the lookup engages at 1024 tokens. Mirrors
    /// the reference `_get_min_token_count_for_deployments`.
    #[test]
    fn engages_globally_uses_group_minimum_not_global() {
        // claude-sonnet-4-20250514 carries a 1024 pricing floor.
        let sonnet = make_dep(
            "claude-sonnet-4-20250514",
            "anthropic/claude-sonnet-4-20250514",
        );
        assert_eq!(
            lookup_pricing("anthropic/claude-sonnet-4-20250514")
                .expect("priced")
                .prompt_cache_min_tokens,
            Some(1024),
        );
        // claude-opus-4-5 carries a 4096 pricing floor.
        let opus = make_dep("claude-opus-4-5", "anthropic/claude-opus-4-5");
        assert_eq!(
            lookup_pricing("claude-opus-4-5")
                .expect("priced")
                .prompt_cache_min_tokens,
            Some(4096),
        );

        // Global default deliberately set HIGH (4096) — the Sonnet member's
        // 1024 floor must still let the lookup engage at 1024 tokens.
        let c = PromptCachingCheck::new(Arc::new(InMemoryCache::new(1_000)), Some(4096));
        let group = [sonnet, opus];
        assert!(
            c.engages_globally(1024, group.iter().collect::<Vec<_>>()),
            "group minimum (1024) must engage the lookup even when the global default is 4096"
        );
        assert!(c.engages_globally(4096, group.iter().collect::<Vec<_>>()),);
        assert!(
            !c.engages_globally(1023, group.iter().collect::<Vec<_>>()),
            "below the group minimum (1024) the lookup must not engage"
        );
    }

    /// C13/MEDIUM: a high per-deployment override on one group member must not
    /// suppress the lookup — the min across the group governs, so a sibling
    /// with a lower resolved floor keeps the gate open.
    #[test]
    fn engages_globally_ignores_per_deployment_override() {
        let mut low = make_dep("gpt-4o", "openai/gpt-4o");
        low.oxllm_params.prompt_caching_min_tokens = Some(1024);
        let mut high = make_dep("gpt-4o", "openai/gpt-4o");
        high.oxllm_params.prompt_caching_min_tokens = Some(8192);
        let group = [low, high];
        let c = check();
        assert!(
            c.engages_globally(1024, group.iter().collect::<Vec<_>>()),
            "low-override sibling keeps the gate at 1024"
        );
        assert!(!c.engages_globally(1023, group.iter().collect::<Vec<_>>()),);
    }

    #[test]
    fn large_prompt_estimate_exceeds_default_threshold() {
        let c = check();
        let dep = make_dep("gpt-4o", "openai/gpt-4o");
        let (_, tokens) = c
            .prompt_hash(&big_prompt_body(2_000), "gpt-4o")
            .expect("prefix present");
        assert!(
            c.engages(&dep, tokens),
            "a ~2000-token system prompt must engage the check (est={tokens})"
        );
    }

    // ------------------------------------------------------------------
    // Pin read/write
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn preferred_deployment_none_when_no_cache_entry() {
        let c = check();
        assert!(c.preferred_deployment(42, "tenant-a").await.is_none());
    }

    #[tokio::test]
    async fn pin_then_preferred_returns_deployment() {
        let c = check();
        c.pin(7, "tenant-a", "gpt-4o:1").await;
        assert_eq!(
            c.preferred_deployment(7, "tenant-a").await.as_deref(),
            Some("gpt-4o:1"),
            "post-call mapping must be visible to the pre-call lookup"
        );
        assert!(
            c.preferred_deployment(8, "tenant-a").await.is_none(),
            "other hashes stay unmapped"
        );
    }

    #[tokio::test]
    async fn pins_are_scoped_by_api_key_hash() {
        // Cross-tenant isolation: a pin written under one key's hash must
        // not be visible under another key's hash, even for the same
        // prompt hash.
        let c = check();
        c.pin(7, "tenant-a", "gpt-4o:1").await;
        assert!(
            c.preferred_deployment(7, "tenant-b").await.is_none(),
            "tenant-b must not see tenant-a's pin"
        );
        c.pin(7, "tenant-b", "gpt-4o:0").await;
        assert_eq!(
            c.preferred_deployment(7, "tenant-a").await.as_deref(),
            Some("gpt-4o:1"),
            "tenant-a's pin is unchanged by tenant-b's write"
        );
        assert_eq!(
            c.preferred_deployment(7, "tenant-b").await.as_deref(),
            Some("gpt-4o:0"),
        );
    }

    /// C13/LOW: the pin TTL must match the reference 300s (`ttl=300` in
    /// `PromptCachingCache.async_set_cache`), not the 3600s the constant held
    /// before this fix. A regression here silently holds stale pins 12×
    /// longer than upstream, directing traffic at deployments that may have
    /// been removed, cooled, or rotated.
    #[test]
    fn pin_ttl_matches_litellm_300s() {
        assert_eq!(
            PIN_TTL,
            Duration::from_secs(300),
            "PIN_TTL must stay aligned with LiteLLM's 300s"
        );
    }

    /// C13/LOW: a pin whose cache entry has expired must read as absent from
    /// `preferred_deployment`. The pin is written through the same shared
    /// cache, so the read path must honour the entry's TTL. We cannot wait
    /// out the full 300s `PIN_TTL` in a unit test, so this writes a pin-shaped
    /// entry (same key format) directly to the cache with a tiny TTL and
    /// asserts the read observes its expiry — pinning the read-honours-TTL
    /// contract that `PIN_TTL`'s value relies on.
    #[tokio::test]
    async fn pin_expires_after_ttl() {
        let cache = Arc::new(InMemoryCache::new(1_000));
        let c = PromptCachingCheck::new(cache.clone(), None);
        let hash = 7u128;
        let key = cache_key("tenant-a", hash);
        cache
            .set(
                &key,
                "gpt-4o:1".to_string(),
                Some(Duration::from_millis(50)),
            )
            .await;
        assert_eq!(
            c.preferred_deployment(hash, "tenant-a").await.as_deref(),
            Some("gpt-4o:1"),
            "pin is visible before its TTL elapses"
        );
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(
            c.preferred_deployment(hash, "tenant-a").await.is_none(),
            "pin must be invisible once its TTL has elapsed"
        );
    }

    /// C13/LOW: `hash128` is SHA-256 truncated to 128 bits (big-endian). Lock
    /// the algorithm with a known input vector so a future change to a
    /// different hash function (or an endianness flip) is caught. SHA-256 of
    /// the empty string is `e3b0c442...`; SHA-256 of `"abc"` is
    /// `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`,
    /// whose first 16 bytes (128 bits) as a big-endian u128 are
    /// `0xba7816bf8f01cfea414140de5dae2223`.
    #[test]
    fn hash128_known_vector() {
        assert_eq!(
            hash128("abc"),
            0xba78_16bf_8f01_cfea_4141_40de_5dae_2223u128,
        );
        // Cross-check against a direct `sha2` computation so the helper is
        // provably a SHA-256 truncation (not just a magic constant match).
        let direct = {
            let digest = Sha256::digest(b"abc");
            let mut buf = [0u8; 16];
            buf.copy_from_slice(&digest[..16]);
            u128::from_be_bytes(buf)
        };
        assert_eq!(hash128("abc"), direct);
        // Empty input — must be the truncated SHA-256 of "".
        let direct_empty = {
            let digest = Sha256::digest(b"");
            let mut buf = [0u8; 16];
            buf.copy_from_slice(&digest[..16]);
            u128::from_be_bytes(buf)
        };
        assert_eq!(hash128(""), direct_empty);
    }

    /// C13/LOW + C13/MEDIUM: the cache key must fold in the request's `tools`
    /// array. Two requests with identical system prompts but different tool
    /// sets must produce distinct pins — the provider-side cache key covers
    /// tool definitions, so a shared pin would route to a deployment whose
    /// cache is cold for the actual (tool-bearing) request.
    #[test]
    fn prompt_hash_distinguishes_different_tool_sets() {
        let c = check();
        let tools_a = json!([{"type": "function", "function": {"name": "get_weather"}}]);
        let tools_b = json!([
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "search_web"}}
        ]);
        let base = json!({
            "messages": [{"role": "system", "content": "shared system prompt"}]
        });
        let with_a = {
            let mut b = base.clone();
            b["tools"] = tools_a.clone();
            b
        };
        let with_b = {
            let mut b = base.clone();
            b["tools"] = tools_b.clone();
            b
        };
        let hash_none = c.prompt_hash(&base, "gpt-4o").map(|(h, _)| h);
        let hash_a = c.prompt_hash(&with_a, "gpt-4o").map(|(h, _)| h);
        let hash_b = c.prompt_hash(&with_b, "gpt-4o").map(|(h, _)| h);
        let (hash_none, hash_a, hash_b) = (hash_none, hash_a, hash_b);
        assert_ne!(hash_none, hash_a, "no-tools vs tools-a must differ");
        assert_ne!(hash_none, hash_b, "no-tools vs tools-b must differ");
        assert_ne!(
            hash_a, hash_b,
            "different tool sets must produce different pins"
        );
    }

    /// C13/LOW: `cache_key` renders the 128-bit hash as 32 hex digits so the
    /// wider hash space is fully represented in the cache key (vs the old
    /// 16-digit FNV-1a rendering).
    #[test]
    fn cache_key_renders_128_bit_hash_as_32_hex_digits() {
        let key = cache_key("tenant", 0xba78_16bf_8f01_cfea_4141_40de_5dae_2223u128);
        assert_eq!(key, "pcache:tenant:ba7816bf8f01cfea414140de5dae2223");
    }

    // ------------------------------------------------------------------
    // PreCallCheck::filter_deployments
    // ------------------------------------------------------------------

    use oxllm_core::RequestContext;
    use oxllm_core::router::PreCallCheck;

    fn ctx_with_pin(pin: Option<&str>, keys: &[&str]) -> RequestContext {
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.pre_call.pinned_deployment_key = pin.map(String::from);
        ctx.pre_call.candidate_keys = keys.iter().map(|s| s.to_string()).collect();
        ctx
    }

    #[test]
    fn filter_narrows_to_pinned_deployment() {
        let c = check();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let d2 = make_dep("gpt-4o", "openai/gpt-4o");
        let candidates: Vec<&Deployment> = vec![&d1, &d2];
        let ctx = ctx_with_pin(Some("gpt-4o:1"), &["gpt-4o:0", "gpt-4o:1"]);
        let out = PreCallCheck::filter_deployments(&c, &candidates, &ctx).expect("infallible");
        assert_eq!(out.len(), 1);
        assert!(
            std::ptr::eq(out[0], &d2),
            "the pinned deployment must be the only candidate"
        );
    }

    #[test]
    fn filter_passes_through_without_pin() {
        let c = check();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let d2 = make_dep("gpt-4o", "openai/gpt-4o");
        let candidates: Vec<&Deployment> = vec![&d1, &d2];
        // No pin resolved (e.g. prompt below threshold, or cache miss).
        let ctx = ctx_with_pin(None, &["gpt-4o:0", "gpt-4o:1"]);
        let out = PreCallCheck::filter_deployments(&c, &candidates, &ctx).expect("infallible");
        assert_eq!(out.len(), 2, "no pin → all candidates stay eligible");
    }

    #[test]
    fn filter_passes_through_when_pinned_not_a_candidate() {
        // The pinned deployment is cooling or outside the model group:
        // candidates (already cooldown-filtered by the router) do not
        // contain it, so the filter must leave the list unchanged.
        let c = check();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let candidates: Vec<&Deployment> = vec![&d1];
        let ctx = ctx_with_pin(Some("gpt-4o:1"), &["gpt-4o:0"]);
        let out = PreCallCheck::filter_deployments(&c, &candidates, &ctx).expect("infallible");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn filter_passes_through_on_misaligned_keys() {
        let c = check();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let candidates: Vec<&Deployment> = vec![&d1];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.pre_call.pinned_deployment_key = Some("gpt-4o:0".to_string());
        // No candidate keys (host bypassed the router pipeline) → fail open.
        let out = PreCallCheck::filter_deployments(&c, &candidates, &ctx).expect("infallible");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn prompt_caching_name_matching() {
        assert!(is_prompt_caching_name("prompt_caching"));
        assert!(is_prompt_caching_name("prompt-caching"));
        assert!(!is_prompt_caching_name("budget_limiter"));
        assert_eq!(PreCallCheck::name(&check()), "prompt_caching");
    }
}
