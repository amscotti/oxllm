//! The routing strategy trait and request context.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use crate::deployment::Deployment;
use crate::router::adaptive::{AdaptiveRouter, BanditStore, RequestType};
use crate::router::complexity::{ComplexityRouter, Tier};
use crate::router::latency_based::LatencyBased;
use crate::router::least_busy::LeastBusy;
use crate::router::lowest_cost::LowestCost;
use crate::router::lowest_latency::LowestLatency;
use crate::router::quality::QualityRouter;
use crate::router::round_robin::RoundRobin;
use crate::router::session_affinity::SessionAffinity;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::stores::{LatencyStore, UsageStore};
use crate::router::tag_based::TagBased;
use crate::router::types::{AdaptiveRouterConfig, ComplexityRouterConfig, RoutingStrategyKind};
use crate::router::usage_based::UsageBased;

/// Context passed to the routing strategy and router for each request.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// The model name requested by the client.
    pub model: String,
    /// A hash of the API key (never the raw key).
    pub api_key_hash: String,
    /// A unique request identifier.
    pub request_id: String,
    /// Optional routing tag. When the `tag_based` strategy is selected,
    /// deployments are filtered/matched by this value. `None` falls back to
    /// `SimpleShuffle`. Carries the request's PRIMARY tag (the first of
    /// [`Self::tags`]) for single-tag compatibility.
    pub tag: Option<String>,
    /// All request tags (the common proxy convention: `metadata.tags` in the
    /// request body, plus any `x-oxllm-tag` header values, comma-separated). The
    /// `tag_based` strategy matches deployments against the full set:
    /// positive tags are subset-matched, `&tag` requires a deployment tag,
    /// `!tag` excludes deployments carrying it. [`Self::tag`] stays the
    /// first element so single-tag call sites behave unchanged.
    pub tags: Vec<String>,
    /// Optional session id for sticky routing. When the `session_affinity`
    /// strategy is selected, the same id consistently maps to the same
    /// deployment (via a stable hash). Typically plumbed from the
    /// `x-oxllm-session-id` request header. `None` falls back to
    /// `SimpleShuffle`.
    pub session_id: Option<String>,
    /// Optional request text (concatenated message contents, capped in
    /// length) used by the `complexity_router` strategy to classify request
    /// complexity. Populated by the host from the request body; `None` is
    /// classified as an empty (simple) request.
    pub request_text: Option<String>,
    /// Per-request data consumed by [`super::PreCallCheck`] implementations.
    /// Fields are populated either by the router itself (candidate keys) or
    /// by the host ahead of the pick (pin lookups, spend snapshots) so the
    /// checks themselves stay synchronous and pure.
    pub pre_call: PreCallData,
}

/// Per-request inputs for pre-call checks.
///
/// The router fills [`PreCallData::candidate_keys`] immediately before
/// invoking each check (keys are aligned by position with the `candidates`
/// slice passed to the check). The host fills the remaining fields when the
/// corresponding check is enabled; checks treat missing data as "no
/// opinion" and pass candidates through unchanged (fail-open).
#[derive(Debug, Clone, Default)]
pub struct PreCallData {
    /// Stable cooldown keys ([`super::Router::cooldown_key`] =
    /// [`super::stable_key`], a hash of the sanitized `oxllm_params` plus a
    /// keyed per-process digest of the credential fields)
    /// for the candidate deployments, aligned by position with the
    /// `candidates` slice a check receives. Populated by the router before
    /// each check runs; checks whose state is keyed by deployment identity
    /// (health, prompt-caching pins) match against these. This is the same
    /// key space the prompt-caching pin write side uses, so a pin written
    /// for one leg is observable by the next — *not* the old positional
    /// `{model_name}:{index}` scheme.
    pub candidate_keys: Vec<String>,
    /// Deployment identity fingerprints (`deployment_key` =
    /// `"{model_name}:{provider_model}"`), position-aligned with
    /// [`PreCallData::candidate_keys`]. Populated by the router alongside
    /// the keys so consumers (see `super::strategy::candidate_keys`) can
    /// verify the cached keys were computed for *these* deployments rather
    /// than merely trusting a length match — a stale same-length key list
    /// (e.g. a reused `RequestContext` across standalone picks with a
    /// different candidate set) would otherwise silently mis-key bandit
    /// cells and usage counters.
    pub candidate_ids: Vec<String>,
    /// The cooldown key of the deployment this request's prompt is pinned
    /// to, pre-resolved by the host (async cache lookup). Consumed by the
    /// `prompt_caching` pre-call check.
    pub pinned_deployment_key: Option<String>,
    /// Deployments rejected by request-local admission (for example, a pricing
    /// override this caller cannot afford). Excluding them must not cool the
    /// deployment globally for unrelated callers.
    pub excluded_deployment_keys: Vec<String>,
    /// Current spend per budget scope key (`spend:dep:{deployment_key}`,
    /// `spend:provider:{slug}`), pre-resolved by the host (async counter
    /// reads). Consumed by the `budget_limiter` pre-call check; a missing
    /// key reads as zero spend (under any cap).
    pub spend_snapshot: Arc<HashMap<String, f64>>,
    /// An optional pre-computed complexity classification for standalone
    /// `complexity_router` strategy picks. `None` classifies on demand.
    /// The router's candidate-expansion path reuses its own classification
    /// directly when picking from the resolved pool.
    pub complexity_classification: Option<(Tier, f64)>,
    /// The request-type classification for the `adaptive_router` strategy,
    /// computed **once per request** — by the host when it builds the
    /// request context (it needs the type to stamp the bandit cell for the
    /// learning hook), or by [`super::Router::pick_deployment`] when the
    /// host did not. The adaptive strategy reuses it instead of
    /// classifying a second time; `None` means the pick classifies on
    /// demand (standalone strategy use).
    pub adaptive_request_type: Option<RequestType>,
    /// The normalized model-group key for this request's candidate group:
    /// the requested model when it resolves by exact `model_name` match or
    /// configured alias, otherwise the wildcard group name `"*"`. Populated
    /// by [`super::Router::pick_deployment`] before the strategy pick;
    /// strategies that keep per-group state keyed by the request's model
    /// (`round_robin`) must use this instead of the raw
    /// [`RequestContext::model`] string — a wildcard deployment resolves
    /// every distinct client-requested model name, so keying state by the
    /// raw string would grow it without bound off client input. `None`
    /// (standalone strategy use) falls back to `RequestContext::model`.
    pub model_group_key: Option<String>,
}

impl RequestContext {
    /// Create a new context with the given model and empty auth/request IDs.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key_hash: String::new(),
            request_id: String::new(),
            tag: None,
            tags: Vec::new(),
            session_id: None,
            request_text: None,
            pre_call: PreCallData::default(),
        }
    }
}

/// A routing strategy picks one deployment from a list of candidates.
///
/// Implementations include [`crate::router::SimpleShuffle`] (default),
/// [`crate::router::LowestCost`], [`crate::router::LowestLatency`],
/// [`crate::router::UsageBased`], [`crate::router::LatencyBased`],
/// [`crate::router::QualityRouter`], [`crate::router::TagBased`],
/// [`crate::router::LeastBusy`], [`crate::router::SessionAffinity`],
/// [`crate::router::ComplexityRouter`], and [`crate::router::AdaptiveRouter`].
pub trait RoutingStrategy: Send + Sync {
    /// Pick one deployment from the provided slice of available deployments.
    ///
    /// Returns `None` if the slice is empty or request constraints cannot be
    /// satisfied. Use [`Self::try_pick`] to distinguish a rejected selector
    /// from an unavailable deployment.
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment>;

    /// Pick with an explicit error when request constraints cannot be
    /// satisfied. The default preserves existing strategies' `Option` API;
    /// constraint-aware strategies override this to prevent failover from
    /// restoring a forbidden candidate pool.
    fn try_pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> crate::Result<Option<&'a Deployment>> {
        Ok(self.pick(deployments, ctx))
    }
}

/// Shared dependencies a strategy factory may need: the router-wide stores
/// and the config slices strategies are built from. Grouped into one
/// struct so registering a new strategy never changes the registry's
/// call sites.
pub struct StrategyDeps<'a> {
    /// Quality scores per deployment model (`QualityRouter`).
    pub quality_scores: &'a HashMap<String, f64>,
    /// Shared latency sample store (`LowestLatency`, `LatencyBased`).
    pub latency_store: &'a Arc<LatencyStore>,
    /// Shared usage / in-flight store (`UsageBased`, `LeastBusy`; adaptive
    /// cold start).
    pub usage_store: &'a Arc<UsageStore>,
    /// Complexity-router configuration (`ComplexityRouter`).
    pub complexity_config: &'a Option<ComplexityRouterConfig>,
    /// Adaptive-router configuration (`AdaptiveRouter`).
    pub adaptive_config: &'a Option<AdaptiveRouterConfig>,
    /// Shared bandit cell store (`AdaptiveRouter`); the server's learning
    /// hook holds a clone of the same `Arc` and writes post-call outcomes.
    pub bandit_store: &'a Arc<BanditStore>,
    /// Shared per-model-group monotonic counters for the `RoundRobin`
    /// strategy, keyed by the request's normalized model-group key (see
    /// [`PreCallData::model_group_key`]; falls back to the requested model
    /// name when the router did not set it) and lazily populated on the
    /// first pick for a group. Built once per `Router` and preserved
    /// across `Router::rebuild_from_stores` (exactly like the other stores)
    /// so a config hot-reload does not reset the strict-rotation position
    /// back to `deployments[0]`. Per-group counters (rather than a single
    /// shared counter) keep strict alternation within each group regardless
    /// of cross-group interleaving — critical for the documented
    /// load-testing / deterministic-debug-trace use cases.
    pub round_robin_counters: &'a Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
}

/// A strategy factory: builds a boxed routing strategy from shared deps.
pub type StrategyFactory = for<'a> fn(&StrategyDeps<'a>) -> Box<dyn RoutingStrategy>;

/// The stable cooldown keys for `deployments`, aligned by position: the
/// per-pick keys the router already memoized
/// ([`PreCallData::candidate_keys`]) when they were computed for exactly
/// these deployments — verified per-position against the
/// [`PreCallData::candidate_ids`] fingerprints (`deployment_key`), not
/// merely trusted on a length match. A different candidate list of the
/// same length therefore falls back to fresh computation instead of
/// silently mis-keying store lookups (bandit cells, usage counters).
/// Store-backed strategies consume these instead of re-serializing every
/// candidate's `oxllm_params` on each pick — the router computes each
/// candidate's key once per `pick_deployment` call
/// ([`super::Router::pick_deployment`]), so recomputing them inside the
/// strategy is pure redundancy on the hot path.
pub(crate) fn candidate_keys(deployments: &[&Deployment], ctx: &RequestContext) -> Vec<String> {
    let ids = &ctx.pre_call.candidate_ids;
    if ids.len() == deployments.len()
        && ids.len() == ctx.pre_call.candidate_keys.len()
        && ids
            .iter()
            .zip(deployments.iter())
            .all(|(id, d)| *id == super::stores::deployment_key(d))
    {
        return ctx.pre_call.candidate_keys.clone();
    }
    deployments
        .iter()
        .map(|d| super::stores::stable_key(d))
        .collect()
}

/// The strategy registry: maps each [`RoutingStrategyKind`] to the factory
/// that builds it. Registering a new strategy means adding its
/// [`RoutingStrategyKind`] variant plus one arm here — router
/// construction needs no changes.
pub fn strategy_factory(kind: RoutingStrategyKind) -> StrategyFactory {
    match kind {
        RoutingStrategyKind::SimpleShuffle => |_| Box::new(SimpleShuffle),
        RoutingStrategyKind::LowestCost => |_| Box::new(LowestCost),
        RoutingStrategyKind::LowestLatency => {
            |d| Box::new(LowestLatency::new(d.latency_store.clone()))
        }
        RoutingStrategyKind::UsageBased => |d| Box::new(UsageBased::new(d.usage_store.clone())),
        RoutingStrategyKind::LatencyBased => {
            |d| Box::new(LatencyBased::new(d.latency_store.clone()))
        }
        RoutingStrategyKind::QualityRouter => {
            |d| Box::new(QualityRouter::new(d.quality_scores.clone()))
        }
        RoutingStrategyKind::TagBased => |_| Box::new(TagBased::new()),
        RoutingStrategyKind::LeastBusy => |d| Box::new(LeastBusy::new(d.usage_store.clone())),
        RoutingStrategyKind::SessionAffinity => |_| Box::new(SessionAffinity::new()),
        RoutingStrategyKind::ComplexityRouter => |d| {
            Box::new(ComplexityRouter::new(
                d.complexity_config.clone().unwrap_or_default(),
            ))
        },
        RoutingStrategyKind::AdaptiveRouter => |d| {
            Box::new(AdaptiveRouter::new(
                d.adaptive_config.clone().unwrap_or_default(),
                d.bandit_store.clone(),
                d.usage_store.clone(),
            ))
        },
        RoutingStrategyKind::RoundRobin => {
            |d| Box::new(RoundRobin::with_counters(d.round_robin_counters.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::{Deployment, OxllmParams};

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

    #[test]
    fn candidate_keys_trusted_only_when_ids_match() {
        // A reused RequestContext whose cached candidate_keys/ids came from a
        // DIFFERENT candidate list of the same length must not leak those
        // stale keys into the pick: the per-position identity fingerprint
        // (deployment_key) fails to match, so the keys are recomputed.
        let stale = [make_dep("other", "openai/x"), make_dep("other", "openai/y")];
        let fresh = [make_dep("m", "openai/a"), make_dep("m", "openai/b")];

        let mut ctx = RequestContext::new("m");
        ctx.pre_call.candidate_keys = vec!["stale-1".to_string(), "stale-2".to_string()];
        ctx.pre_call.candidate_ids = stale
            .iter()
            .map(crate::router::stores::deployment_key)
            .collect();

        let refs: Vec<&Deployment> = fresh.iter().collect();
        let keys = candidate_keys(&refs, &ctx);
        assert_eq!(
            keys,
            refs.iter()
                .map(|d| crate::router::stores::stable_key(d))
                .collect::<Vec<_>>(),
            "same-length but foreign candidate list must fall back to fresh keys"
        );

        // With matching ids the cached keys are reused verbatim.
        ctx.pre_call.candidate_ids = refs
            .iter()
            .map(|d| crate::router::stores::deployment_key(d))
            .collect();
        assert_eq!(candidate_keys(&refs, &ctx), ctx.pre_call.candidate_keys);
    }
}
