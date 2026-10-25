//! The adaptive router: a Thompson-sampling multi-armed bandit that learns
//! which deployment is best for each request type, from post-call outcomes.
//!
//! One **Beta cell per (request type, deployment key)**
//! ([`BanditStore`]); a pick samples `Beta(alpha, beta)` for every
//! candidate cell and takes the argmax, optionally blended with a
//! normalized cost term: `score = sample * quality_weight −
//! normalized_cost * cost_weight`. Successes increment `alpha`, failures
//! increment `beta` — updates arrive from the server's learning hook
//! (`oxllm-server`'s `adaptive_hook`), never from the strategy itself
//! (strategies stay pure and synchronous).
//!
//! Deployment identity is the router's **cooldown key** — the stable
//! `{model_name}:{16-hex hash}` key
//! ([`Router::cooldown_key`](crate::router::Router::cooldown_key) =
//! [`stable_key`](crate::router::stable_key), a hash of the sanitized
//! `oxllm_params` plus a keyed credential digest; the positional
//! `{model_name}:{index}` scheme was removed, and the
//! `{model_name}:{provider_model}` spelling is the *separate*
//! `deployment_key` fingerprint) — the same key
//! `PreCallData::candidate_keys` carries, so the server hook re-derives the
//! cell at pick time without a second mapping table. [`Router::pick_deployment`](crate::router::Router::pick_deployment)
//! populates the candidate keys and the (once-per-request) request-type
//! classification on [`PreCallData`](crate::router::PreCallData) before
//! invoking this strategy;
//! standalone use classifies on demand and falls back to store-style
//! deployment keys.
//!
//! Cold start (every cell near its `Beta(1, 1)` prior) would spray traffic
//! randomly, so until **every** candidate cell has at least
//! `cold_start_min_observations` observations the pick defers to
//! [`LeastBusy`] over the shared [`UsageStore`] — an existing, well-tested
//! signal with sane behavior from the first request. Once warm, picks are
//! epsilon-greedy Thompson sampling: with probability `epsilon` a uniform
//! random candidate is chosen (exploration), otherwise the sampled scores
//! decide (exploitation).
//!
//! Cross-replica sync: the store is in-memory, matching every other router
//! store (cooldown, latency, usage, health — none of them sync today).
//! [`BanditStore::merge_delta`] is the documented extension point for a
//! future delta-sync: alpha/beta deltas commute, so replicas may merge
//! them in any order without losing counts.
//!
//! Non-chat pick sites (embeddings, passthrough, realtime) go through the
//! same [`Router::pick_deployment`](crate::router::Router::pick_deployment)
//! but with a bare request context: their requests classify as
//! [`RequestType::Other`] on demand, and because only the chat
//! orchestration stamps bandit cells for the learning hook, their outcomes
//! never reach the store. When the adaptive strategy is configured
//! globally, those endpoints therefore ride the least-busy cold start
//! indefinitely (or sample the uniform prior when
//! `cold_start_min_observations` is 0) — a deliberate safe degradation:
//! they neither pollute chat's learned posteriors nor starve themselves.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rand::Rng;

use crate::cost::{deployment_pricing_model, lookup_pricing_with_provider};
use crate::deployment::Deployment;
use crate::router::complexity::TextMatcher;
use crate::router::least_busy::LeastBusy;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::stores::UsageStore;
use crate::router::strategy::{RequestContext, RoutingStrategy, candidate_keys};
use crate::router::types::AdaptiveRouterConfig;

// ---------------------------------------------------------------------------
// Request-type classification
// ---------------------------------------------------------------------------

/// The request type a prompt is classified into. Bandit cells are keyed by
/// `(RequestType, deployment key)`, so each type learns its own per-
/// deployment posterior. Keyword/regex classification only — no LLM call on
/// the hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RequestType {
    /// Programming tasks: code fences, languages, debugging, APIs.
    Code,
    /// Proof-shaped or deep-reasoning requests.
    Reasoning,
    /// Stories, poems, lyrics, fiction, and other creative writing.
    Creative,
    /// Factual question answering ("what is", "who was", "how does", ...).
    QA,
    /// Summarization requests ("summarize", "tl;dr", "key points", ...).
    Summarization,
    /// Translation requests ("translate", "in french", "to english", ...).
    Translation,
    /// Anything that matches no keyword table (the fallback type).
    Other,
}

impl RequestType {
    /// The config/tracing name of the request type (`"code"`,
    /// `"reasoning"`, ...) — for tracing fields, metric labels, and JSON.
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestType::Code => "code",
            RequestType::Reasoning => "reasoning",
            RequestType::Creative => "creative",
            RequestType::QA => "qa",
            RequestType::Summarization => "summarization",
            RequestType::Translation => "translation",
            RequestType::Other => "other",
        }
    }
}

/// Code signals: symbol markers (punctuation-bearing, raw-substring
/// matched) plus word-ish markers (`def`, `fn` — word-boundary matched).
/// The bare-English pitfalls from the complexity classifier apply here too
/// — words like "from"/"class" are excluded; distinctive shapes (a
/// language name, "stack trace", a code fence) carry the signal.
const CODE_SYMBOL_MARKERS: &[&str] = &["```", "def", "fn", "#include", "console.log", "println"];

/// Word-ish code markers (word-boundary matched).
const CODE_WORD_MARKERS: &[&str] = &[
    "code",
    "coding",
    "program",
    "programming",
    "function",
    "debug",
    "debugger",
    "compile",
    "compiler",
    "syntax",
    "stack trace",
    "exception",
    "python",
    "javascript",
    "typescript",
    "rust",
    "golang",
    "sql",
    "regex",
    "refactor",
    "algorithm",
    "docker",
    "kubernetes",
];

/// Reasoning markers: proofs, derivations, math, step-by-step work.
const REASONING_MARKERS: &[&str] = &[
    "prove",
    "proof",
    "derive",
    "theorem",
    "lemma",
    "step by step",
    "step-by-step",
    "reasoning",
    "logic",
    "logical",
    "deduce",
    "solve",
    "calculate",
    "equation",
    "integral",
    "derivative",
    "probability",
    "algebra",
    "explain why",
    "why does",
];

/// Creative-writing markers.
const CREATIVE_MARKERS: &[&str] = &[
    "story",
    "poem",
    "poetry",
    "haiku",
    "sonnet",
    "song",
    "lyrics",
    "fiction",
    "novel",
    "screenplay",
    "imagine",
    "character",
    "plot",
    "tale",
    "joke",
    "rhyme",
    "creative",
];

/// Question-answering shapes: factual lookups and how-to questions.
const QA_MARKERS: &[&str] = &[
    "what is",
    "what are",
    "what was",
    "what does",
    "who is",
    "who was",
    "who are",
    "when did",
    "where is",
    "where are",
    "define",
    "definition",
    "meaning of",
    "tell me about",
    "how do i",
    "how to",
    "how does",
    "how did",
];

/// Summarization markers.
const SUMMARIZATION_MARKERS: &[&str] = &[
    "summarize",
    "summarise",
    "summary",
    "tl;dr",
    "tldr",
    "condense",
    "recap",
    "key points",
    "main points",
    "bullet points",
    "digest",
    "shorten",
];

/// Translation markers. Bare language names are deliberately excluded
/// ("french toast" is not a translation request); the phrase shapes
/// "translate", "in <language>", "to/into english" carry the signal.
const TRANSLATION_MARKERS: &[&str] = &[
    "translate",
    "translation",
    "translated",
    "in french",
    "in spanish",
    "in german",
    "in italian",
    "in portuguese",
    "in japanese",
    "in chinese",
    "in korean",
    "in russian",
    "in arabic",
    "in hindi",
    "in dutch",
    "to english",
    "into english",
    "from english",
];

/// Classify a request text into a [`RequestType`] from keyword hits (no LLM
/// call). The type with the most marker hits wins; ties resolve to the
/// earliest type in [`RequestType`] declaration order (deterministic), and
/// zero hits classify as [`RequestType::Other`].
///
/// Matching reuses the complexity classifier's `TextMatcher`: case-
/// insensitive, punctuation-stripped, word-boundary matching for word-ish
/// markers and raw substring matching for symbol markers.
pub fn classify_request_type(text: &str) -> RequestType {
    if text.is_empty() {
        return RequestType::Other;
    }
    let matcher = TextMatcher::new(text);
    let hits = |markers: &[&str]| markers.iter().filter(|m| matcher.matches(m)).count();
    let code = hits(CODE_SYMBOL_MARKERS) + hits(CODE_WORD_MARKERS);
    // Declaration order is the tie-break order: the first type with the
    // maximal hit count wins (strict `>` keeps the earliest type on ties;
    // zero hits fall through to Other).
    let scored = [
        (RequestType::Code, code),
        (RequestType::Reasoning, hits(REASONING_MARKERS)),
        (RequestType::Creative, hits(CREATIVE_MARKERS)),
        (RequestType::QA, hits(QA_MARKERS)),
        (RequestType::Summarization, hits(SUMMARIZATION_MARKERS)),
        (RequestType::Translation, hits(TRANSLATION_MARKERS)),
    ];
    let mut best: Option<(RequestType, usize)> = None;
    for (ty, count) in scored {
        if count > best.map(|(_, best_count)| best_count).unwrap_or(0) {
            best = Some((ty, count));
        }
    }
    best.map(|(ty, _)| ty).unwrap_or(RequestType::Other)
}

// ---------------------------------------------------------------------------
// Bandit state
// ---------------------------------------------------------------------------

/// A Beta-distribution cell: the posterior for one
/// `(request type, deployment key)` arm. Starts at the uniform prior
/// `Beta(1, 1)`; successes increment `alpha`, failures increment `beta`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaCell {
    /// Success count plus the prior's pseudo-success (`1 + successes`).
    pub alpha: f64,
    /// Failure count plus the prior's pseudo-failure (`1 + failures`).
    pub beta: f64,
}

impl Default for BetaCell {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl BetaCell {
    /// Recorded outcomes (`alpha − 1 + beta − 1`, clamped at 0): the cold-
    /// start check compares this against
    /// [`AdaptiveRouterConfig::cold_start_min_observations`]. Exposed so the
    /// management `/routing/state` snapshot and any future observer reuse the
    /// single source of truth for the prior (`Beta(1,1)`) instead of
    /// recomputing the formula and silently drifting if the prior changes.
    pub fn observations(&self) -> f64 {
        (self.alpha + self.beta - 2.0).max(0.0)
    }
}

/// The bandit cell table: one [`BetaCell`] per `(RequestType, deployment
/// key)`, where the deployment key is the router's stable cooldown key
/// ([`Router::cooldown_key`](crate::router::Router::cooldown_key) =
/// [`stable_key`](crate::router::stable_key), `{model_name}:{16-hex
/// hash}`). A newtype over `Mutex<HashMap>`
/// matching the other router stores
/// ([`crate::router::stores::LatencyStore`],
/// [`crate::router::stores::UsageStore`]): in-memory, updated under a
/// single lock acquisition per call.
///
/// Shared via `Arc` between the [`Router`](crate::router::Router) (whose
/// [`AdaptiveRouter`] strategies read it) and the server's learning hook
/// (which writes it) — clones of the `Arc` see the same cells.
///
/// Cross-replica sync is an explicit non-goal today (no router store syncs
/// cross-replica). [`BanditStore::merge_delta`] is the extension point: a
/// future DualCache/Redis path can publish `(cell, alpha_delta,
/// beta_delta)` deltas that replicas merge here; deltas commute, so merge
/// order is irrelevant and Thompson sampling tolerates the noise.
#[derive(Debug, Default)]
pub struct BanditStore(Mutex<HashMap<(RequestType, String), BetaCell>>);

impl BanditStore {
    /// Create an empty bandit store (all cells at the `Beta(1, 1)` prior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a consistent snapshot of `cells` under a single lock
    /// acquisition, aligned by position with the input. Missing cells
    /// snapshot as the `Beta(1, 1)` prior.
    pub fn sample_snapshot(&self, cells: &[(RequestType, String)]) -> Vec<BetaCell> {
        let guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        cells
            .iter()
            .map(|cell| guard.get(cell).copied().unwrap_or_default())
            .collect()
    }

    /// Record a success for `cell` (`alpha += 1`).
    pub fn record_success(&self, cell: &(RequestType, String)) {
        let mut guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        guard.entry(cell.clone()).or_default().alpha += 1.0;
    }

    /// Record a failure for `cell` (`beta += 1`).
    pub fn record_failure(&self, cell: &(RequestType, String)) {
        let mut guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        guard.entry(cell.clone()).or_default().beta += 1.0;
    }

    /// Recorded outcomes for `cell` (0 for a missing cell).
    pub fn observations(&self, cell: &(RequestType, String)) -> f64 {
        let guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        guard.get(cell).map(BetaCell::observations).unwrap_or(0.0)
    }

    /// A cloned snapshot of every cell in the store, for state inspection
    /// and metric export. Iteration order is unspecified (hash-map order);
    /// exporters that need stable output should sort by cell key.
    pub fn all_cells(&self) -> Vec<((RequestType, String), BetaCell)> {
        let guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .iter()
            .map(|(cell, beta)| (cell.clone(), *beta))
            .collect()
    }

    /// Merge a cross-replica delta (`alpha += alpha_delta`, `beta +=
    /// beta_delta`). Deltas commute: concurrent increments from sibling
    /// replicas never overwrite each other, so merge order does not
    /// matter. **Sync extension point** — no caller today; the store is
    /// in-memory only, like every other router store.
    #[doc(hidden)]
    pub fn merge_delta(&self, cell: &(RequestType, String), alpha_delta: f64, beta_delta: f64) {
        let mut guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let entry = guard.entry(cell.clone()).or_default();
        entry.alpha += alpha_delta;
        entry.beta += beta_delta;
    }

    /// Drop cells whose deployment key (the cooldown-key half of the cell) is
    /// not in `known`. Called at config hot-reload (see
    /// [`crate::router::Router::prune_stale_store_keys`]) so deployments
    /// removed from the model list do not leak their learned posteriors —
    /// without this, every `(RequestType, "{model}:{hash}")` cell
    /// is permanent and churn (canary rollouts, wildcard re-indexing) grows
    /// the cell map monotonically.
    pub fn retain_keys(&self, known: &std::collections::HashSet<String>) {
        let mut guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        guard.retain(|(_, key), _| known.contains(key));
    }
}

// ---------------------------------------------------------------------------
// The strategy
// ---------------------------------------------------------------------------

/// The adaptive routing strategy: Thompson sampling over per-request-type
/// Beta posteriors, blended with a normalized cost term, with epsilon-
/// greedy exploration and a least-busy cold start.
///
/// Randomness is injectable: production uses a [`rand_chacha::ChaChaRng`]
/// seeded from OS entropy ([`rand::rng`] is `!Send` and cannot live in a
/// shared strategy); tests use `AdaptiveRouter::with_rng` with a seeded
/// RNG so convergence and exploration-rate tests are deterministic.
pub struct AdaptiveRouter {
    config: AdaptiveRouterConfig,
    bandit: Arc<BanditStore>,
    least_busy: LeastBusy,
    rng: Mutex<Box<dyn rand::RngCore + Send>>,
}

impl std::fmt::Debug for AdaptiveRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveRouter")
            .field("config", &self.config)
            .field("bandit", &self.bandit)
            .field("least_busy", &self.least_busy)
            .finish_non_exhaustive()
    }
}

impl AdaptiveRouter {
    /// Create an adaptive router over the given configuration, bandit store
    /// (shared with the learning hook), and usage store (cold-start
    /// least-busy signal). Production randomness is a
    /// [`rand_chacha::ChaChaRng`] seeded from OS entropy.
    pub fn new(
        config: AdaptiveRouterConfig,
        bandit: Arc<BanditStore>,
        usage_store: Arc<UsageStore>,
    ) -> Self {
        use rand::SeedableRng;
        Self {
            config: sanitize_config(config),
            bandit,
            least_busy: LeastBusy::new(usage_store),
            rng: Mutex::new(Box::new(rand_chacha::ChaChaRng::from_rng(&mut rand::rng()))),
        }
    }

    /// Test-only constructor with injectable randomness (e.g. a seeded
    /// `rand_chacha::ChaChaRng`) so convergence and exploration tests are
    /// deterministic.
    #[cfg(test)]
    fn with_rng(
        config: AdaptiveRouterConfig,
        bandit: Arc<BanditStore>,
        usage_store: Arc<UsageStore>,
        rng: Box<dyn rand::RngCore + Send>,
    ) -> Self {
        Self {
            config: sanitize_config(config),
            bandit,
            least_busy: LeastBusy::new(usage_store),
            rng: Mutex::new(rng),
        }
    }

    /// The bandit store this strategy samples from (the handle the learning
    /// hook writes to).
    pub fn bandit_store(&self) -> &Arc<BanditStore> {
        &self.bandit
    }
}

impl RoutingStrategy for AdaptiveRouter {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        // Reuse the once-per-request classification the host/router
        // pre-computed; classify on demand only for standalone use.
        let request_type = ctx
            .pre_call
            .adaptive_request_type
            .unwrap_or_else(|| classify_request_type(ctx.request_text.as_deref().unwrap_or("")));
        let keys = candidate_keys(deployments, ctx);
        let cells: Vec<(RequestType, String)> =
            keys.iter().map(|key| (request_type, key.clone())).collect();

        // Cold start: until EVERY candidate arm has enough observations,
        // defer to least-busy — a fresh bandit would otherwise spray
        // production traffic uniformly at random.
        let min_observations = f64::from(self.config.cold_start_min_observations);
        if cells
            .iter()
            .any(|cell| self.bandit.observations(cell) < min_observations)
        {
            let picked = self.least_busy.pick(deployments, ctx);
            tracing::debug!(
                event = "adaptive_route",
                mode = "cold_start",
                request_type = request_type.as_str(),
                model = %ctx.model,
                picked = picked.map(|d| d.model_name.as_str()).unwrap_or(""),
                "adaptive pick deferred to least-busy (cold start)"
            );
            return picked;
        }

        let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());

        // Epsilon-greedy exploration: with probability ε pick uniformly at
        // random so every arm keeps gathering outcomes.
        if self.config.epsilon > 0.0 && rng.random::<f64>() < self.config.epsilon {
            let idx = rng.random_range(..deployments.len());
            tracing::debug!(
                event = "adaptive_route",
                mode = "explore",
                request_type = request_type.as_str(),
                model = %ctx.model,
                picked = deployments[idx].model_name.as_str(),
                "adaptive pick explored uniformly at random"
            );
            return Some(deployments[idx]);
        }

        // Thompson sampling blended with normalized cost:
        // score = sample * quality_weight − normalized_cost * cost_weight.
        let snapshot = self.bandit.sample_snapshot(&cells);
        let costs = normalized_costs(deployments, &ctx.model, self.config.cost_weight);
        // NaN scores (pathological pricing with cost_weight > 0) always lose
        // the `>` comparison, so `best_score` stays at `NEG_INFINITY` iff
        // EVERY candidate scored NaN. In that case the argmax would otherwise
        // deterministically pin index 0; fall back to a weighted shuffle so
        // the all-NaN edge keeps making progress instead. (The cost_weight ==
        // 0 short-circuit in `normalized_costs` prevents NaN from ever
        // reaching here for the documented default config.)
        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for (idx, cell) in snapshot.iter().enumerate() {
            let sample = sample_beta(cell, &mut *rng);
            let score = sample * self.config.quality_weight - costs[idx] * self.config.cost_weight;
            if score > best_score {
                best_score = score;
                best = Some(idx);
            }
        }
        let best = match best {
            Some(best) => best,
            None => {
                tracing::debug!(
                    event = "adaptive_route",
                    mode = "all_nan_fallback",
                    request_type = request_type.as_str(),
                    model = %ctx.model,
                    "every candidate score was NaN; falling back to weighted shuffle"
                );
                return SimpleShuffle.pick(deployments, ctx);
            }
        };
        let picked_cell = &snapshot[best];
        tracing::debug!(
            event = "adaptive_route",
            mode = "thompson",
            request_type = request_type.as_str(),
            model = %ctx.model,
            picked = deployments[best].model_name.as_str(),
            alpha = picked_cell.alpha,
            beta = picked_cell.beta,
            score = best_score,
            "adaptive pick sampled Thompson scores"
        );
        Some(deployments[best])
    }
}

/// Defensive config normalization. The server-side config path validates
/// (`AdaptiveRouterConfig::validate`), but the core is usable without it —
/// a caller constructing the strategy directly (or a future code path)
/// must not be able to poison every pick: epsilon is clamped into
/// `[0, 1]` (NaN reads as 0 — no exploration) and non-finite or negative
/// weights read as 0 (a NaN weight would make every score NaN and pin the
/// pick to the first candidate; a negative weight would invert the
/// optimization).
fn sanitize_config(mut config: AdaptiveRouterConfig) -> AdaptiveRouterConfig {
    let weight = |w: f64| if w.is_finite() && w >= 0.0 { w } else { 0.0 };
    config.quality_weight = weight(config.quality_weight);
    config.cost_weight = weight(config.cost_weight);
    config.epsilon = if config.epsilon.is_finite() {
        config.epsilon.clamp(0.0, 1.0)
    } else {
        0.0
    };
    config
}

/// Per-token costs normalized by the maximum among candidates, aligned with
/// `deployments`. The rate is `input + output` per token, preferring the
/// deployment's pricing override and falling back to the price table (the
/// same rates the cost hooks charge). When NO candidate has any pricing data
/// the cost term is disabled (all zeros) — an all-unpriced pick under a
/// `cost_weight > 0` config therefore silently loses its cost signal, so a
/// debug event is emitted (debug, not warn: this fires per pick, and
/// all-unpriced is a steady configuration state rather than a transient —
/// a per-request warning would spam the hot path). When some candidates have
/// pricing, unpriced ones normalize to 1.0 (pessimistic — a cost-conscious
/// config should not be steered toward deployments with unknown cost). A NaN
/// rate (pathological pricing data) propagates into the candidate's score as
/// NaN, which always loses the argmax comparison — the candidate is never
/// picked by scoring when at least one other candidate has a finite score.
/// If EVERY candidate scores NaN, the argmax falls back to
/// [`SimpleShuffle`] (see [`AdaptiveRouter::pick`]) rather than
/// deterministically pinning index 0. Negative (refund-style) pricing is
/// nonsensical but handled defensively: when every priced candidate is
/// non-positive and at least one is negative, the cost term is normalized
/// against the most-negative magnitude (sign preserved) so a cheaper —
/// more-negative — deployment still scores higher, matching how
/// [`LowestCost`](crate::router::LowestCost) ranks negative rates via the
/// `<` comparison instead of silently zeroing the term.
fn normalized_costs(
    deployments: &[&Deployment],
    requested_model: &str,
    cost_weight: f64,
) -> Vec<f64> {
    // Cost blending is opt-in: the documented default is `cost_weight: 0.0`,
    // in which case the cost term is semantically disabled. Short-circuit
    // before any pricing lookup so the hot path does no map work, and so a
    // pathological non-finite rate (e.g. `inf` deserialized from JSON) can
    // never reach the score as `inf * 0.0 == NaN` (which would poison the
    // argmax: with every candidate NaN, index 0 is returned silently).
    if cost_weight == 0.0 {
        return vec![0.0; deployments.len()];
    }
    let raw: Vec<Option<f64>> = deployments
        .iter()
        .map(|d| {
            // A per-second-only override has no per-token sum to blend, so
            // it falls through to the table entry (a 0.0 sum would read as
            // "free" and poison the normalization).
            if let Some(override_pricing) = d.oxllm_params.pricing_override() {
                if !override_pricing.is_per_second_priced() {
                    return Some(
                        override_pricing.input_cost_per_token
                            + override_pricing.output_cost_per_token,
                    );
                }
            }
            let pricing_model = deployment_pricing_model(d, requested_model);
            lookup_pricing_with_provider(&pricing_model, d.provider_slug())
                .filter(|pricing| !pricing.is_per_second_priced())
                .map(|p| p.input_cost_per_token + p.output_cost_per_token)
        })
        .collect();
    // Fold from `-inf` (not `0.0`) so an all-negative pricing config still
    // resolves a real maximum instead of leaving the seed `0.0` dominant —
    // which would trip the `max <= 0.0` disable below and silently zero the
    // cost term. `f64::max` returns the non-NaN operand, so a NaN rate never
    // poisons the fold (it is carried through to the candidate's score as NaN
    // and loses the argmax, per the function doc).
    let max = raw
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        // No candidate carries any pricing data → disable the cost term.
        if cost_weight > 0.0 {
            tracing::debug!(
                event = "adaptive_route",
                candidates = deployments.len(),
                "cost_weight > 0 but no candidate has pricing data; cost term disabled for this pick"
            );
        }
        return vec![0.0; deployments.len()];
    }
    if max <= 0.0 {
        // No candidate has positive pricing. If at least one is negative,
        // keep the cost term meaningful by normalizing against the
        // most-negative magnitude (sign preserved) so a cheaper deployment
        // scores higher. An all-zero (genuinely unpriced) config still
        // disables the term.
        let denom = raw
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .abs();
        if denom == 0.0 || !denom.is_finite() {
            if cost_weight > 0.0 {
                tracing::debug!(
                    event = "adaptive_route",
                    candidates = deployments.len(),
                    "cost_weight > 0 but no candidate has positive pricing; cost term disabled for this pick"
                );
            }
            return vec![0.0; deployments.len()];
        }
        return raw
            .iter()
            .map(|cost| cost.map(|c| c / denom).unwrap_or(1.0))
            .collect();
    }
    raw.iter()
        .map(|cost| cost.map(|c| c / max).unwrap_or(1.0))
        .collect()
}

/// Sample `Beta(alpha, beta)`. Cells never go below their `Beta(1, 1)`
/// prior in practice; the clamp + uniform fallback defend against a
/// pathological delta merge producing a non-positive parameter.
fn sample_beta<R: Rng + ?Sized>(cell: &BetaCell, rng: &mut R) -> f64 {
    const MIN_PARAM: f64 = 1e-3;
    match rand_distr::Beta::new(cell.alpha.max(MIN_PARAM), cell.beta.max(MIN_PARAM)) {
        Ok(dist) => rand_distr::Distribution::sample(&dist, rng),
        Err(_) => rng.random(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::OxllmParams;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    fn make_dep(name: &str) -> Deployment {
        Deployment {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: format!("openai/{name}"),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn make_dep_priced(name: &str, input: f64, output: f64) -> Deployment {
        let mut dep = make_dep(name);
        dep.oxllm_params.input_cost_per_token = Some(input);
        dep.oxllm_params.output_cost_per_token = Some(output);
        dep
    }

    fn seeded_router(
        config: AdaptiveRouterConfig,
        bandit: Arc<BanditStore>,
        seed: u64,
    ) -> AdaptiveRouter {
        AdaptiveRouter::with_rng(
            config,
            bandit,
            Arc::new(UsageStore::new()),
            Box::new(ChaChaRng::seed_from_u64(seed)),
        )
    }

    /// A two-deployment context whose candidate keys match the router's
    /// cooldown-key scheme (`stable_key`, `{model_name}:{16-hex hash}`;
    /// the literal values here are arbitrary test controls).
    fn ctx_with_keys(request_type: Option<RequestType>, keys: &[&str]) -> RequestContext {
        let mut ctx = RequestContext::new("model");
        ctx.pre_call.adaptive_request_type = request_type;
        ctx.pre_call.candidate_keys = keys.iter().map(|k| k.to_string()).collect();
        // Identity fingerprints matching `two_deps()`, so the precomputed
        // (arbitrary, test-controlled) keys are trusted by `candidate_keys`.
        ctx.pre_call.candidate_ids = vec![
            "dep-a:openai/dep-a".to_string(),
            "dep-b:openai/dep-b".to_string(),
        ];
        ctx
    }

    fn two_deps() -> [Deployment; 2] {
        [make_dep("dep-a"), make_dep("dep-b")]
    }

    const KEYS: [&str; 2] = ["model:0", "model:1"];

    fn warm_config() -> AdaptiveRouterConfig {
        // Zero cold-start threshold: the bandit engages from the first pick.
        AdaptiveRouterConfig {
            cold_start_min_observations: 0,
            ..Default::default()
        }
    }

    // ------------------------------------------------------------------
    // Classification
    // ------------------------------------------------------------------

    #[test]
    fn classify_code_requests() {
        assert_eq!(
            classify_request_type("fix this python function\n```\ndef f(x):\n```"),
            RequestType::Code
        );
        assert_eq!(
            classify_request_type("why does this sql query throw a syntax exception?"),
            RequestType::Code
        );
        assert_eq!(
            classify_request_type("debug this stack trace from my rust program"),
            RequestType::Code
        );
    }

    #[test]
    fn classify_reasoning_requests() {
        assert_eq!(
            classify_request_type("prove that the square root of 2 is irrational"),
            RequestType::Reasoning
        );
        assert_eq!(
            classify_request_type("derive the quadratic formula, step by step"),
            RequestType::Reasoning
        );
    }

    #[test]
    fn classify_creative_requests() {
        assert_eq!(
            classify_request_type("write a story about a dragon who loves poetry"),
            RequestType::Creative
        );
        assert_eq!(
            classify_request_type("compose a haiku and a song about autumn"),
            RequestType::Creative
        );
    }

    #[test]
    fn classify_qa_requests() {
        assert_eq!(
            classify_request_type("what is the capital of france?"),
            RequestType::QA
        );
        assert_eq!(
            classify_request_type("who was the first person on the moon and when did it happen?"),
            RequestType::QA
        );
    }

    #[test]
    fn classify_summarization_requests() {
        assert_eq!(
            classify_request_type("summarize this article into key points"),
            RequestType::Summarization
        );
        assert_eq!(
            classify_request_type("tl;dr of the attached report, give me the recap"),
            RequestType::Summarization
        );
    }

    #[test]
    fn classify_translation_requests() {
        assert_eq!(
            classify_request_type("translate this paragraph into french"),
            RequestType::Translation
        );
        assert_eq!(
            classify_request_type("how do you say good morning in japanese?"),
            RequestType::Translation
        );
    }

    #[test]
    fn classify_other_for_empty_and_unmatched_text() {
        assert_eq!(classify_request_type(""), RequestType::Other);
        assert_eq!(classify_request_type("hi"), RequestType::Other);
        assert_eq!(
            classify_request_type("the weather is nice today"),
            RequestType::Other
        );
    }

    #[test]
    fn classification_is_case_insensitive_and_punctuation_aware() {
        // Punctuation glued to a keyword must not defeat matching; a
        // keyword inside a larger word must not match.
        assert_eq!(
            classify_request_type("SUMMARIZE, please!"),
            RequestType::Summarization
        );
        assert_eq!(
            classify_request_type("the storyteller was a character in his own tale"),
            RequestType::Creative
        );
        // "hi" is not a keyword and must not match inside "this".
        assert_eq!(classify_request_type("consider this"), RequestType::Other);
    }

    #[test]
    fn classification_tie_breaks_deterministically() {
        // One translation marker + one summarization marker → tie → the
        // earlier declaration order (Summarization before Translation)
        // wins, deterministically.
        assert_eq!(
            classify_request_type("summarize and translate"),
            RequestType::Summarization
        );
    }

    #[test]
    fn bare_language_names_do_not_classify_as_translation() {
        assert_eq!(
            classify_request_type("i had french toast for breakfast"),
            RequestType::Other
        );
    }

    // ------------------------------------------------------------------
    // BanditStore
    // ------------------------------------------------------------------

    #[test]
    fn bandit_store_records_and_counts_observations() {
        let store = BanditStore::new();
        let cell = (RequestType::Code, "model:0".to_string());
        assert_eq!(store.observations(&cell), 0.0);
        store.record_success(&cell);
        store.record_success(&cell);
        store.record_failure(&cell);
        assert_eq!(store.observations(&cell), 3.0);
        let snapshot = store.sample_snapshot(std::slice::from_ref(&cell));
        assert_eq!(snapshot[0].alpha, 3.0);
        assert_eq!(snapshot[0].beta, 2.0);
    }

    #[test]
    fn bandit_store_missing_cells_snapshot_as_prior() {
        let store = BanditStore::new();
        let cells = vec![
            (RequestType::QA, "model:0".to_string()),
            (RequestType::QA, "model:1".to_string()),
        ];
        let snapshot = store.sample_snapshot(&cells);
        assert_eq!(snapshot, vec![BetaCell::default(), BetaCell::default()]);
    }

    #[test]
    fn merge_delta_commutes() {
        // Delta merges are order-independent: concurrent replica updates
        // never lose counts regardless of arrival order.
        let cell = (RequestType::Other, "model:0".to_string());
        let a = BanditStore::new();
        a.merge_delta(&cell, 2.0, 0.0);
        a.merge_delta(&cell, 0.0, 1.0);
        a.merge_delta(&cell, 3.0, 2.0);
        let b = BanditStore::new();
        b.merge_delta(&cell, 3.0, 2.0);
        b.merge_delta(&cell, 0.0, 1.0);
        b.merge_delta(&cell, 2.0, 0.0);
        assert_eq!(
            a.sample_snapshot(std::slice::from_ref(&cell)),
            b.sample_snapshot(&[cell])
        );
    }

    #[test]
    fn all_cells_snapshots_every_recorded_cell() {
        let store = BanditStore::new();
        assert!(store.all_cells().is_empty(), "fresh store has no cells");
        let cell_a = (RequestType::Code, "model:0".to_string());
        let cell_b = (RequestType::QA, "model:1".to_string());
        store.record_success(&cell_a);
        store.record_failure(&cell_b);
        store.record_failure(&cell_b);
        let cells = store.all_cells();
        // Snapshot order is unspecified (hash-map order); look cells up by
        // key instead of asserting positions.
        assert_eq!(cells.len(), 2, "every recorded cell is exported");
        let params = |cell: &(RequestType, String)| {
            cells
                .iter()
                .find(|(c, _)| c == cell)
                .map(|(_, b)| (b.alpha, b.beta))
        };
        assert_eq!(params(&cell_a), Some((2.0, 1.0)));
        assert_eq!(params(&cell_b), Some((1.0, 3.0)));
        // The snapshot is cloned: later writes do not leak into it.
        store.record_success(&cell_a);
        assert_eq!(cells.len(), 2);
    }

    #[test]
    fn bandit_retain_keys_drops_cells_for_removed_deployments() {
        let store = BanditStore::new();
        let keep = (RequestType::Other, "model:0".to_string());
        let drop_cell = (RequestType::Code, "model:1".to_string());
        store.record_success(&keep);
        store.record_success(&drop_cell);
        assert_eq!(store.all_cells().len(), 2);
        let known: std::collections::HashSet<String> =
            ["model:0".to_string()].into_iter().collect();
        store.retain_keys(&known);
        assert_eq!(store.all_cells().len(), 1);
        assert_eq!(store.observations(&keep), 1.0);
        assert_eq!(store.observations(&drop_cell), 0.0, "dropped cell reads 0");
    }

    // ------------------------------------------------------------------
    // Picking
    // ------------------------------------------------------------------

    #[test]
    fn pick_empty_returns_none() {
        let router = seeded_router(warm_config(), Arc::new(BanditStore::new()), 1);
        let ctx = ctx_with_keys(Some(RequestType::Other), &[]);
        assert!(router.pick(&[], &ctx).is_none());
    }

    #[test]
    fn thompson_pick_prefers_stronger_posterior() {
        let bandit = Arc::new(BanditStore::new());
        let cell_a = (RequestType::Other, KEYS[0].to_string());
        let cell_b = (RequestType::Other, KEYS[1].to_string());
        // A: Beta(11, 1) — mean ~0.92; B: Beta(1, 11) — mean ~0.08.
        for _ in 0..10 {
            bandit.record_success(&cell_a);
            bandit.record_failure(&cell_b);
        }
        let router = seeded_router(warm_config(), bandit, 7);
        let deps = two_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(Some(RequestType::Other), &KEYS);
        let mut a_picks = 0;
        for _ in 0..200 {
            let picked = router.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "dep-a" {
                a_picks += 1;
            }
        }
        // ε = 0.1 exploration caps B's share at ~5%; Thompson nearly never
        // samples B above A. Loose bound against RNG drift.
        assert!(
            a_picks >= 160,
            "strong posterior must dominate picks ({a_picks}/200 for A)"
        );
    }

    #[test]
    fn bandit_converges_to_better_arm() {
        // Simulation: arm A succeeds 90% of the time, arm B 50%. Over 5000
        // seeded picks the strategy must converge to picking A > 85% of the
        // time in the last 1000 picks (convergence lands ~950/1000; the
        // bound keeps a ~10σ margin, so no flake risk).
        let bandit = Arc::new(BanditStore::new());
        let router = seeded_router(warm_config(), bandit.clone(), 42);
        let deps = two_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(Some(RequestType::Other), &KEYS);
        let mut outcome_rng = ChaChaRng::seed_from_u64(1);
        let mut recent_a = 0usize;
        for round in 0..5000 {
            let picked = router.pick(&refs, &ctx).expect("should pick");
            let is_a = picked.model_name == "dep-a";
            let cell = (
                RequestType::Other,
                if is_a { KEYS[0] } else { KEYS[1] }.to_string(),
            );
            let p_success = if is_a { 0.9 } else { 0.5 };
            if outcome_rng.random::<f64>() < p_success {
                bandit.record_success(&cell);
            } else {
                bandit.record_failure(&cell);
            }
            if round >= 4000 && is_a {
                recent_a += 1;
            }
        }
        assert!(
            recent_a > 850,
            "must converge to the better arm (>850/1000 recent picks for A, got {recent_a})"
        );
    }

    #[test]
    fn cold_start_defers_to_least_busy() {
        let bandit = Arc::new(BanditStore::new());
        let usage = Arc::new(UsageStore::new());
        // Make dep-a the busy one; cold start must pick the idle dep-b. The
        // in-flight count is seeded under the pick's candidate keys — the
        // router memoizes the true cooldown keys into the context before
        // the adaptive pick, and the least-busy cold-start leg consumes
        // those (not fresh stable_key recomputations).
        usage.incr_in_flight(KEYS[0], 5);
        let config = AdaptiveRouterConfig {
            cold_start_min_observations: 20,
            ..Default::default()
        };
        let router =
            AdaptiveRouter::with_rng(config, bandit, usage, Box::new(ChaChaRng::seed_from_u64(3)));
        let deps = two_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(Some(RequestType::Other), &KEYS);
        for _ in 0..10 {
            let picked = router.pick(&refs, &ctx).expect("should pick");
            assert_eq!(
                picked.model_name, "dep-b",
                "cold start must defer to the least-busy (idle) deployment"
            );
        }
    }

    #[test]
    fn bandit_engages_once_all_arms_are_warm() {
        let bandit = Arc::new(BanditStore::new());
        let cell_a = (RequestType::Other, KEYS[0].to_string());
        let cell_b = (RequestType::Other, KEYS[1].to_string());
        let config = AdaptiveRouterConfig {
            cold_start_min_observations: 2,
            epsilon: 0.0,
            ..Default::default()
        };
        let router = seeded_router(config, bandit.clone(), 11);
        let deps = two_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(Some(RequestType::Other), &KEYS);

        // One arm warm, the other cold → still least-busy (any-cold rule).
        bandit.record_failure(&cell_a);
        bandit.record_failure(&cell_a);
        let _ = router.pick(&refs, &ctx);

        // Both arms warm; B is the clearly better arm (Beta(3,1) vs
        // Beta(1,3)) → Thompson engages and favors B.
        bandit.record_success(&cell_b);
        bandit.record_success(&cell_b);
        let mut b_picks = 0;
        for _ in 0..100 {
            if router.pick(&refs, &ctx).expect("pick").model_name == "dep-b" {
                b_picks += 1;
            }
        }
        assert!(
            b_picks > 60,
            "warm bandit with ε=0 must favor the better arm ({b_picks}/100 for B)"
        );
    }

    #[test]
    fn epsilon_greedy_explores_at_configured_rate() {
        // ε = 0.2 with a dominant A arm: exploration picks B at ~ε/2 ≈ 10%
        // of picks; Thompson exploitation essentially never picks B.
        // Seeded; assert loose bounds around the expectation.
        let bandit = Arc::new(BanditStore::new());
        let cell_a = (RequestType::Other, KEYS[0].to_string());
        for _ in 0..50 {
            bandit.record_success(&cell_a);
        }
        let config = AdaptiveRouterConfig {
            epsilon: 0.2,
            ..warm_config()
        };
        let router = seeded_router(config, bandit, 99);
        let deps = two_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(Some(RequestType::Other), &KEYS);
        let mut b_picks = 0usize;
        let rounds = 4000;
        for _ in 0..rounds {
            if router.pick(&refs, &ctx).expect("pick").model_name == "dep-b" {
                b_picks += 1;
            }
        }
        let rate = b_picks as f64 / rounds as f64;
        assert!(
            (0.06..=0.16).contains(&rate),
            "exploration rate should center near ε/2 = 0.10, got {rate}"
        );
    }

    #[test]
    fn cost_blending_prefers_cheaper_arm() {
        // Equal posteriors; with cost_weight > 0 the cheaper deployment
        // must win the blend most of the time.
        let bandit = Arc::new(BanditStore::new());
        let config = AdaptiveRouterConfig {
            quality_weight: 1.0,
            cost_weight: 0.5,
            ..warm_config()
        };
        let router = seeded_router(config, bandit, 5);
        let cheap = make_dep_priced("dep-cheap", 1e-9, 1e-9);
        let pricey = make_dep_priced("dep-pricey", 1e-5, 1e-5);
        let deps = [cheap, pricey];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(Some(RequestType::Other), &KEYS);
        let mut cheap_picks = 0;
        for _ in 0..200 {
            if router.pick(&refs, &ctx).expect("pick").model_name == "dep-cheap" {
                cheap_picks += 1;
            }
        }
        assert!(
            cheap_picks > 150,
            "cost blending must prefer the cheaper arm ({cheap_picks}/200)"
        );
    }

    #[test]
    fn cost_term_disabled_without_pricing() {
        // No candidate has pricing data → the cost term is zero for all
        // candidates and the pick is purely Thompson quality sampling
        // (a positive cost_weight is passed to exercise the warn path).
        let deps = two_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        assert!(
            normalized_costs(&refs, "model", 0.5)
                .iter()
                .all(|c| *c == 0.0),
            "unpriced candidates must disable the cost term"
        );
    }

    #[test]
    fn unpriced_candidate_is_pessimistic_when_others_are_priced() {
        let priced = make_dep_priced("dep-priced", 1e-5, 1e-5);
        let unpriced = make_dep("dep-unpriced");
        let deps = [priced, unpriced];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let costs = normalized_costs(&refs, "model", 0.5);
        assert_eq!(costs[0], 1.0, "the only priced candidate is the max");
        assert_eq!(costs[1], 1.0, "unpriced candidate reads pessimistic");
    }

    #[test]
    fn normalized_costs_uses_base_model_for_opaque_aliases() {
        let mut expensive = make_dep("dep-expensive");
        expensive.oxllm_params.model = "openai/gpt-4o".to_string();
        let mut aliased = make_dep("dep-aliased");
        aliased.oxllm_params.model = "openai/private-alias".to_string();
        aliased.model_info = Some(crate::deployment::ModelInfo {
            base_model: Some("openai/gpt-4o-mini".to_string()),
            ..Default::default()
        });
        let deps = [expensive, aliased];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let costs = normalized_costs(&refs, "model", 0.5);
        assert!(
            costs[1] < costs[0],
            "the base-model-priced alias must be cheaper: {costs:?}"
        );
    }

    #[test]
    fn normalized_costs_reflects_negative_pricing_not_silently_zero() {
        // A pathological all-negative pricing config (refund-style rates or a
        // YAML typo) must not silently disable the cost term: the rates are
        // normalized against the most-negative magnitude (sign preserved) so
        // the cheaper — more-negative — deployment scores higher, matching
        // how `LowestCost` ranks negative rates via `<`.
        let cheap = make_dep_priced("dep-cheap", -0.002, 0.0); // sum = -0.002
        let pricey = make_dep_priced("dep-pricey", -0.001, 0.0); // sum = -0.001
        let deps = [cheap, pricey];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let costs = normalized_costs(&refs, "model", 0.5);
        assert!(
            costs.iter().any(|c| *c != 0.0),
            "negative pricing must not silently zero the cost term (got {costs:?})"
        );
        assert_eq!(costs[0], -1.0, "most-negative rate normalizes to -1.0");
        assert_eq!(costs[1], -0.5, "less-negative rate normalizes to -0.5");
        assert!(
            costs[0] < costs[1],
            "the cheaper (more-negative) deployment must get the lower normalized cost (got {costs:?})"
        );
    }

    #[test]
    fn normalized_costs_skips_pricing_when_cost_weight_zero() {
        // cost_weight == 0 disables the cost term: the function must return
        // all-zeros WITHOUT consulting pricing, so no non-finite rate can
        // poison a score via `inf * 0.0 == NaN`.
        let inf = make_dep_priced("dep-inf", f64::INFINITY, f64::INFINITY);
        let nan = make_dep_priced("dep-nan", f64::NAN, f64::NAN);
        let plain = make_dep("dep-plain");
        let deps = [inf, nan, plain];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let costs = normalized_costs(&refs, "model", 0.0);
        assert!(
            costs.iter().all(|c| *c == 0.0),
            "cost_weight=0 must skip pricing entirely (got {costs:?})"
        );
    }

    #[test]
    fn pick_ignores_pathological_pricing_when_cost_weight_zero() {
        // With cost blending disabled, a candidate carrying non-finite
        // pricing must not corrupt the argmax: the pick stays driven purely
        // by the Thompson sample, deterministically favoring the dominant
        // posterior.
        let bandit = Arc::new(BanditStore::new());
        let cell_a = (RequestType::Other, KEYS[0].to_string());
        let cell_b = (RequestType::Other, KEYS[1].to_string());
        for _ in 0..20 {
            bandit.record_success(&cell_a);
            bandit.record_failure(&cell_b);
        }
        let config = AdaptiveRouterConfig {
            epsilon: 0.0,
            ..warm_config()
        };
        let router = seeded_router(config, bandit, 31);
        let inf = make_dep_priced("dep-a", f64::INFINITY, f64::INFINITY);
        let nan = make_dep_priced("dep-b", f64::NAN, f64::NAN);
        let deps = [inf, nan];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(Some(RequestType::Other), &KEYS);
        let mut a_picks = 0;
        for _ in 0..100 {
            if router.pick(&refs, &ctx).expect("pick").model_name == "dep-a" {
                a_picks += 1;
            }
        }
        assert!(
            a_picks >= 95,
            "non-finite pricing with cost_weight=0 must not poison the pick ({a_picks}/100 for A)"
        );
    }

    #[test]
    fn pick_all_nan_scores_fall_back_to_shuffle() {
        // When EVERY candidate's score is NaN (cost_weight > 0 + pathological
        // pricing on all arms), the argmax would otherwise deterministically
        // pin index 0. Instead it must fall back to a weighted shuffle so the
        // all-NaN edge keeps distributing traffic. Warm both arms equally so
        // cold-start and Thompson-quality cannot explain the distribution.
        let bandit = Arc::new(BanditStore::new());
        let cell_a = (RequestType::Other, KEYS[0].to_string());
        let cell_b = (RequestType::Other, KEYS[1].to_string());
        for _ in 0..10 {
            bandit.record_success(&cell_a);
            bandit.record_success(&cell_b);
        }
        let config = AdaptiveRouterConfig {
            // A positive cost_weight lets the NaN pricing reach the score.
            cost_weight: 0.5,
            epsilon: 0.0,
            ..warm_config()
        };
        let router = seeded_router(config, bandit, 4);
        let nan_a = make_dep_priced("dep-a", f64::NAN, f64::NAN);
        let nan_b = make_dep_priced("dep-b", f64::NAN, f64::NAN);
        let deps = [nan_a, nan_b];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(Some(RequestType::Other), &KEYS);
        let mut saw_a = false;
        let mut saw_b = false;
        for _ in 0..200 {
            match router.pick(&refs, &ctx).expect("pick").model_name.as_str() {
                "dep-a" => saw_a = true,
                "dep-b" => saw_b = true,
                _ => {}
            }
        }
        // Under the old all-NaN argmax bug (epsilon 0), dep-b would NEVER be
        // picked. The shuffle fallback distributes across both.
        assert!(
            saw_a && saw_b,
            "all-NaN scores must fall back to shuffle (saw_a={saw_a}, saw_b={saw_b})"
        );
    }

    #[test]
    fn pick_reuses_precomputed_request_type() {
        // The once-per-request classification rides PreCallData; the pick
        // must trust it even when the request text says something else.
        let bandit = Arc::new(BanditStore::new());
        // Make the TRANSLATION cell for dep-b dominant.
        let cell = (RequestType::Translation, KEYS[1].to_string());
        for _ in 0..20 {
            bandit.record_success(&cell);
        }
        let config = AdaptiveRouterConfig {
            epsilon: 0.0,
            ..warm_config()
        };
        let router = seeded_router(config, bandit, 13);
        let deps = two_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let mut ctx = ctx_with_keys(Some(RequestType::Translation), &KEYS);
        ctx.request_text = Some("prove this theorem step by step".to_string());
        let mut b_picks = 0;
        for _ in 0..100 {
            if router.pick(&refs, &ctx).expect("pick").model_name == "dep-b" {
                b_picks += 1;
            }
        }
        // The reasoning-sounding text must NOT drive the pick: only the
        // pre-computed Translation type has a dominant posterior (dep-b
        // wins ~95% of samples per pick).
        assert!(
            b_picks > 80,
            "pre-computed request type must drive the cell lookup ({b_picks}/100 for B)"
        );
    }

    #[test]
    fn standalone_pick_falls_back_to_store_keys() {
        // No candidate keys on the context (standalone use): the strategy
        // falls back to store-style keys and still picks.
        let bandit = Arc::new(BanditStore::new());
        let router = seeded_router(warm_config(), bandit, 17);
        let deps = two_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = RequestContext::new("model");
        assert!(router.pick(&refs, &ctx).is_some());
    }

    #[test]
    fn invalid_config_values_are_sanitized_defensively() {
        // Server-side `validate()` rejects these, but a direct core caller
        // gets silent defense: epsilon clamps into [0, 1], non-finite or
        // negative weights read as 0.
        let config = sanitize_config(AdaptiveRouterConfig {
            quality_weight: f64::NAN,
            cost_weight: f64::NEG_INFINITY,
            epsilon: 2.0,
            cold_start_min_observations: 0,
        });
        assert_eq!(config.quality_weight, 0.0, "NaN weight reads as 0");
        assert_eq!(config.cost_weight, 0.0, "non-finite weight reads as 0");
        assert_eq!(config.epsilon, 1.0, "epsilon clamps to 1");

        let config = sanitize_config(AdaptiveRouterConfig {
            quality_weight: -3.0,
            cost_weight: 0.5,
            epsilon: -0.5,
            cold_start_min_observations: 0,
        });
        assert_eq!(config.quality_weight, 0.0, "negative weight reads as 0");
        assert_eq!(config.cost_weight, 0.5, "valid weight is preserved");
        assert_eq!(config.epsilon, 0.0, "negative epsilon clamps to 0");

        let valid = AdaptiveRouterConfig::default();
        assert_eq!(
            sanitize_config(valid.clone()),
            valid,
            "a valid config passes through unchanged"
        );
    }

    #[test]
    fn epsilon_one_always_explores() {
        let bandit = Arc::new(BanditStore::new());
        let cell_a = (RequestType::Other, KEYS[0].to_string());
        for _ in 0..100 {
            bandit.record_success(&cell_a);
        }
        let config = AdaptiveRouterConfig {
            epsilon: 1.0,
            ..warm_config()
        };
        let router = seeded_router(config, bandit, 23);
        let deps = two_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(Some(RequestType::Other), &KEYS);
        let mut b_picks = 0;
        for _ in 0..1000 {
            if router.pick(&refs, &ctx).expect("pick").model_name == "dep-b" {
                b_picks += 1;
            }
        }
        // Uniform exploration over two arms ≈ 500; loose bounds.
        assert!(
            (350..=650).contains(&b_picks),
            "ε = 1.0 must explore uniformly, got {b_picks}/1000 for B"
        );
    }
}
