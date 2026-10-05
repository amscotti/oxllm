//! The complexity router: classify request complexity from keyword scoring
//! (no LLM call) and route to the model pool configured for that tier.
//!
//! Scoring combines six weighted dimensions over the request text
//! ([`RequestContext::request_text`]):
//!
//! | dimension         | signal                                              | weight |
//! |-------------------|-----------------------------------------------------|--------|
//! | code presence     | code fences, definitions, imports                   | +0.25  |
//! | reasoning markers | "prove", "derive", "step by step", "analyze", ...   | +0.80  |
//! | technical terms   | "api", "database", "algorithm", "protocol", ...     | +0.15  |
//! | question shape    | multi-part questions, comparisons, evaluations      | +0.10  |
//! | token count       | estimated request size (saturates at ~2000 tokens)  | +0.05  |
//! | simple indicators | "hello", "hi", "quick", "easy", ... (negative)      | −0.20  |
//!
//! The weighted sum is clamped to `[0, 1]` and mapped to a tier by
//! configurable boundaries (defaults): SIMPLE `< 0.15`, MEDIUM `< 0.35`,
//! COMPLEX `< 0.60`, REASONING `>= 0.60`. Reasoning markers dominate by
//! design: a proof-shaped prompt is the strongest signal that a request
//! needs a reasoning-tier model, while casual small talk scores near zero.
//! Configured `escalation_keywords` bump the tier one level up when
//! present.
//!
//! Candidate expansion happens in
//! [`Router::pick_deployment`](crate::router::Router::pick_deployment): the
//! classified tier's configured model pool becomes the candidate list (with
//! `default_model`, then the requested model's group as fallbacks — each
//! level cooldown-filtered, first non-empty level wins). The router uses
//! the shared resolved-pool picker with that classification, preserving
//! alias/wildcard expansion without classifying or filtering a second time.
//! Standalone strategy picks classify and filter on demand; callers may
//! also supply a precomputed classification through
//! [`RequestContext::pre_call`](crate::router::strategy::PreCallData::complexity_classification).

use serde_json::Value;

use crate::deployment::Deployment;
use crate::router::SimpleShuffle;
use crate::router::strategy::{RequestContext, RoutingStrategy};
use crate::router::types::ComplexityRouterConfig;
use crate::tokens::estimate_tokens;

/// The complexity tier a request is classified into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Casual chat, greetings, trivial lookups.
    Simple,
    /// Everyday questions and single-step tasks.
    Medium,
    /// Multi-step technical work: code, comparisons, design questions.
    Complex,
    /// Proof-shaped or deep-reasoning requests.
    Reasoning,
}

impl Tier {
    /// The config/tracing name of the tier (`tiers` map key).
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Simple => "simple",
            Tier::Medium => "medium",
            Tier::Complex => "complex",
            Tier::Reasoning => "reasoning",
        }
    }

    /// The next higher tier (Reasoning stays Reasoning). Applied when an
    /// escalation keyword is present.
    pub fn next(self) -> Tier {
        match self {
            Tier::Simple => Tier::Medium,
            Tier::Medium => Tier::Complex,
            Tier::Complex => Tier::Reasoning,
            Tier::Reasoning => Tier::Reasoning,
        }
    }
}

// ---------------------------------------------------------------------------
// Keyword tables (all matched case-insensitively on word-ish boundaries)
// ---------------------------------------------------------------------------

/// Code fences, definitions, imports — strong signal of a coding task.
/// Markers that are bare English words (`from`, `class`, `import`,
/// `function`, and the word-boundary-matched `def`, `fn`) are matched on
/// word boundaries by [`TextMatcher`]: bare-word trailing-space spellings
/// (`"def "`, `"fn "`) trim down to the same word markers, so the lists
/// spell them without the space. Prose forms ("from Paris", "math class",
/// "import tariffs", "defendant") do not match.
const CODE_MARKERS: &[&str] = &["```", "def", "fn", "#include", "console.log", "println"];

/// Code phrases whose words must ALL be present (word-boundary matched)
/// to count as a code signal. Each word alone is too prose-prone, but
/// together they form a distinctive shape — `from X import Y` is a
/// Python import statement, not prose.
const CODE_MARKER_COMBOS: &[(&str, &str)] = &[("from", "import")];

/// Strong reasoning markers: each match scores 0.75 of the reasoning
/// dimension (two or more saturate it).
const STRONG_REASONING_MARKERS: &[&str] = &[
    "prove",
    "proof",
    "derive",
    "theorem",
    "step by step",
    "step-by-step",
    "explain why",
    "reasoning",
];

/// Weak reasoning markers: each match scores 0.35 of the reasoning
/// dimension.
const WEAK_REASONING_MARKERS: &[&str] = &[
    "analyze",
    "analyse",
    "explain",
    "evaluate",
    "compare",
    "discuss",
    "describe how",
];

/// Technical vocabulary: two or more matches saturate the dimension.
const TECHNICAL_TERMS: &[&str] = &[
    "api",
    "database",
    "algorithm",
    "protocol",
    "architecture",
    "microservices",
    "distributed",
    "kubernetes",
    "docker",
    "compiler",
    "encryption",
];

/// Comparison / evaluation shapes: two or more matches saturate the
/// question-complexity dimension. Two or more `?` in the text (a
/// multi-part question) also count as one match.
const QUESTION_MARKERS: &[&str] = &[
    " vs ",
    "versus",
    "compare",
    "comparison",
    "difference between",
    "tradeoffs",
    "trade-offs",
    "pros and cons",
    "which is better",
    "evaluate",
];

/// Casual-task vocabulary (negative weight): two or more matches saturate
/// the simple dimension.
const SIMPLE_INDICATORS: &[&str] = &[
    "hello",
    "hi",
    "hey",
    "simple",
    "quick",
    "easy",
    "thanks",
    "thank you",
];

// Dimension weights (see the module table). The maximum positive sum is
// 1.35; the total is clamped to [0, 1].
const W_CODE: f64 = 0.25;
const W_REASONING: f64 = 0.80;
const W_TECHNICAL: f64 = 0.15;
const W_QUESTION: f64 = 0.10;
const W_TOKENS: f64 = 0.05;
const W_SIMPLE: f64 = 0.20;

/// Estimated request size (tokens) at which the token-count dimension
/// saturates.
const TOKEN_SATURATION: f64 = 2000.0;

/// Default tier ceilings (see [`ComplexityRouterConfig::tier_boundaries`]).
/// Public so config validation can reason about the effective boundaries
/// when only some keys are set.
pub const DEFAULT_SIMPLE_MAX: f64 = 0.15;
/// See [`DEFAULT_SIMPLE_MAX`].
pub const DEFAULT_MEDIUM_MAX: f64 = 0.35;
/// See [`DEFAULT_SIMPLE_MAX`].
pub const DEFAULT_COMPLEX_MAX: f64 = 0.60;

/// Classify the request's complexity from its text. Returns the tier and
/// the raw score in `[0, 1]`.
///
/// Escalation keywords (when configured) bump the tier one level up after
/// scoring; the score itself is returned unmodified.
pub fn classify(ctx: &RequestContext, config: &ComplexityRouterConfig) -> (Tier, f64) {
    let text = ctx.request_text.as_deref().unwrap_or("");
    let score = score_text(text, &ctx.model);
    let mut tier = tier_for_score(score, config);
    if let Some(keywords) = &config.escalation_keywords {
        let matcher = TextMatcher::new(text);
        if keywords.iter().any(|kw| matcher.matches(kw)) {
            tier = tier.next();
        }
    }
    (tier, score)
}

/// The raw complexity score for `text` in `[0, 1]`.
fn score_text(text: &str, model: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let matcher = TextMatcher::new(text);
    let count = |markers: &[&str]| markers.iter().filter(|m| matcher.matches(m)).count() as f64;

    let combo_hit = CODE_MARKER_COMBOS
        .iter()
        .any(|combo| [combo.0, combo.1].iter().all(|m| matcher.matches(m)));
    let code = if count(CODE_MARKERS) > 0.0 || combo_hit {
        1.0
    } else {
        0.0
    };
    let reasoning =
        (0.75 * count(STRONG_REASONING_MARKERS) + 0.35 * count(WEAK_REASONING_MARKERS)).min(1.0);
    let technical = (count(TECHNICAL_TERMS) / 2.0).min(1.0);
    let multi_part = if text.matches('?').count() >= 2 {
        1.0
    } else {
        0.0
    };
    let question = ((count(QUESTION_MARKERS) + multi_part) / 2.0).min(1.0);
    let tokens = (estimate_tokens(text, model) as f64 / TOKEN_SATURATION).min(1.0);
    let simple = (count(SIMPLE_INDICATORS) / 2.0).min(1.0);

    (W_CODE * code
        + W_REASONING * reasoning
        + W_TECHNICAL * technical
        + W_QUESTION * question
        + W_TOKENS * tokens
        - W_SIMPLE * simple)
        .clamp(0.0, 1.0)
}

/// Map a score to a tier using the configured (or default) boundaries.
fn tier_for_score(score: f64, config: &ComplexityRouterConfig) -> Tier {
    let boundary = |tier: &str, default: f64| {
        config
            .tier_boundaries
            .as_ref()
            .and_then(|b| b.get(tier))
            .copied()
            .unwrap_or(default)
    };
    let simple_max = boundary("simple", DEFAULT_SIMPLE_MAX);
    let medium_max = boundary("medium", DEFAULT_MEDIUM_MAX);
    let complex_max = boundary("complex", DEFAULT_COMPLEX_MAX);
    if score < simple_max {
        Tier::Simple
    } else if score < medium_max {
        Tier::Medium
    } else if score < complex_max {
        Tier::Complex
    } else {
        Tier::Reasoning
    }
}

/// Case-insensitive keyword matcher over a request text.
///
/// The marker is trimmed and lowercased first; the trimmed marker's shape
/// then picks the match mode:
///
/// - **Word-ish markers** (letters, digits, spaces only — e.g. `"prove"`,
///   `"step by step"`, `"def"`, `"fn"`): matched on word boundaries so
///   `"hi"` does not match inside `"this"`, `"easy"` does not match
///   `"uneasy"`, and `"def"` does not match `"defendant"`. Whitespace is
///   collapsed, so multi-word markers match across newlines/double spaces.
///   (A trailing-space spelling such as `"def "` therefore matches as the
///   word `"def"` — the space is not part of the match.)
/// - **Symbol markers** (containing punctuation after trimming — e.g.
///   `"```"`, `"#include"`, `"console.log"`): matched as raw substrings,
///   since word boundaries are meaningless for symbols.
///
/// Shared with the adaptive router's request-type classification
/// (`pub(crate)`); the matching semantics must stay identical for both
/// classifiers.
pub(crate) struct TextMatcher {
    /// The lowercased text (for raw substring matches).
    raw: String,
    /// The lowercased text with every non-alphanumeric character replaced
    /// by a space, whitespace runs collapsed, padded with a leading and
    /// trailing space (for word-boundary matches). Replacing punctuation
    /// (instead of only splitting on whitespace) keeps punctuation-glued
    /// tokens matchable: `"hello,"` must still match `"hello"`.
    words: String,
}

impl TextMatcher {
    pub(crate) fn new(text: &str) -> Self {
        let raw = text.to_lowercase();
        let stripped: String = raw
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();
        let collapsed = stripped.split_whitespace().collect::<Vec<_>>().join(" ");
        Self {
            raw,
            words: format!(" {collapsed} "),
        }
    }

    pub(crate) fn matches(&self, marker: &str) -> bool {
        let marker = marker.trim().to_lowercase();
        if marker.is_empty() {
            return false;
        }
        if marker.chars().all(|c| c.is_alphanumeric() || c == ' ') {
            let collapsed = marker.split_whitespace().collect::<Vec<_>>().join(" ");
            self.words.contains(&format!(" {collapsed} "))
        } else {
            self.raw.contains(&marker)
        }
    }
}

/// Extract the request text used for complexity classification from a chat
/// request body: all message contents concatenated (string content and
/// `text` parts of multipart content), capped at 16 KiB so classification
/// stays cheap on huge prompts.
pub fn extract_request_text(body: &Value) -> String {
    // Byte cap: `String::len()` is bytes, and truncation below cuts at a
    // byte offset (rounded down to a char boundary).
    const MAX_BYTES: usize = 16 * 1024;
    let mut out = String::new();
    let Some(messages) = body.get("messages").and_then(|m| m.as_array()) else {
        return out;
    };
    for message in messages {
        match message.get("content") {
            Some(Value::String(s)) => {
                out.push_str(s);
                out.push('\n');
            }
            Some(Value::Array(parts)) => {
                for part in parts {
                    if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            out.push_str(text);
                            out.push('\n');
                        }
                    }
                }
            }
            _ => {}
        }
        if out.len() >= MAX_BYTES {
            // Truncate on a char boundary: `String::truncate` panics when
            // the cut point splits a multibyte character (CJK, emoji).
            let mut end = MAX_BYTES.min(out.len());
            while end > 0 && !out.is_char_boundary(end) {
                end -= 1;
            }
            out.truncate(end);
            break;
        }
    }
    out
}

/// The complexity routing strategy: classify the request, restrict the
/// candidate list to the tier's model pool, and pick randomly within it
/// (via [`SimpleShuffle`], so per-deployment weights are honored).
///
/// The router expands candidates (including aliases and wildcards), then
/// calls the shared resolved-pool picker. Standalone `pick` restricts its
/// input by literal tier model names, classifying on demand unless
/// [`RequestContext::pre_call`](crate::router::strategy::PreCallData::complexity_classification)
/// already carries a classification. When the tier pool has no candidates,
/// the provided candidate list is used.
#[derive(Debug)]
pub struct ComplexityRouter {
    config: ComplexityRouterConfig,
}

impl ComplexityRouter {
    /// Create a complexity router over the given configuration.
    pub fn new(config: ComplexityRouterConfig) -> Self {
        Self { config }
    }

    /// Classify a request context (exposed for tests and observability).
    pub fn classify(&self, ctx: &RequestContext) -> (Tier, f64) {
        classify(ctx, &self.config)
    }
}

impl RoutingStrategy for ComplexityRouter {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        // Reuse the router's pre-computed classification when present —
        // classifying (matcher build + token estimate) is the most
        // expensive part of a complexity pick.
        let (tier, score) = ctx
            .pre_call
            .complexity_classification
            .unwrap_or_else(|| self.classify(ctx));
        let pool: Vec<&Deployment> = match self.config.tiers.get(tier.as_str()) {
            Some(models) => deployments
                .iter()
                .filter(|d| models.iter().any(|m| m == &d.model_name))
                .copied()
                .collect(),
            None => Vec::new(),
        };
        if pool.is_empty() {
            pick_resolved_pool(deployments, ctx, (tier, score))
        } else {
            pick_resolved_pool(&pool, ctx, (tier, score))
        }
    }
}

/// Pick from a pool whose model names have already been resolved. The router
/// can expand aliases/wildcards; standalone strategy picks resolve literal
/// names. Neither caller should re-interpret the other's resolved candidates.
pub(crate) fn pick_resolved_pool<'a>(
    deployments: &[&'a Deployment],
    ctx: &RequestContext,
    (tier, score): (Tier, f64),
) -> Option<&'a Deployment> {
    let picked = SimpleShuffle.pick(deployments, ctx);
    tracing::debug!(
        event = "complexity_route",
        tier = tier.as_str(),
        score = score,
        model = %ctx.model,
        picked = picked.map(|d| d.model_name.as_str()).unwrap_or(""),
        "request classified and routed by complexity"
    );
    picked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::OxllmParams;
    use std::collections::HashMap;

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

    fn ctx_with_text(text: &str) -> RequestContext {
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.request_text = Some(text.to_string());
        ctx
    }

    fn default_config() -> ComplexityRouterConfig {
        ComplexityRouterConfig::default()
    }

    fn tier_of(text: &str) -> Tier {
        let ctx = ctx_with_text(text);
        classify(&ctx, &default_config()).0
    }

    // ------------------------------------------------------------------
    // Tier classification
    // ------------------------------------------------------------------

    #[test]
    fn simple_text_classifies_simple() {
        assert_eq!(tier_of("hello"), Tier::Simple);
        assert_eq!(tier_of("hi there"), Tier::Simple);
        assert_eq!(tier_of("what is 2+2?"), Tier::Simple);
    }

    #[test]
    fn empty_or_missing_text_is_simple() {
        let ctx = RequestContext::new("gpt-4o");
        let (tier, score) = classify(&ctx, &default_config());
        assert_eq!(tier, Tier::Simple);
        assert_eq!(score, 0.0);
        assert_eq!(tier_of(""), Tier::Simple);
    }

    #[test]
    fn complex_text_classifies_complex() {
        assert_eq!(
            tier_of("analyze the tradeoffs of microservices vs monoliths"),
            Tier::Complex
        );
    }

    #[test]
    fn reasoning_text_classifies_reasoning() {
        assert_eq!(tier_of("prove that P != NP"), Tier::Reasoning);
        assert_eq!(
            tier_of("derive the closed form of the recurrence, step by step"),
            Tier::Reasoning
        );
    }

    #[test]
    fn code_presence_boosts_score() {
        let ctx_plain = ctx_with_text("sort a list of numbers");
        let ctx_code = ctx_with_text(
            "sort a list of numbers\n```python\ndef sort(xs):\n    import heapq\n```",
        );
        let cfg = default_config();
        let (_, plain) = classify(&ctx_plain, &cfg);
        let (_, with_code) = classify(&ctx_code, &cfg);
        assert!(
            with_code > plain,
            "code presence must raise the score ({with_code} <= {plain})"
        );
        assert!(with_code >= W_CODE, "code dimension must contribute");
    }

    #[test]
    fn code_markers_do_not_match_ordinary_prose() {
        // Bare English words that used to be code markers ("from",
        // "class", "import", "function") must not fire the code
        // dimension when they appear in ordinary prose.
        let cfg = default_config();
        let score_of = |text: &str| classify(&ctx_with_text(text), &cfg).1;
        for prose in [
            "where is the train from here?",
            "i have a math class today",
            "import tariffs rose again",
            "the doctor checked my liver function",
        ] {
            assert!(
                score_of(prose) < W_CODE,
                "prose must not trip the code dimension: {prose:?} scored {}",
                score_of(prose)
            );
        }
        // A Python import statement still reads as code: "from" and
        // "import" together are a distinctive shape.
        assert!(
            score_of("from itertools import product") >= W_CODE,
            "python import must trip the code dimension"
        );
    }

    #[test]
    fn reasoning_markers_boost_score() {
        let cfg = default_config();
        let ctx_plain = ctx_with_text("tell me about caches");
        let ctx_reason = ctx_with_text("explain why this fails, step by step");
        let (_, plain) = classify(&ctx_plain, &cfg);
        let (_, reasoned) = classify(&ctx_reason, &cfg);
        assert!(reasoned > plain);
        // "explain why" + "step by step" saturate the reasoning dimension.
        assert!(reasoned >= W_REASONING * 0.99);
    }

    #[test]
    fn technical_terms_boost_score() {
        let cfg = default_config();
        let ctx_plain = ctx_with_text("how do i make toast");
        let ctx_tech = ctx_with_text("how do i design an api with a database and an algorithm");
        let (_, plain) = classify(&ctx_plain, &cfg);
        let (_, tech) = classify(&ctx_tech, &cfg);
        assert!(tech > plain, "technical terms must raise the score");
    }

    #[test]
    fn simple_indicators_lower_score() {
        let cfg = default_config();
        let ctx_plain = ctx_with_text("summarize this document");
        let ctx_simple = ctx_with_text("hello, quick and easy question: summarize this");
        let (_, plain) = classify(&ctx_plain, &cfg);
        let (_, simple) = classify(&ctx_simple, &cfg);
        assert!(
            simple < plain,
            "simple indicators must lower the score ({simple} >= {plain})"
        );
    }

    #[test]
    fn simple_indicator_word_boundary_no_false_match() {
        // "hi" must not match inside "this"; "easy" must not match "uneasy".
        let cfg = default_config();
        let ctx = ctx_with_text("analyze this uneasy protocol design");
        let (_, score) = classify(&ctx, &cfg);
        assert!(
            score > 0.0,
            "content words must score; no simple-indicator subtraction"
        );
    }

    #[test]
    fn word_boundary_matching_strips_punctuation() {
        // Punctuation glued to a token must not defeat word-boundary
        // matching; partial words still must not match.
        let matcher = TextMatcher::new("hello, prove. analyze: world");
        assert!(matcher.matches("hello"));
        assert!(matcher.matches("prove"));
        assert!(matcher.matches("analyze"));
        assert!(!matcher.matches("hell"));
        assert!(!matcher.matches("pro"));
    }

    #[test]
    fn word_markers_do_not_match_prefixes_inside_longer_words() {
        // Pin the word-boundary contract for the ex-trailing-space code
        // markers: "def" must not match "defendant", "fn" must not match
        // "fnord" — in both the plain and the legacy trailing-space
        // spelling (which trims to the same word marker).
        let matcher =
            TextMatcher::new("the defendant mentioned import tariffs and the fnord module");
        assert!(!matcher.matches("def"));
        assert!(!matcher.matches("fn"));
        assert!(!matcher.matches("def "));
        assert!(!matcher.matches("fn "));
        assert!(matcher.matches("import"));
        // The standalone words still match inside actual code text.
        let code = TextMatcher::new("def sort(xs):\n    fn helper() {}\n");
        assert!(code.matches("def"));
        assert!(code.matches("fn"));
    }

    #[test]
    fn punctuation_adjacent_keywords_classify_the_same() {
        let cfg = default_config();
        let tier_of = |text: &str| classify(&ctx_with_text(text), &cfg).0;
        assert_eq!(tier_of("hello,"), tier_of("hello"));
        assert_eq!(tier_of("prove."), Tier::Reasoning);
        assert_eq!(tier_of("analyze:"), tier_of("analyze"));
        assert_eq!(tier_of("analyze"), Tier::Medium);
    }

    #[test]
    fn token_count_boosts_score() {
        let cfg = default_config();
        let short = ctx_with_text("tell me a story");
        let long_text = format!("tell me a story {}", "once upon a time ".repeat(2000));
        let long = ctx_with_text(&long_text);
        let (_, short_score) = classify(&short, &cfg);
        let (_, long_score) = classify(&long, &cfg);
        assert!(
            long_score > short_score,
            "larger requests must score higher ({long_score} <= {short_score})"
        );
    }

    #[test]
    fn multi_part_question_boosts_score() {
        let cfg = default_config();
        let single = ctx_with_text("what is the capital of france");
        let multi = ctx_with_text("what is the capital of france? and of italy? and spain?");
        let (_, single_score) = classify(&single, &cfg);
        let (_, multi_score) = classify(&multi, &cfg);
        assert!(multi_score > single_score);
    }

    // ------------------------------------------------------------------
    // Boundaries and escalation
    // ------------------------------------------------------------------

    #[test]
    fn tier_boundaries_from_config() {
        let cfg = ComplexityRouterConfig {
            tier_boundaries: Some(HashMap::from([
                ("simple".to_string(), 0.5),
                ("medium".to_string(), 0.9),
                ("complex".to_string(), 0.95),
            ])),
            ..Default::default()
        };
        // This text scores ~0.46 by default (COMPLEX); with a 0.5 simple
        // ceiling it must classify as SIMPLE.
        let ctx = ctx_with_text("analyze the tradeoffs of microservices vs monoliths");
        let (tier, score) = classify(&ctx, &cfg);
        assert!(score >= 0.35, "fixture text must keep its default score");
        assert_eq!(tier, Tier::Simple, "custom boundaries must apply");
        // A second known input: scores 0.43025 with default boundaries
        // (COMPLEX). The raised simple ceiling (0.5) must pull it into
        // SIMPLE — hardcoded expectations, not a recomputation of the
        // classifier's logic.
        let ctx2 = ctx_with_text("explain how a database protocol works in detail here");
        let (default_tier, default_score) = classify(&ctx2, &default_config());
        assert!(
            (default_score - 0.43025).abs() < 1e-9,
            "fixture score drifted (got {default_score}); update the hardcoded expectations"
        );
        assert_eq!(default_tier, Tier::Complex);
        assert_eq!(
            classify(&ctx2, &cfg).0,
            Tier::Simple,
            "score 0.43025 < 0.5 simple ceiling must classify as SIMPLE"
        );
    }

    #[test]
    fn boundary_edge_values() {
        let cfg = default_config();
        // Scores exactly on a ceiling fall into the HIGHER tier.
        assert_eq!(tier_for_score(0.15, &cfg), Tier::Medium);
        assert_eq!(tier_for_score(0.35, &cfg), Tier::Complex);
        assert_eq!(tier_for_score(0.60, &cfg), Tier::Reasoning);
        assert_eq!(tier_for_score(0.149, &cfg), Tier::Simple);
    }

    #[test]
    fn escalation_keywords_bump_tier_up() {
        let cfg = ComplexityRouterConfig {
            escalation_keywords: Some(vec!["urgent".to_string()]),
            ..Default::default()
        };
        let ctx = ctx_with_text("hello, this is urgent");
        let (tier, _) = classify(&ctx, &cfg);
        assert_eq!(tier, Tier::Medium, "escalation bumps simple → medium");

        let ctx = ctx_with_text("analyze the tradeoffs of microservices vs monoliths, urgent");
        let (tier, _) = classify(&ctx, &cfg);
        assert_eq!(
            tier,
            Tier::Reasoning,
            "escalation bumps complex → reasoning"
        );

        // Without the keyword there is no bump.
        let ctx = ctx_with_text("hello there");
        let (tier, _) = classify(&ctx, &cfg);
        assert_eq!(tier, Tier::Simple);
    }

    #[test]
    fn escalation_never_exceeds_reasoning() {
        let cfg = ComplexityRouterConfig {
            escalation_keywords: Some(vec!["prove".to_string()]),
            ..Default::default()
        };
        let ctx = ctx_with_text("prove that P != NP");
        let (tier, _) = classify(&ctx, &cfg);
        assert_eq!(tier, Tier::Reasoning);
    }

    // ------------------------------------------------------------------
    // Model pool selection
    // ------------------------------------------------------------------

    fn pool_config() -> ComplexityRouterConfig {
        ComplexityRouterConfig {
            tiers: HashMap::from([
                ("simple".to_string(), vec!["gpt-4o-mini".to_string()]),
                ("complex".to_string(), vec!["gpt-4o".to_string()]),
                ("reasoning".to_string(), vec!["o3".to_string()]),
            ]),
            ..Default::default()
        }
    }

    fn three_pool_deps() -> [Deployment; 3] {
        [make_dep("gpt-4o-mini"), make_dep("gpt-4o"), make_dep("o3")]
    }

    #[test]
    fn pick_selects_from_simple_pool() {
        let router = ComplexityRouter::new(pool_config());
        let deps = three_pool_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_text("hello");
        for _ in 0..20 {
            let picked = router.pick(&refs, &ctx).expect("should pick");
            assert_eq!(picked.model_name, "gpt-4o-mini");
        }
    }

    #[test]
    fn pick_selects_from_complex_pool() {
        let router = ComplexityRouter::new(pool_config());
        let deps = three_pool_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_text("analyze the tradeoffs of microservices vs monoliths");
        for _ in 0..20 {
            let picked = router.pick(&refs, &ctx).expect("should pick");
            assert_eq!(picked.model_name, "gpt-4o");
        }
    }

    #[test]
    fn pick_selects_from_reasoning_pool() {
        let router = ComplexityRouter::new(pool_config());
        let deps = three_pool_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_text("prove that P != NP");
        for _ in 0..20 {
            let picked = router.pick(&refs, &ctx).expect("should pick");
            assert_eq!(picked.model_name, "o3");
        }
    }

    #[test]
    fn pick_falls_back_to_all_candidates_when_tier_pool_missing() {
        // "reasoning" pool configured but no matching deployments among
        // candidates → the full candidate list is used.
        let router = ComplexityRouter::new(pool_config());
        let deps = [make_dep("gpt-4o-mini"), make_dep("gpt-4o")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_text("prove that P != NP");
        let mut saw_mini = false;
        let mut saw_full = false;
        for _ in 0..100 {
            let picked = router.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "gpt-4o-mini" {
                saw_mini = true;
            }
            if picked.model_name == "gpt-4o" {
                saw_full = true;
            }
        }
        assert!(
            saw_mini && saw_full,
            "missing tier pool must fall back to the full candidate list"
        );
    }

    #[test]
    fn pick_falls_back_when_tier_not_configured() {
        // MEDIUM tier has no configured pool → full candidate list.
        let router = ComplexityRouter::new(pool_config());
        let deps = [make_dep("gpt-4o-mini")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_text("explain how photosynthesis works in a leaf");
        let picked = router.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "gpt-4o-mini");
    }

    #[test]
    fn pick_reuses_precomputed_classification() {
        // The router hands its classification to the strategy via
        // `ctx.pre_call`; `pick` must trust it instead of re-classifying
        // (the text here says "hello" but the pre-computed tier is
        // REASONING, so the reasoning pool must be picked).
        let router = ComplexityRouter::new(pool_config());
        let deps = three_pool_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let mut ctx = ctx_with_text("hello");
        ctx.pre_call.complexity_classification = Some((Tier::Reasoning, 0.9));
        let picked = router.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "o3");
    }

    #[test]
    fn pick_classifies_when_no_precomputed_classification() {
        // Standalone use (no router expansion): classify on demand.
        let router = ComplexityRouter::new(pool_config());
        let deps = three_pool_deps();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_text("prove that P != NP");
        assert!(ctx.pre_call.complexity_classification.is_none());
        let picked = router.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "o3");
    }

    #[test]
    fn pick_empty_returns_none() {
        let router = ComplexityRouter::new(pool_config());
        let ctx = ctx_with_text("hello");
        assert!(router.pick(&[], &ctx).is_none());
    }

    // ------------------------------------------------------------------
    // extract_request_text
    // ------------------------------------------------------------------

    #[test]
    fn extract_request_text_concatenates_messages() {
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hello world"}
            ]
        });
        let text = extract_request_text(&body);
        assert!(text.contains("you are helpful"));
        assert!(text.contains("hello world"));
    }

    #[test]
    fn extract_request_text_handles_multipart_and_missing() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    {"type": "text", "text": "part two"}
                ]
            }]
        });
        let text = extract_request_text(&body);
        assert!(text.contains("part one"));
        assert!(text.contains("part two"));
        assert!(!text.contains("image_url"));

        assert!(extract_request_text(&serde_json::json!({"model": "gpt-4o"})).is_empty());
        assert!(extract_request_text(&serde_json::json!({"messages": []})).is_empty());
    }

    #[test]
    fn extract_request_text_caps_length() {
        let big = "x".repeat(64 * 1024);
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": big}]
        });
        assert_eq!(extract_request_text(&body).len(), 16 * 1024);
    }

    #[test]
    fn extract_request_text_truncates_on_char_boundary() {
        // Multibyte content crossing the 16 KiB cap must not panic and
        // must truncate at a char boundary.
        let cjk = "日".repeat(6000); // 3 bytes/char → 18_000 bytes
        let emoji = "🎉".repeat(5000); // 4 bytes/char → 20_000 bytes
        for fill in [cjk, emoji] {
            let body = serde_json::json!({
                "messages": [{"role": "user", "content": fill}]
            });
            let text = extract_request_text(&body);
            assert!(
                text.len() <= 16 * 1024,
                "truncated text must fit the cap, got {}",
                text.len()
            );
            assert!(
                text.len() > 16 * 1024 - 8,
                "truncation must land within a char of the cap, got {}",
                text.len()
            );
        }
    }
}
