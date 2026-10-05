//! Tag-based routing strategy.
//!
//! Picks deployments based on a `tag` field carried in the
//! [`RequestContext`]. Deployments expose their tags via `model_info.tags`
//! and optional regex patterns via `model_info.tag_regex` (a list of regex
//! patterns matched against the request tag). When the request carries no
//! tag, or no deployment matches a positive tag, the strategy falls back to
//! [`SimpleShuffle`] within the deployments allowed by required/excluded tags.
//!
//! Rich matching (H4):
//! - **Exact match** (backward compatible): a deployment whose `tags`
//!   contains the request tag is selected.
//! - **Regex match**: a deployment with `tag_regex` in `model_info` matches
//!   when any pattern matches the request tag. Patterns support a useful
//!   subset of regex: literal chars, `.` (any char), `*` (zero-or-more of
//!   the preceding element), `\x` escapes (a literal `x`), and `^`/`$`
//!   anchors. Unanchored patterns
//!   match anywhere in the tag (substring semantics, like `re.search`).
//! - **Exclude (`!` prefix)**: a request tag starting with `!` excludes
//!   deployments matching the remainder of the tag (exact or regex), then
//!   picks from the survivors.
//! - **Require (`&` prefix)**: every selected deployment must match the
//!   remainder of the tag (exact or regex).
//! - **Default tag**: applied at the router level — untagged requests use
//!   `RouterConfig::default_tag` when set.
//!
//! Multi-tag requests (carried in [`RequestContext::tags`]) use the richer
//! semantics of `TagBased::pick_multi`: positive tags subset-match a
//! deployment's tags, `&tag` requires a deployment tag, `!tag` excludes
//! deployments matching it, and deployments tagged `default` form the
//! fallback pool when no positive tags match. Required/excluded tags apply
//! to every fallback pool. Unsatisfiable constraints return an error from
//! [`RoutingStrategy::try_pick`] (`None` from the compatibility `pick` API).

use crate::deployment::Deployment;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::strategy::{RequestContext, RoutingStrategy};

/// A tag-based routing strategy.
pub struct TagBased {
    shuffle: SimpleShuffle,
}

impl TagBased {
    /// Create a new tag-based strategy.
    pub fn new() -> Self {
        Self {
            shuffle: SimpleShuffle,
        }
    }
}

impl Default for TagBased {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingStrategy for TagBased {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        self.try_pick(deployments, ctx).ok().flatten()
    }

    fn try_pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> crate::Result<Option<&'a Deployment>> {
        let deployments = constrain_deployments(deployments.to_vec(), ctx)?;
        // Tag-list path (body `metadata.tags` + header values): run the
        // full require/exclude/subset matching over the whole set,
        // including one-element lists (so `&tag` / `!tag` work with a
        // single carried tag, and the `default` deployment pool applies).
        // The legacy path below serves contexts that only set
        // `RequestContext::tag` (no list).
        if !ctx.tags.is_empty() {
            return Ok(self.pick_multi(&deployments, ctx));
        }
        let Some(tag) = ctx.tag.as_deref() else {
            return Ok(self.shuffle.pick(&deployments, ctx));
        };
        // Hard selectors were already applied by `constrain_deployments`.
        if tag.is_empty() || tag.starts_with('!') || tag.starts_with('&') {
            return Ok(self.shuffle.pick(&deployments, ctx));
        }
        let matching: Vec<&Deployment> = deployments
            .iter()
            .copied()
            .filter(|d| deployment_matches_tag(d, tag))
            .collect();
        if matching.is_empty() {
            return Ok(self.shuffle.pick(&deployments, ctx));
        }
        Ok(self.shuffle.pick(&matching, ctx))
    }
}

/// Apply hard selectors before any preference/fallback selection. The router
/// also uses this before priority tiers and pre-call checks, so a forbidden
/// high-priority deployment cannot hide an eligible lower-priority one.
pub(crate) fn constrain_deployments<'a>(
    mut deployments: Vec<&'a Deployment>,
    ctx: &RequestContext,
) -> crate::Result<Vec<&'a Deployment>> {
    if deployments.is_empty() {
        return Ok(deployments);
    }
    let tags = ctx
        .tags
        .iter()
        .map(String::as_str)
        .chain(ctx.tag.as_deref().filter(|_| ctx.tags.is_empty()));
    let mut required = Vec::new();
    let mut excluded = Vec::new();
    for tag in tags {
        if let Some(tag) = tag.strip_prefix('&').filter(|t| !t.is_empty()) {
            required.push(tag);
        } else if let Some(tag) = tag.strip_prefix('!').filter(|t| !t.is_empty()) {
            excluded.push(tag);
        }
    }
    deployments.retain(|d| {
        required.iter().all(|tag| deployment_matches_tag(d, tag))
            && !excluded.iter().any(|tag| deployment_matches_tag(d, tag))
    });
    if deployments.is_empty() {
        return Err(crate::Error::BadRequest(
            "no available deployment satisfies the required/excluded routing tags".to_string(),
        ));
    }
    Ok(deployments)
}

/// The multi-tag pick:
///
/// - required/excluded tags have already constrained the candidate list;
/// - the remaining positive tags subset-match: a deployment qualifies when
///   every positive request tag matches one of its `tags`/`tag_regex`
///   (a request's positive tags must be a subset of the deployment's);
/// - deployments tagged `default` form the fallback pool when nothing
///   matches; otherwise the allowed candidates shuffle. Neither fallback
///   can reintroduce deployments rejected by a hard selector.
impl TagBased {
    fn pick_multi<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        let positive: Vec<&str> = ctx
            .tags
            .iter()
            .map(String::as_str)
            .filter(|t| !t.starts_with('&') && !t.starts_with('!'))
            .collect();
        let survivors: Vec<&Deployment> = deployments
            .iter()
            .copied()
            .filter(|d| positive.iter().all(|p| deployment_matches_tag(d, p)))
            .collect();
        if survivors.is_empty() {
            // Fallbacks draw only from the already-constrained candidates.
            let defaults: Vec<&Deployment> = deployments
                .iter()
                .copied()
                .filter(|d| {
                    d.model_info
                        .as_ref()
                        .is_some_and(|i| i.tags.iter().any(|t| t == "default"))
                })
                .collect();
            if defaults.is_empty() {
                return self.shuffle.pick(deployments, ctx);
            }
            return self.shuffle.pick(&defaults, ctx);
        }
        self.shuffle.pick(&survivors, ctx)
    }
}

/// Upper bound on the tag length fed to the regex matcher. The matcher is
/// linear in the tag length, but a pathologically long `x-oxllm-tag` header
/// (attacker-controlled) should not pin a core scanning megabytes per pick.
/// Legitimate tags are short selectors (`canary`, `prod-blue-v2`), so 8 KiB
/// is far beyond any real value.
const MAX_TAG_LEN: usize = 8 * 1024;

/// Check if a deployment advertises the given tag via exact `tags` match
/// or via `tag_regex` patterns in `model_info`.
fn deployment_matches_tag(deployment: &Deployment, tag: &str) -> bool {
    let Some(info) = &deployment.model_info else {
        return false;
    };
    // Exact-match fast path (backward compatible).
    if info.tags.iter().any(|t| t == tag) {
        return true;
    }
    // Regex patterns from model_info.tag_regex (an array of pattern strings).
    if let Some(patterns) = info.extra.get("tag_regex").and_then(|v| v.as_array()) {
        let capped = cap_tag_len(tag);
        for p in patterns {
            if let Some(pattern) = p.as_str() {
                if regex_match(capped, pattern) {
                    return true;
                }
            }
        }
    }
    false
}

/// Truncate `tag` to at most [`MAX_TAG_LEN`] bytes on a UTF-8 char boundary,
/// matching the truncation style used by the complexity classifier. The exact
/// match fast path uses the full tag; only the regex path is bounded.
fn cap_tag_len(tag: &str) -> &str {
    if tag.len() <= MAX_TAG_LEN {
        return tag;
    }
    let mut end = MAX_TAG_LEN;
    while end > 0 && !tag.is_char_boundary(end) {
        end -= 1;
    }
    &tag[..end]
}

/// A minimal regex matcher supporting the subset of regex syntax most
/// useful for tag routing: literal characters, `.` (any single character),
/// `*` (zero or more of the preceding element), and `^`/`$` anchors.
/// Unanchored patterns perform a substring search (matching `re.search`
/// semantics). This avoids a hard dependency on the `regex` crate while
/// covering the common tag-pattern shapes (`canary`, `^prod-`, `.*-v2`,
/// `prod-.*`).
///
/// The matcher compiles the pattern to a Thompson NFA and simulates it
/// **non-backtrackingly** (forward state-set simulation), so match time is
/// `O(text_len × pattern_len)` with no catastrophic backtracking. It
/// iterates over `text.chars()`, so multibyte (non-ASCII) tags never hit a
/// byte-boundary panic — the input is attacker-controlled (`x-oxllm-tag`
/// header) and must never crash the worker.
fn regex_match(text: &str, pattern: &str) -> bool {
    let anchored_start = pattern.starts_with('^');
    let anchored_end = pattern.ends_with('$') && pattern != "^" && !pattern.ends_with("\\$");
    // Strip the leading `^` and trailing `$` to get the core pattern. The
    // trailing `$` is only an anchor (and thus only stripped) when it is not
    // escaped; `\$` denotes a literal dollar sign.
    let core = pattern.strip_prefix('^').unwrap_or(pattern);
    let core = if anchored_end {
        core.strip_suffix('$').unwrap_or(core)
    } else {
        core
    };

    let insts = match compile(core) {
        Some(t) => t,
        None => {
            // Fall back to substring match for unparseable patterns.
            return text.contains(core);
        }
    };
    nfa_run(&insts, text, anchored_start, anchored_end)
}

/// A compiled regex instruction (Thompson NFA), supporting the tag-routing
/// subset: literal characters, `.` (any single character), and `*` (zero or
/// more of the preceding element).
#[derive(Debug, Clone, Copy)]
enum Inst {
    /// Match a literal character, then advance to the next instruction.
    Char(char),
    /// Match any single character, then advance to the next instruction.
    Any,
    /// Epsilon: branch to two instructions (used for the `*` split).
    Split(usize, usize),
    /// Epsilon: jump to an instruction (used to loop a `*` back).
    Jmp(usize),
    /// Accept (the pattern matched).
    Match,
}

/// Compile the core pattern into a Thompson NFA instruction list, returning
/// `None` for patterns outside the supported subset (e.g. a leading `*`).
fn compile(pattern: &str) -> Option<Vec<Inst>> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut insts = Vec::with_capacity(chars.len() * 2);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' {
            // `\x` escapes: treat the next char as a literal (e.g. `\.` or
            // `\$`). A trailing lone backslash matches a literal backslash.
            let lit = chars.get(i + 1).copied().unwrap_or('\\');
            if chars.get(i + 1).is_some() {
                i += 2;
            } else {
                i += 1;
            }
            let star = i < chars.len() && chars[i] == '*';
            if star {
                let l1 = insts.len();
                insts.push(Inst::Split(0, 0));
                let l2 = insts.len();
                insts.push(Inst::Char(lit));
                insts.push(Inst::Jmp(l1));
                let l3 = insts.len();
                insts[l1] = Inst::Split(l2, l3);
                i += 1;
            } else {
                insts.push(Inst::Char(lit));
            }
            continue;
        }
        if c == '*' {
            // A leading '*' or '**' is invalid in this subset.
            return None;
        }
        let any = c == '.';
        // Peek for a following '*'.
        let star = i + 1 < chars.len() && chars[i + 1] == '*';
        if star {
            // `e*` = split(L2, L3) ; L2: e ; jmp L1 ; L3:
            let l1 = insts.len();
            insts.push(Inst::Split(0, 0));
            let l2 = insts.len();
            insts.push(if any { Inst::Any } else { Inst::Char(c) });
            insts.push(Inst::Jmp(l1));
            let l3 = insts.len();
            insts[l1] = Inst::Split(l2, l3);
            i += 2;
        } else {
            insts.push(if any { Inst::Any } else { Inst::Char(c) });
            i += 1;
        }
    }
    insts.push(Inst::Match);
    Some(insts)
}

/// Simulate the NFA non-backtrackingly over `text`.
///
/// `anchored_start` requires the match to begin at position 0; otherwise the
/// start state is re-seeded after each consumed character so the match may
/// begin anywhere (substring semantics). `anchored_end` requires the match to
/// consume all of `text`; otherwise reaching `Match` at any step accepts.
fn nfa_run(insts: &[Inst], text: &str, anchored_start: bool, anchored_end: bool) -> bool {
    let m = insts.len();
    let match_pc = m - 1;
    let mut onlist = vec![false; m];
    let mut clist: Vec<usize> = Vec::with_capacity(m);
    let mut nlist: Vec<usize> = Vec::with_capacity(m);
    let mut eps_stack: Vec<usize> = Vec::with_capacity(m);

    onlist.fill(false);
    add_state(insts, 0, &mut clist, &mut onlist, &mut eps_stack);
    if !anchored_end && onlist[match_pc] {
        return true;
    }
    for c in text.chars() {
        nlist.clear();
        onlist.fill(false);
        for &pc in &clist {
            match insts[pc] {
                Inst::Char(ch) if ch == c => {
                    add_state(insts, pc + 1, &mut nlist, &mut onlist, &mut eps_stack)
                }
                Inst::Any => add_state(insts, pc + 1, &mut nlist, &mut onlist, &mut eps_stack),
                _ => {}
            }
        }
        // Unanchored start: allow the match to begin at this position too.
        if !anchored_start {
            add_state(insts, 0, &mut nlist, &mut onlist, &mut eps_stack);
        }
        std::mem::swap(&mut clist, &mut nlist);
        if !anchored_end && onlist[match_pc] {
            return true;
        }
    }
    // After consuming all input: reachable for both anchored_end (must reach
    // Match exactly here) and a full-consuming non-anchored match.
    onlist[match_pc]
}

/// Add `pc` and every epsilon-reachable instruction to `list`, marking
/// visited instructions in `onlist` so each is added at most once. Uses an
/// explicit stack (no recursion) bounded by the instruction count.
fn add_state(
    insts: &[Inst],
    pc: usize,
    list: &mut Vec<usize>,
    onlist: &mut [bool],
    stack: &mut Vec<usize>,
) {
    stack.clear();
    stack.push(pc);
    while let Some(p) = stack.pop() {
        if onlist[p] {
            continue;
        }
        onlist[p] = true;
        list.push(p);
        match insts[p] {
            Inst::Split(a, b) => {
                stack.push(a);
                stack.push(b);
            }
            Inst::Jmp(t) => stack.push(t),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::{Deployment, ModelInfo, OxllmParams};
    use serde_json::json;

    fn make_dep_with_tags(name: &str, model: &str, tags: Vec<String>) -> Deployment {
        Deployment {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                ..Default::default()
            },
            model_info: Some(ModelInfo {
                tags,
                ..Default::default()
            }),
        }
    }

    fn make_dep_with_regex(name: &str, model: &str, patterns: Vec<&str>) -> Deployment {
        Deployment {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                ..Default::default()
            },
            model_info: Some(ModelInfo {
                tags: Vec::new(),
                extra: {
                    let mut m = serde_json::Map::new();
                    m.insert("tag_regex".to_string(), json!(patterns));
                    m
                },
                ..Default::default()
            }),
        }
    }

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

    #[test]
    fn picks_tagged_deployment_when_tag_matches() {
        let strategy = TagBased::new();
        let d1 = make_dep_with_tags("gpt-4o", "openai/gpt-4o", vec!["canary".to_string()]);
        let d2 = make_dep_with_tags("gpt-4o", "openai/gpt-4o", vec!["stable".to_string()]);
        let deps = [&d1, &d2];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.tag = Some("canary".to_string());
        let picked = strategy.pick(&deps, &ctx).expect("should pick");
        assert_eq!(picked.model_info.as_ref().unwrap().tags, vec!["canary"]);
    }

    #[test]
    fn falls_back_to_shuffle_when_no_tag_set() {
        let strategy = TagBased::new();
        let d1 = make_dep("gpt-4o");
        let d2 = make_dep("gpt-4o");
        let deps = [&d1, &d2];
        let ctx = RequestContext::new("gpt-4o");
        let picked = strategy.pick(&deps, &ctx);
        assert!(picked.is_some());
    }

    #[test]
    fn falls_back_to_shuffle_when_no_deployment_matches_tag() {
        let strategy = TagBased::new();
        let d1 = make_dep_with_tags("gpt-4o", "openai/gpt-4o", vec!["stable".to_string()]);
        let deps = [&d1];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.tag = Some("canary".to_string());
        let picked = strategy.pick(&deps, &ctx).expect("falls back to shuffle");
        assert_eq!(picked.model_name, "gpt-4o");
    }

    #[test]
    fn empty_tag_falls_back_to_shuffle() {
        let strategy = TagBased::new();
        let d1 = make_dep("gpt-4o");
        let deps = [&d1];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.tag = Some(String::new());
        assert!(strategy.pick(&deps, &ctx).is_some());
    }

    #[test]
    fn returns_none_when_no_deployments() {
        let strategy = TagBased::new();
        let ctx = RequestContext::new("gpt-4o");
        assert!(strategy.pick(&[], &ctx).is_none());
    }

    // --- H4: rich tag routing ---

    #[test]
    fn regex_matches_substring_pattern() {
        let strategy = TagBased::new();
        let d1 = make_dep_with_regex("gpt-4o", "openai/gpt-4o", vec!["can"]);
        let deps = [&d1];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.tag = Some("canary".to_string());
        let picked = strategy.pick(&deps, &ctx).expect("regex substring match");
        assert_eq!(picked.model_name, "gpt-4o");
    }

    #[test]
    fn regex_anchored_prefix_pattern() {
        let strategy = TagBased::new();
        let d1 = make_dep_with_regex("gpt-4o", "openai/gpt-4o-a", vec!["^prod-"]);
        let d2 = make_dep_with_regex("gpt-4o", "openai/gpt-4o-b", vec!["^dev-"]);
        let deps = [&d1, &d2];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.tag = Some("prod-canary".to_string());
        let picked = strategy.pick(&deps, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-a");
    }

    #[test]
    fn regex_dot_star_pattern() {
        let strategy = TagBased::new();
        let d1 = make_dep_with_regex("gpt-4o", "openai/gpt-4o-a", vec!["prod-.*-v2"]);
        let d2 = make_dep_with_regex("gpt-4o", "openai/gpt-4o-b", vec!["stable"]);
        let deps = [&d1, &d2];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.tag = Some("prod-blue-v2".to_string());
        let picked = strategy.pick(&deps, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-a");
    }

    #[test]
    fn regex_exact_anchor_pattern() {
        let strategy = TagBased::new();
        let d1 = make_dep_with_regex("gpt-4o", "openai/gpt-4o-a", vec!["^canary$"]);
        let d2 = make_dep_with_regex("gpt-4o", "openai/gpt-4o-b", vec!["^stable$"]);
        let deps = [&d1, &d2];
        let mut ctx = RequestContext::new("gpt-4o");
        // Substring "canary-blue" must NOT match the anchored `^canary$`.
        ctx.tag = Some("canary-blue".to_string());
        // Falls back to shuffle since neither pattern matches.
        assert!(strategy.pick(&deps, &ctx).is_some());
        ctx.tag = Some("canary".to_string());
        let picked = strategy.pick(&deps, &ctx).expect("anchored exact match");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-a");
    }

    #[test]
    fn exclude_tag_drops_matching_deployments() {
        let strategy = TagBased::new();
        let d1 = make_dep_with_tags("gpt-4o", "openai/gpt-4o-a", vec!["canary".to_string()]);
        let d2 = make_dep_with_tags("gpt-4o", "openai/gpt-4o-b", vec!["stable".to_string()]);
        let deps = [&d1, &d2];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.tag = Some("!canary".to_string());
        // Every pick must land on the survivor (stable), never the excluded.
        for _ in 0..20 {
            let picked = strategy.pick(&deps, &ctx).expect("should pick survivor");
            assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-b");
        }
    }

    #[test]
    fn exclude_tag_excludes_all_rejects_request() {
        let strategy = TagBased::new();
        let d1 = make_dep_with_tags("gpt-4o", "openai/gpt-4o-a", vec!["canary".to_string()]);
        let d2 = make_dep_with_tags("gpt-4o", "openai/gpt-4o-b", vec!["canary".to_string()]);
        let deps = [&d1, &d2];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.tag = Some("!canary".to_string());
        assert!(strategy.pick(&deps, &ctx).is_none());
        let err = strategy
            .try_pick(&deps, &ctx)
            .expect_err("all candidates excluded");
        assert!(err.to_string().contains("required/excluded routing tags"));
    }

    #[test]
    fn regex_match_unit_tests() {
        // Core engine sanity: substring, anchors, wildcards.
        assert!(regex_match("canary", "can"));
        assert!(regex_match("prod-blue", "^prod-"));
        assert!(regex_match("blue-prod", "prod$"));
        assert!(regex_match("prod-blue-v2", "prod-.*-v2"));
        assert!(regex_match("canary", "^canary$"));
        assert!(!regex_match("canary-blue", "^canary$"));
        assert!(regex_match("aaab", "a*b"));
        assert!(regex_match("xyz", "x.z"));
        assert!(!regex_match("xyz", "x.b"));
        // Star-repeats: zero-or-more semantics.
        assert!(regex_match("aaa", "a*"));
        assert!(regex_match("", "a*"));
        assert!(regex_match("", "^$"));
        assert!(!regex_match("x", "^$"));
    }

    #[test]
    fn regex_match_handles_non_ascii_tags_without_panicking() {
        // The `x-oxllm-tag` header is attacker-controlled and may carry
        // multibyte UTF-8. The matcher iterates over `char`s (never slicing
        // at a raw byte offset), so it must not panic on non-ASCII input.
        // A non-match (the common "tag doesn't match any pattern" case) used
        // to panic on the byte-index loop.
        assert!(
            std::panic::catch_unwind(|| regex_match("café", "stable")).is_ok(),
            "non-ASCII tag must not panic"
        );
        assert!(!regex_match("café", "stable"));
        // Multibyte matching still works: "café" ends with the 2-byte é.
        assert!(regex_match("café", "caf.*"));
        assert!(regex_match("café", "caf."));
        assert!(!regex_match("café", "^caf$"));
        // CJK + emoji (4-byte chars).
        assert!(regex_match("こんにちは", "こん.*"));
        assert!(!regex_match("😀😁", "stable"));
    }

    #[test]
    fn regex_match_router_pick_with_non_ascii_tag_does_not_panic() {
        let strategy = TagBased::new();
        let d1 = make_dep_with_regex("gpt-4o", "openai/gpt-4o-a", vec!["prod-.*-v2"]);
        let deps = [&d1];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.tag = Some("café-ària-π".to_string());
        // Must not panic and must fall back to shuffle (no pattern matches).
        assert!(strategy.pick(&deps, &ctx).is_some());
    }

    #[test]
    fn regex_match_no_catastrophic_backtracking() {
        // `a*a*a*a*X` against a long run of 'a' with no trailing X is the
        // textbook catastrophic-backtracking input for a greedy recursive
        // matcher. The Thompson NFA is linear, so this completes in a few
        // milliseconds (generous bound for debug builds).
        let pattern = "a*a*a*a*X";
        let text = "a".repeat(4_000);
        let start = std::time::Instant::now();
        let matched = regex_match(&text, pattern);
        let elapsed = start.elapsed();
        assert!(!matched, "no trailing X → no match");
        assert!(
            elapsed.as_millis() < 500,
            "NFA must be linear, took {elapsed:?}"
        );
        // Sanity: with the trailing X it does match.
        let mut text2 = text;
        text2.push('X');
        assert!(regex_match(&text2, pattern));
    }

    // ---- multi-tag requests (body metadata.tags + header values) --------

    fn ctx_with_tags(tags: &[&str]) -> RequestContext {
        let mut ctx = RequestContext::new("gpt-4o");
        let owned: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
        ctx.tags = owned.clone();
        ctx.tag = owned.first().cloned();
        ctx
    }

    /// Positive tags subset-match: a request carrying ["app-a", "env-prod"]
    /// only routes to deployments advertising BOTH (every positive request
    /// tag must be covered by the deployment's tags).
    #[test]
    fn multi_tag_positive_subset_matching() {
        let both = make_dep_with_tags("both", "m", vec!["app-a".into(), "env-prod".into()]);
        let only_a = make_dep_with_tags("only-a", "m", vec!["app-a".into()]);
        let only_b = make_dep_with_tags("only-b", "m", vec!["env-prod".into()]);
        let deps = [&both, &only_a, &only_b];
        let picked = TagBased::new()
            .pick(&deps, &ctx_with_tags(&["app-a", "env-prod"]))
            .expect("a match exists");
        assert_eq!(picked.model_name, "both");
    }

    /// A single positive tag keeps picking any deployment covering it.
    #[test]
    fn multi_tag_single_positive_matches_its_carrier() {
        let a = make_dep_with_tags("a", "m", vec!["app-a".into()]);
        let b = make_dep_with_tags("b", "m", vec!["app-b".into()]);
        let deps = [&a, &b];
        for _ in 0..8 {
            let picked = TagBased::new()
                .pick(&deps, &ctx_with_tags(&["app-b"]))
                .expect("match");
            assert_eq!(picked.model_name, "b");
        }
    }

    /// `&tag` requires every surviving deployment to carry the tag;
    /// `!tag` excludes deployments matching it — both compose with
    /// positive tags.
    #[test]
    fn multi_tag_require_and_exclude_compose() {
        let cheap = make_dep_with_tags("cheap", "m", vec!["tier-free".into()]);
        let premium = make_dep_with_tags("premium", "m", vec!["tier-paid".into()]);
        let deps = [&cheap, &premium];
        // require tier-free, exclude nothing.
        let picked = TagBased::new()
            .pick(&deps, &ctx_with_tags(&["&tier-free"]))
            .expect("match");
        assert_eq!(picked.model_name, "cheap");
        // exclude tier-paid.
        let picked = TagBased::new()
            .pick(&deps, &ctx_with_tags(&["!tier-paid"]))
            .expect("match");
        assert_eq!(picked.model_name, "cheap");
        // require tier-free AND exclude tier-free: no survivors.
        let ctx = ctx_with_tags(&["&tier-free", "!tier-free"]);
        assert!(TagBased::new().pick(&deps, &ctx).is_none());
        assert!(TagBased::new().try_pick(&deps, &ctx).is_err());
    }

    #[test]
    fn multi_tag_default_fallback_respects_required_and_excluded_tags() {
        let allowed = make_dep_with_tags("allowed", "m", vec!["region-eu".into()]);
        let forbidden =
            make_dep_with_tags("forbidden", "m", vec!["default".into(), "region-us".into()]);
        let deps = [&allowed, &forbidden];
        // A default outside the allowed pool must not override either kind
        // of hard selector when the positive preference has no match.
        for tags in [
            vec!["no-such-tag", "&region-eu"],
            vec!["no-such-tag", "!region-us"],
        ] {
            let picked = TagBased::new()
                .try_pick(&deps, &ctx_with_tags(&tags))
                .expect("constraints are satisfiable")
                .expect("allowed fallback");
            assert_eq!(picked.model_name, "allowed");
        }
    }

    #[test]
    fn multi_tag_unsatisfied_constraints_never_use_default_or_whole_group() {
        let default = make_dep_with_tags("default", "m", vec!["default".into()]);
        let deps = [&default];
        for tags in [
            vec!["&missing"],
            vec!["!default"],
            vec!["no-such-tag", "!default"],
        ] {
            let ctx = ctx_with_tags(&tags);
            assert!(TagBased::new().try_pick(&deps, &ctx).is_err());
            assert!(TagBased::new().pick(&deps, &ctx).is_none());
        }
    }

    #[test]
    fn multi_tag_regex_constraints_apply_to_fallbacks() {
        let allowed = make_dep_with_regex("allowed", "m", vec!["^region-eu-"]);
        let mut forbidden = make_dep_with_regex("forbidden", "m", vec!["^region-us-"]);
        forbidden
            .model_info
            .as_mut()
            .unwrap()
            .tags
            .push("default".into());
        let deps = [&allowed, &forbidden];
        let ctx = ctx_with_tags(&["unknown-preference", "&region-eu-west", "!region-us-east"]);
        let picked = TagBased::new()
            .try_pick(&deps, &ctx)
            .expect("constraints match regexes")
            .expect("allowed fallback");
        assert_eq!(picked.model_name, "allowed");
    }

    /// When nothing matches, deployments tagged `default` form the fallback
    /// pool; without one, the strategy fails open to the whole group.
    #[test]
    fn multi_tag_default_deployment_fallback_pool() {
        let a = make_dep_with_tags("a", "m", vec!["app-a".into()]);
        let dflt = make_dep_with_tags("dflt", "m", vec!["default".into()]);
        let deps = [&a, &dflt];
        for _ in 0..8 {
            let picked = TagBased::new()
                .pick(&deps, &ctx_with_tags(&["no-such-tag"]))
                .expect("fallback");
            assert_eq!(picked.model_name, "dflt");
        }
    }

    /// Regex `tag_regex` patterns satisfy a positive tag just like exact
    /// tags do: a deployment mixing an exact tag and a regex pattern
    /// covers a request carrying one of each, while a deployment covering
    /// only one of the two positives is not a subset match.
    #[test]
    fn multi_tag_regex_satisfies_positives_like_exact_tags() {
        let mut mixed = make_dep_with_tags("mixed", "m", vec!["app-x".into()]);
        mixed
            .model_info
            .as_mut()
            .unwrap()
            .extra
            .insert("tag_regex".to_string(), json!(["^env-.*"]));
        let partial = make_dep_with_tags("partial", "m", vec!["app-x".into()]);
        let deps = [&mixed, &partial];
        for _ in 0..8 {
            let picked = TagBased::new()
                .pick(&deps, &ctx_with_tags(&["app-x", "env-prod-eu"]))
                .expect("match");
            assert_eq!(picked.model_name, "mixed");
        }
    }
}
