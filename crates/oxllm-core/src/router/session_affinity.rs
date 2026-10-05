//! Session-affinity routing strategy.
//!
//! When the request carries a `session_id` (typically from the
//! `x-oxllm-session-id` header), this strategy deterministically maps that id
//! onto the candidate list so the same session sticks to the same deployment.
//! That keeps provider-side prompt caches warm across multi-turn conversations
//! when multiple deployments share a model group.
//!
//! NOTE: this is deterministic modulo-based sticky routing, NOT consistent
//! hashing. The pick is `fnv1a64(session_id) % len` over a stably-sorted
//! candidate list, so adding or removing a deployment remaps a fraction of
//! sessions to a different host (a consistent-hash ring would minimise that
//! churn). Acceptable for prompt-cache warmth; revisit if session migration
//! cost becomes significant.
//!
//! The `session_id` is attacker-controlled (plumbed from a request header),
//! so it is capped to `MAX_SESSION_ID_LEN` bytes before hashing: `fnv1a_64`
//! is O(n) in the id length, so a megabyte-scale header could pin a worker
//! core per pick. Hashing the first 8 KiB is more than enough entropy for
//! sticky routing (the cap matches [`tag_based`]'s analogous bound on its
//! regex path).
//!
//! Without a session id (or with an empty one), falls back to [`SimpleShuffle`].
//!
//! [`tag_based`]: crate::router::tag_based

use crate::deployment::Deployment;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::stores::fnv1a_64;
use crate::router::strategy::{RequestContext, RoutingStrategy};

/// Sticky routing by session id: deterministic modulo pick among candidates
/// (see module docs for why this is not a consistent-hash ring).
pub struct SessionAffinity {
    shuffle: SimpleShuffle,
}

impl SessionAffinity {
    /// Create a new session-affinity strategy.
    pub fn new() -> Self {
        Self {
            shuffle: SimpleShuffle,
        }
    }
}

impl Default for SessionAffinity {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingStrategy for SessionAffinity {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        preferred_from_full_group(deployments, ctx).or_else(|| self.shuffle.pick(deployments, ctx))
    }
}

/// Sticky pick over a full (pre-cooldown) candidate group.
///
/// Used by the router so a session stays on its preferred deployment when that
/// host is still available, even if other siblings cooled and the available
/// set would otherwise remap the hash.
///
/// Ordering is independent of the process-scoped runtime keys. The private
/// tie-breaker distinguishes otherwise identical siblings without exposing an
/// unkeyed credential digest in metrics or management responses.
pub(crate) fn preferred_from_full_group<'a>(
    deployments: &[&'a Deployment],
    ctx: &RequestContext,
) -> Option<&'a Deployment> {
    if deployments.is_empty() {
        return None;
    }
    let session_id = ctx.session_id.as_deref().filter(|s| !s.is_empty())?;

    let mut ordered: Vec<(&Deployment, String)> = deployments
        .iter()
        .map(|d| {
            let params = serde_json::to_value(&d.oxllm_params)
                .map(|value| value.to_string())
                .unwrap_or_default();
            (*d, params)
        })
        .collect();
    ordered.sort_by(|a, b| {
        (&a.0.model_name, &a.0.oxllm_params.model, &a.1).cmp(&(
            &b.0.model_name,
            &b.0.oxllm_params.model,
            &b.1,
        ))
    });

    // Cap the attacker-controlled id before hashing (see MAX_SESSION_ID_LEN
    // doc). The hash is over the truncated prefix; ids that differ only
    // past the cap intentionally collapse to the same deployment.
    let capped = cap_session_id_len(session_id);

    // Reduce the modulo in `u64` before casting to `usize`. On 64-bit
    // targets `usize == u64` so this is a no-op; on 32-bit targets
    // `usize == u32`, and `(hash as usize) % len` would truncate the
    // 64-bit hash first, producing a different index from a 64-bit
    // replica and splitting the session across deployments — defeating
    // the prompt-cache-warmth invariant promised in the module docs.
    let len = ordered.len() as u64;
    let idx = (fnv1a_64(capped.as_bytes()) % len) as usize;
    Some(ordered[idx].0)
}

/// Upper bound on the `session_id` length fed to the hash. `fnv1a_64` is
/// O(n) in the id length, so a pathologically long `x-oxllm-session-id`
/// header (attacker-controlled, plumbed through [`RequestContext::session_id`])
/// would burn a core per pick scanning megabytes. Legitimate session ids are
/// short opaque tokens; 8 KiB is far beyond any real value and matches the
/// [`tag_based`] cap on its regex path for the same DoS reason.
///
/// [`tag_based`]: crate::router::tag_based
/// [`RequestContext::session_id`]: crate::router::strategy::RequestContext::session_id
const MAX_SESSION_ID_LEN: usize = 8 * 1024;

/// Truncate `session_id` to at most [`MAX_SESSION_ID_LEN`] bytes on a UTF-8
/// char boundary, matching the truncation style used by
/// [`tag_based::cap_tag_len`](crate::router::tag_based) and the complexity
/// classifier.
fn cap_session_id_len(session_id: &str) -> &str {
    if session_id.len() <= MAX_SESSION_ID_LEN {
        return session_id;
    }
    let mut end = MAX_SESSION_ID_LEN;
    while end > 0 && !session_id.is_char_boundary(end) {
        end -= 1;
    }
    &session_id[..end]
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
    fn same_session_always_picks_same_deployment() {
        let strategy = SessionAffinity::new();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let d2 = make_dep("gpt-4o", "xai/grok-4");
        let d3 = make_dep("gpt-4o", "openrouter/openai/gpt-4o");
        let deps = [&d1, &d2, &d3];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.session_id = Some("user-thread-42".to_string());

        let first = strategy.pick(&deps, &ctx).expect("pick");
        for _ in 0..20 {
            let again = strategy.pick(&deps, &ctx).expect("pick");
            assert_eq!(
                first.oxllm_params.model, again.oxllm_params.model,
                "session affinity must be sticky"
            );
        }
    }

    #[test]
    fn pick_is_stable_across_input_reorder() {
        let strategy = SessionAffinity::new();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let d2 = make_dep("gpt-4o", "xai/grok-4");
        let d3 = make_dep("gpt-4o", "openrouter/openai/gpt-4o");
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.session_id = Some("stable-order-session".to_string());

        let a = strategy
            .pick(&[&d1, &d2, &d3], &ctx)
            .expect("pick")
            .oxllm_params
            .model
            .clone();
        let b = strategy
            .pick(&[&d3, &d1, &d2], &ctx)
            .expect("pick")
            .oxllm_params
            .model
            .clone();
        let c = strategy
            .pick(&[&d2, &d3, &d1], &ctx)
            .expect("pick")
            .oxllm_params
            .model
            .clone();
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn different_sessions_can_spread() {
        let strategy = SessionAffinity::new();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let d2 = make_dep("gpt-4o", "xai/grok-4");
        let d3 = make_dep("gpt-4o", "openrouter/openai/gpt-4o");
        let deps = [&d1, &d2, &d3];

        let mut seen = std::collections::HashSet::new();
        for i in 0..64 {
            let mut ctx = RequestContext::new("gpt-4o");
            ctx.session_id = Some(format!("session-{i}"));
            let picked = strategy.pick(&deps, &ctx).expect("pick");
            seen.insert(picked.oxllm_params.model.clone());
        }
        assert!(
            seen.len() > 1,
            "different session ids should spread across deployments, got {seen:?}"
        );
    }

    #[test]
    fn falls_back_to_shuffle_when_no_session_id() {
        let strategy = SessionAffinity::new();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let d2 = make_dep("gpt-4o", "xai/grok-4");
        let deps = [&d1, &d2];
        let ctx = RequestContext::new("gpt-4o");
        assert!(strategy.pick(&deps, &ctx).is_some());
    }

    #[test]
    fn empty_session_id_falls_back_to_shuffle() {
        let strategy = SessionAffinity::new();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let deps = [&d1];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.session_id = Some(String::new());
        assert!(strategy.pick(&deps, &ctx).is_some());
    }

    #[test]
    fn returns_none_when_no_deployments() {
        let strategy = SessionAffinity::new();
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.session_id = Some("s".to_string());
        assert!(strategy.pick(&[], &ctx).is_none());
    }

    #[test]
    fn single_deployment_always_selected() {
        let strategy = SessionAffinity::new();
        let d1 = make_dep("gpt-4o", "openai/gpt-4o");
        let deps = [&d1];
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.session_id = Some("any".to_string());
        let picked = strategy.pick(&deps, &ctx).expect("pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o");
    }

    #[test]
    fn pick_index_uses_u64_modulo_not_truncated() {
        // The pick must reduce the modulo in `u64` before casting to `usize`,
        // so a 32-bit replica picks the same deployment as a 64-bit replica
        // for a given session id. `session-1` over a 3-deployment group is a
        // discriminator: `fnv1a_64("session-1") % 3 == 1`, but truncating the
        // hash to 32 bits first gives `0x_cf49184d % 3 == 0`. A 32-bit target
        // running the old `(hash as usize) % len` formula would pick index 0;
        // the fix picks index 1 on every target.
        let strategy = SessionAffinity::new();
        let d0 = make_dep("g", "openai/a");
        let d1 = make_dep("g", "xai/b");
        let d2 = make_dep("g", "openrouter/c");

        let h = fnv1a_64(b"session-1");
        assert_eq!(
            h, 0x1688_595d_cf49_184d,
            "discriminator hash must match the value the test was built around"
        );
        let u64_idx = (h % 3) as usize;
        let trunc_idx = (h as u32 as usize) % 3;
        assert_ne!(
            u64_idx, trunc_idx,
            "this test only exercises the bug if the two formulas disagree"
        );
        assert_eq!(u64_idx, 1);

        // The public model order must agree across replicas regardless of
        // their independently generated runtime-key peppers.
        let expected_model = "openrouter/c";

        let mut ctx = RequestContext::new("g");
        ctx.session_id = Some("session-1".to_string());
        let picked = strategy
            .pick(&[&d0, &d1, &d2], &ctx)
            .expect("pick")
            .oxllm_params
            .model
            .clone();
        assert_eq!(
            picked, expected_model,
            "must pick the u64-modulo index even on 32-bit targets"
        );
    }

    #[test]
    fn oversized_session_id_does_not_pin_core_and_collapses_past_cap() {
        // An attacker-controlled `session_id` plumbed from a request header
        // must not pin a core: the id is capped to MAX_SESSION_ID_LEN before
        // hashing, so a megabyte-scale id completes in well under a
        // millisecond. Two ids that share their first MAX_SESSION_ID_LEN
        // bytes (and differ only past the cap) must collapse to the same
        // deployment, pinning the truncation contract.
        let strategy = SessionAffinity::new();
        // Use enough deployments that distinct hashes reliably spread
        // (the test asserts both collapse and distinctness, so a 2-deployment
        // coin-flip is too fragile).
        let deps: Vec<Deployment> = (0..16)
            .map(|i| make_dep("g", &format!("openai/m{i}")))
            .collect();
        let refs: Vec<&Deployment> = deps.iter().collect();

        // 1 MiB id.
        let mut big = String::with_capacity(1024 * 1024);
        big.push_str("session-");
        big.push_str(&"x".repeat(1024 * 1024 - "session-".len()));
        let mut ctx_big = RequestContext::new("g");
        ctx_big.session_id = Some(big);
        let start = std::time::Instant::now();
        let picked_big = strategy
            .pick(&refs, &ctx_big)
            .expect("pick")
            .oxllm_params
            .model
            .clone();
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 100,
            "oversized id must complete quickly (<100ms even on a slow CI runner), \
             took {elapsed:?}"
        );

        // Same prefix, different tail — must pick the same deployment.
        let mut sibling = String::with_capacity(1024 * 1024 + 4);
        sibling.push_str(ctx_big.session_id.as_deref().unwrap());
        sibling.push_str("YYYY");
        let mut ctx_sibling = RequestContext::new("g");
        ctx_sibling.session_id = Some(sibling);
        let picked_sibling = strategy
            .pick(&refs, &ctx_sibling)
            .expect("pick")
            .oxllm_params
            .model
            .clone();
        assert_eq!(
            picked_big, picked_sibling,
            "ids differing only past MAX_SESSION_ID_LEN must collapse to the same deployment"
        );

        // An id whose prefix differs within the cap must hash to a different
        // index. Over 16 deployments and a differing prefix, the chance of
        // collision is ~1/16; if they do collide, retry with a different
        // prefix (deterministic, not flaky).
        let mut found_distinct = false;
        for ch in ['Z', 'Q', 'R', 'T', 'W', 'P', 'K', 'L'] {
            let mut other = String::from("session-");
            other.push(ch);
            other.push_str(&"x".repeat(1024 * 1024 - "session-".len() - 1));
            let mut ctx_other = RequestContext::new("g");
            ctx_other.session_id = Some(other);
            let picked_other = strategy
                .pick(&refs, &ctx_other)
                .expect("pick")
                .oxllm_params
                .model
                .clone();
            if picked_other != picked_big {
                found_distinct = true;
                break;
            }
        }
        assert!(
            found_distinct,
            "ids differing within the cap must hash to different indices across \
             at least one of the tried prefixes (sanity check for the cap boundary)"
        );
    }
}
