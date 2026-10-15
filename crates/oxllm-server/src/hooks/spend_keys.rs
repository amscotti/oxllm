//! Shared spend-counter key builders.
//!
//! [`BudgetHook`](crate::hooks::BudgetHook) (reservation/settlement) and
//! [`AlertingHook`](crate::hooks::AlertingHook) (threshold alerts) must agree
//! byte-for-byte on the cache keys that hold per-scope spend counters, and
//! the [`BudgetResetScheduler`](crate::scheduler::BudgetResetScheduler) scans
//! for the same `spend:` prefix. These functions are the single source of
//! truth for that key format — do not inline `format!("spend:...")` in hook
//! code.
//!
//! # Identifier invariant
//!
//! The unscoped key builder accepts only an API-key hash (hex, no `:`), so
//! it cannot collide with named scope prefixes. Typed builders add their
//! namespace before the opaque identifier. A tag such as `product:demo`
//! therefore remains distinct from every other scope: reverse lookup strips
//! the fixed namespace once, retaining the complete identifier. Do not split
//! scoped identifiers on every separator.

/// Spend counter key for an API key hash: `spend:{api_key_hash}`.
///
/// `api_key_hash` must not contain `:` — see the module-level identifier
/// invariant.
pub fn spend_key(api_key_hash: &str) -> String {
    format!("spend:{api_key_hash}")
}

/// Spend counter key for a team: `spend:team:{team_id}`.
///
/// `team_id` must not contain `:` — see the module-level identifier invariant.
pub fn team_spend_key(team_id: &str) -> String {
    format!("spend:team:{team_id}")
}

/// Spend counter key for a user: `spend:user:{user_id}`.
///
/// `user_id` must not contain `:` — see the module-level identifier invariant.
pub fn user_spend_key(user_id: &str) -> String {
    format!("spend:user:{user_id}")
}

/// Spend counter key for an end-user / customer: `spend:customer:{end_user_id}`.
///
/// `end_user_id` must not contain `:` — see the module-level identifier
/// invariant. Management create/update rejects ids containing `:`.
pub fn customer_spend_key(end_user_id: &str) -> String {
    format!("spend:customer:{end_user_id}")
}

/// Spend counter key for a request tag: `spend:tag:{tag_name}`.
///
/// Tag names are literal, including `:` in `key:value` attribution pairs.
pub fn tag_spend_key(tag_name: &str) -> String {
    format!("spend:tag:{tag_name}")
}

/// Spend counter key for a model (global cap): `spend:model:{model}`.
///
/// `model` must not contain `:` — see the module-level identifier invariant.
pub fn model_spend_key(model: &str) -> String {
    format!("spend:model:{model}")
}

/// Spend counter key for an organization: `spend:org:{org_id}`.
///
/// `org_id` must not contain `:` — see the module-level identifier invariant.
pub fn org_spend_key(org_id: &str) -> String {
    format!("spend:org:{org_id}")
}

/// Spend counter key for a project: `spend:project:{project_id}`.
///
/// `project_id` must not contain `:` — see the module-level identifier
/// invariant.
pub fn project_spend_key(project_id: &str) -> String {
    format!("spend:project:{project_id}")
}

/// Spend counter key for a deployment (per-deployment budget):
/// `spend:dep:{deployment_key}`. The `deployment_key` is the router's
/// deployment key (`{model_name}:{oxllm_params.model}`, see
/// [`oxllm_core::router::deployment_key`]) — the same identifier the
/// `dep_rpm:` / `dep_tpm:` counters use. It intentionally contains a `:`
/// separating the model name from the deployment-params model, which is why the
/// scheduler scopes spend-key parsing to the `team`/`user`/`org`/`project`/
/// `model` prefixes and leaves `dep:`/`provider:` keys to the router.
pub fn deployment_spend_key(deployment_key: &str) -> String {
    format!("spend:dep:{deployment_key}")
}

/// Spend counter key for a provider (per-provider budget):
/// `spend:provider:{provider_slug}`.
pub fn provider_spend_key(provider_slug: &str) -> String {
    format!("spend:provider:{provider_slug}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_formats_are_stable() {
        // The exact byte format is a cross-component contract (budget hook,
        // alerting hook, reset scheduler, spend-analytics queries).
        assert_eq!(spend_key("abc"), "spend:abc");
        assert_eq!(team_spend_key("t1"), "spend:team:t1");
        assert_eq!(user_spend_key("u1"), "spend:user:u1");
        assert_eq!(customer_spend_key("eu-1"), "spend:customer:eu-1");
        assert_eq!(tag_spend_key("canary"), "spend:tag:canary");
        assert_eq!(model_spend_key("gpt-4o"), "spend:model:gpt-4o");
        assert_eq!(org_spend_key("o1"), "spend:org:o1");
        assert_eq!(project_spend_key("p1"), "spend:project:p1");
        assert_eq!(
            deployment_spend_key("gpt-4o:openai/gpt-4o"),
            "spend:dep:gpt-4o:openai/gpt-4o"
        );
        assert_eq!(provider_spend_key("openai"), "spend:provider:openai");
    }

    /// The unscoped builder requires a key hash; typed builders retain their
    /// fixed namespace even when the opaque identifier contains separators.
    #[test]
    fn spend_key_does_not_escape_colon_in_id() {
        // A malicious/buggy id carrying a scope prefix is passed through
        // unchanged — confirming there is no escaping today. This is the
        // hazard the invariant guards against: such an id would collide with
        // the team scope and be mis-cleared by the scheduler on reset.
        assert_eq!(spend_key("team:evil"), "spend:team:evil");
        assert_eq!(team_spend_key("x:user:y"), "spend:team:x:user:y");
        assert_eq!(tag_spend_key("product:demo"), "spend:tag:product:demo");
        assert_ne!(tag_spend_key("team:demo"), team_spend_key("demo"));
    }
}
