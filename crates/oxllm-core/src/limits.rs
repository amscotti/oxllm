//! Per-scope rate limits and budgets, grouped for reuse.
//!
//! `AuthUser`, `VirtualKeyInfo`, and `CallContext` all carry the same five
//! scopes of limits as flat `{scope}_{field}` columns (e.g. `max_budget`,
//! `team_rpm`, `org_tpm`). Those flat fields remain the storage format (and
//! the public access pattern), but the grouped types here give construction
//! and conversion code a single place to expand/collapse them, so adding a
//! new scoped limit no longer means hand-editing every constructor.

use serde::{Deserialize, Serialize};

/// Rate limits and budget for one scope (key, team, user, org, or project).
///
/// Not every scope populates every field today (the user scope has
/// `max_budget` + `budget_duration` but no rate-limit columns). Unpopulated
/// fields stay `None`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScopeLimits {
    /// Max budget (USD) for this scope's spend bucket.
    pub max_budget: Option<f64>,
    /// Budget window (e.g. `"30d"`).
    pub budget_duration: Option<String>,
    /// Requests-per-minute limit.
    pub rpm: Option<u32>,
    /// Tokens-per-minute limit.
    pub tpm: Option<u32>,
}

/// The limits for every scope a request can belong to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AllScopeLimits {
    /// Per-key limits (the key's own ceiling).
    pub key: ScopeLimits,
    /// Team limits, when the key belongs to a team with its own ceiling.
    pub team: ScopeLimits,
    /// User limits, when the key is attached to a user with a ceiling.
    pub user: ScopeLimits,
    /// Org limits, when the key belongs to an org with its own ceiling.
    pub org: ScopeLimits,
    /// Project limits, when the key belongs to a project with a ceiling.
    pub project: ScopeLimits,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_limits_default_is_all_none() {
        let limits = ScopeLimits::default();
        assert_eq!(limits.max_budget, None);
        assert_eq!(limits.budget_duration, None);
        assert_eq!(limits.rpm, None);
        assert_eq!(limits.tpm, None);
    }

    #[test]
    fn all_scope_limits_default_is_all_none() {
        let all = AllScopeLimits::default();
        assert_eq!(all, AllScopeLimits::default());
        assert_eq!(all.key, ScopeLimits::default());
        assert_eq!(all.team, ScopeLimits::default());
        assert_eq!(all.user, ScopeLimits::default());
        assert_eq!(all.org, ScopeLimits::default());
        assert_eq!(all.project, ScopeLimits::default());
    }
}
