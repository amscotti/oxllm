//! Budget reset scheduler.
//!
//! A small background task that runs every 60 seconds, examines all spend /
//! budget counters in the cache, and zeros any whose `budget_duration` window
//! has elapsed since the counter was last reset.

use std::sync::Arc;
use std::time::Duration;

use oxllm_core::Cache as CacheTrait;

use crate::cache::InMemoryCache;
use crate::hooks::budget::BudgetHook;
use crate::hooks::scope_for_spend_key;
use crate::state::SharedState;

/// How often the scheduler wakes up to scan counters.
const TICK_INTERVAL: Duration = Duration::from_secs(60);

/// A budget-reset scheduler. It scans the cache for counters prefixed with
/// `spend:` (set by [`BudgetHook`]) and zeros them when their configured
/// `budget_duration` has elapsed.
///
/// When `redis_cache` is `Some` (a `DualCache` is configured), spend counters
/// are also deleted from Redis so fleet-wide spend is zeroed — not just the
/// local in-memory mirror.
///
/// The scheduler holds the [`SharedState`] rather than a fixed
/// `Arc<AlertState>`: a config hot-reload may swap in a NEW `AlertState`
/// (e.g. when the alerting config changes), and the scheduler must clear
/// alerts on the state that is live *now* — not the startup snapshot.
pub struct BudgetResetScheduler {
    cache: Arc<InMemoryCache>,
    redis_cache: Option<Arc<dyn CacheTrait>>,
    shared: SharedState,
}

impl BudgetResetScheduler {
    /// Create a new scheduler wired to the given cache and shared application
    /// state. The current `AlertState` AND `budget_duration` are re-read from
    /// `shared` on every tick, so hot-reloads that replace either are honored.
    ///
    /// `_budget_hook` is accepted for API stability and retained by the caller
    /// (it is still the live per-call budget hook used by the hook chain); the
    /// scheduler itself does not read it — the global window is re-derived from
    /// the live shared config (or the canonical 30-day default) on every tick
    /// so a reload that removes `budget_duration` matches a freshly-started
    /// peer rather than sticking at the startup snapshot.
    pub fn new(
        cache: Arc<InMemoryCache>,
        _budget_hook: Arc<BudgetHook>,
        shared: SharedState,
    ) -> Self {
        Self {
            cache,
            redis_cache: None,
            shared,
        }
    }

    /// Attach the full cache layer (typically a `DualCache`) so that spend
    /// counters are also reset in Redis when configured. When `None` (or when
    /// the layer is a pure `InMemoryCache`), only the local memory counter is
    /// reset.
    pub fn with_redis_cache(mut self, cache: Arc<dyn CacheTrait>) -> Self {
        self.redis_cache = Some(cache);
        self
    }

    /// Run the scheduler loop until the given shutdown signal fires.
    ///
    /// The first tick runs immediately on startup so stale TTL-less counters
    /// (the scheduler's reason for existing) are cleaned in the first seconds
    /// of boot rather than after a full `TICK_INTERVAL`.
    pub async fn run(self, mut shutdown: tokio::sync::oneshot::Receiver<()>) {
        self.tick_once().await;
        loop {
            tokio::select! {
            _ = &mut shutdown => {
            tracing::info!("budget reset scheduler shutting down");
            return;
            }
            _ = tokio::time::sleep(TICK_INTERVAL) => {
            self.tick_once().await;
            }
            }
        }
    }

    /// Run a single pass over the spend counters. Exposed for unit tests so
    /// they don't have to wait 60 seconds.
    ///
    /// The scheduler is **defense-in-depth** for the *global* key-budget
    /// window only. Per-key expiry is owned by the cache: `BudgetHook::pre_call`
    /// attaches a per-counter TTL when first incrementing a `spend:*` counter,
    /// and the cache reports expired counters as absent from `counter_keys`.
    /// This pass zeroes only counters whose *global* `budget_duration` has
    /// elapsed AND whose own per-key TTL does not extend past the next tick
    /// (so a longer per-key window is never truncated — Bug #30).
    ///
    /// Router-scoped counters (`spend:dep:*`, `spend:provider:*`) carry TTLs
    /// unrelated to the key budget (set by `orchestration/budget.rs`) and are
    /// skipped entirely; their cache TTL is authoritative.
    ///
    /// The eligibility check and the zero happen atomically under one write
    /// lock (see [`InMemoryCache::reset_spend_counter_if_due`]), so a
    /// concurrent `incr_if_below` cannot land in the gap between the check
    /// and the reset.
    ///
    /// Known limitation: when a Redis layer is configured, `delete` on the
    /// `DualCache` is infallible (`Cache::delete` returns `()`), so a
    /// Redis-side delete failure is invisible and the local alert scope is
    /// cleared regardless. The observable failure mode is an alert re-firing
    /// after a reset that did not propagate fleet-wide.
    pub async fn tick_once(&self) {
        let keys = self.cache.counter_keys().await;
        // Take a SINGLE atomic snapshot of the live shared state. Two reads
        // (`budget_duration` and `alert_state`) could otherwise straddle a
        // hot-reload swap, pairing a new duration with a stale `AlertState`
        // Arc (or vice versa). Both fields live on the same `SharedState`, so
        // one `load_full()` is cheaper AND self-consistent. The snapshot
        // re-reads `budget_duration` every tick so a reload that changes it
        // (or removes it, falling back to the canonical 30-day default a
        // freshly-started peer would compute — NOT the startup snapshot) is
        // honored rather than frozen at construction time.
        let snap = self.shared.load_full();
        let budget_duration = snap
            .config
            .oxllm_settings
            .budget_duration
            .as_deref()
            .map(parse_budget_duration)
            .unwrap_or_else(default_duration);
        // Re-read the CURRENT alert state from the snapshot. A hot-reload that
        // changes the alerting config swaps in a brand-new `AlertState` Arc;
        // holding the startup snapshot here would clear alerts on a dead state
        // while the live hook writes to the new one.
        let alert_state = snap.alert_state.clone();
        for key in keys {
            if !key.starts_with("spend:") {
                continue;
            }
            // Router-scoped counters carry their own TTLs (deployment /
            // provider budgets); the global key-budget duration does not
            // apply to them. Let their cache TTL be authoritative.
            if key.starts_with("spend:dep:") || key.starts_with("spend:provider:") {
                continue;
            }
            let reset = self
                .cache
                .reset_spend_counter_if_due(&key, budget_duration, TICK_INTERVAL)
                .await;
            if !reset {
                continue;
            }
            tracing::info!(
            counter = %key,
            budget_duration_secs = budget_duration.as_secs(),
            redis = self.redis_cache.is_some(),
            "resetting budget counter (global window elapsed)"
            );
            if let Some(redis_cache) = &self.redis_cache {
                redis_cache.delete(&key).await;
            }
            if let Some(alert_state) = &alert_state {
                if let Some(scope) = scope_for_spend_key(&key) {
                    alert_state.clear_scope(&scope).await;
                }
            }
        }
    }
}

/// Parse a budget duration string into a [`Duration`].
///
/// Supported suffixes (matches the reference `duration_in_seconds` grammar,
/// `core_utils/duration_parser.py`):
/// - `s` — seconds (e.g. `"30s"`)
/// - `m` — minutes (e.g. `"5m"`)
/// - `h` — hours (e.g. `"1h"`)
/// - `d` — days (e.g. `"30d"`)
/// - `w` — weeks (e.g. `"1w"`)
/// - `mo` — months, approximated as 30 days (e.g. `"1mo"`)
///
/// Word aliases (the reference `_BUDGET_DURATION_WORD_ALIASES`, applied before
/// suffix parsing so a config migrated verbatim behaves
/// identically):
/// - `hourly` → `1h`, `daily` → `24h`, `weekly` → `7d`, `monthly` → `30d`.
///
/// Oxllm extension (outside the reference grammar, for native configs):
/// - `ms` — milliseconds (e.g. `"500ms"`); sub-second budget windows are
///   operationally meaningless (the scheduler tick is 60s), but the suffix is
///   accepted so tests can drive sub-second windows without long sleeps.
///
/// Defaults to 30 days when the input is empty. Emits a `warn!` and falls
/// back to the 30-day default when the input is non-empty but unparseable
/// (e.g. `"1.5h"`, `"xxd"`) so a silent typo cannot turn into a billing
/// surprise.
pub fn parse_budget_duration(s: &str) -> Duration {
    let trimmed = s.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return default_duration();
    }
    match parse_duration_suffix(&trimmed) {
        Some(d) => d,
        None => {
            tracing::warn!(input = %s, "unparseable budget_duration; defaulting to 30d");
            default_duration()
        }
    }
}

/// Check whether `s` is a valid `budget_duration` string (e.g. `"30d"`,
/// `"1mo"`, `"5m"`, or a word alias like `"daily"`). Used by the
/// management API to reject malformed values at the boundary rather than
/// silently defaulting to 30d at enforcement time. Empty strings are valid
/// (= use the default window).
pub fn budget_duration_is_valid(s: &str) -> bool {
    let trimmed = s.trim().to_ascii_lowercase();
    trimmed.is_empty() || parse_duration_suffix(&trimmed).is_some()
}

fn parse_duration_suffix(trimmed: &str) -> Option<Duration> {
    // Normalize word aliases BEFORE suffix parsing
    // (`_BUDGET_DURATION_WORD_ALIASES`: hourly/daily/weekly/monthly). Without
    // this, `"daily"` would reach the suffix parser, its last char (`'y'`)
    // matches no unit, and `parse_budget_duration` would silently fall back to
    // the 30-day default — a 30x divergence from the reference for a config migrated
    // verbatim. `monthly` coincidentally matches (30d), but the other three do
    // not, so they must be mapped explicitly.
    let trimmed = match trimmed {
        "hourly" => "1h",
        "daily" => "24h",
        "weekly" => "7d",
        "monthly" => "30d",
        other => other,
    };
    let (num_str, suffix): (&str, &str) = if let Some(stripped) = trimmed.strip_suffix("mo") {
        (stripped, "mo")
    } else if let Some(stripped) = trimmed.strip_suffix("ms") {
        (stripped, "ms")
    } else {
        let c = trimmed.chars().last()?;
        let s = match c {
            's' => "s",
            'm' => "m",
            'h' => "h",
            'd' => "d",
            'w' => "w",
            _ => return None,
        };
        (&trimmed[..trimmed.len() - 1], s)
    };

    let n: u64 = num_str.parse().ok()?;
    let secs = match suffix {
        "s" => n,
        "ms" => return Some(Duration::from_millis(n)),
        "m" => n.saturating_mul(60),
        "h" => n.saturating_mul(3600),
        "d" => n.saturating_mul(24 * 3600),
        "w" => n.saturating_mul(7 * 24 * 3600),
        "mo" => n.saturating_mul(30 * 24 * 3600),
        _ => return None,
    };
    Some(Duration::from_secs(secs))
}

fn default_duration() -> Duration {
    Duration::from_secs(30 * 24 * 3600)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::AlertState;

    /// Build a `SharedState` wrapping a minimal `AppState` whose only
    /// meaningful field is the given alert state.
    fn shared_state(alert_state: Option<Arc<AlertState>>) -> SharedState {
        shared_state_with_budget(alert_state, None)
    }

    /// Like [`shared_state`] but also sets `oxllm_settings.budget_duration`,
    /// so the scheduler's per-tick re-read of the global window can be
    /// exercised (including a hot-reload swap to a new value).
    fn shared_state_with_budget(
        alert_state: Option<Arc<AlertState>>,
        budget_duration: Option<&str>,
    ) -> SharedState {
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings};
        let oxllm_settings = OxllmSettings {
            budget_duration: budget_duration.map(|s| s.to_string()),
            ..Default::default()
        };
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings,
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        let mut state = crate::state::AppState::new(config).expect("state");
        state.alert_state = alert_state;
        Arc::new(arc_swap::ArcSwap::from_pointee(state))
    }

    #[test]
    fn parses_seconds() {
        assert_eq!(parse_budget_duration("30s"), Duration::from_secs(30));
    }

    #[test]
    fn parses_minutes() {
        assert_eq!(parse_budget_duration("5m"), Duration::from_secs(300));
    }

    #[test]
    fn parses_hours() {
        assert_eq!(parse_budget_duration("1h"), Duration::from_secs(3600));
    }

    #[test]
    fn parses_days() {
        assert_eq!(
            parse_budget_duration("30d"),
            Duration::from_secs(30 * 24 * 3600)
        );
    }

    #[test]
    fn parses_months_as_30_days() {
        assert_eq!(
            parse_budget_duration("1mo"),
            Duration::from_secs(30 * 24 * 3600)
        );
        assert_eq!(
            parse_budget_duration("2mo"),
            Duration::from_secs(2 * 30 * 24 * 3600)
        );
    }

    #[test]
    fn parses_weeks() {
        assert_eq!(
            parse_budget_duration("1w"),
            Duration::from_secs(7 * 24 * 3600)
        );
        assert_eq!(
            parse_budget_duration("2w"),
            Duration::from_secs(2 * 7 * 24 * 3600)
        );
    }

    /// Word aliases (`_BUDGET_DURATION_WORD_ALIASES`) are normalized
    /// before suffix parsing so a config migrated verbatim is
    /// interpreted identically. Without this, `daily`/`hourly`/`weekly`
    /// silently fall back to the 30-day default (30x / 720x / 4x divergences).
    #[test]
    fn parses_litellm_word_aliases() {
        assert_eq!(
            parse_budget_duration("hourly"),
            Duration::from_secs(3600),
            "hourly -> 1h"
        );
        assert_eq!(
            parse_budget_duration("daily"),
            Duration::from_secs(24 * 3600),
            "daily -> 24h"
        );
        assert_eq!(
            parse_budget_duration("weekly"),
            Duration::from_secs(7 * 24 * 3600),
            "weekly -> 7d"
        );
        assert_eq!(
            parse_budget_duration("monthly"),
            Duration::from_secs(30 * 24 * 3600),
            "monthly -> 30d"
        );
    }

    /// Word aliases are case-insensitive and trimmed like suffix forms.
    #[test]
    fn word_aliases_are_case_insensitive_and_trimmed() {
        assert_eq!(
            parse_budget_duration("DAILY"),
            Duration::from_secs(24 * 3600)
        );
        assert_eq!(
            parse_budget_duration("  Hourly  "),
            Duration::from_secs(3600)
        );
    }

    /// The management-API boundary check must accept word aliases so
    /// a ported config is not rejected while enforcement accepts
    /// it (or silently defaults).
    #[test]
    fn budget_duration_is_valid_accepts_word_aliases() {
        for ok in ["hourly", "daily", "weekly", "monthly"] {
            assert!(
                budget_duration_is_valid(ok),
                "'{ok}' should be a valid budget_duration"
            );
        }
    }

    /// `ms` is an oxllm-only extension (the reference has no millisecond branch);
    /// kept so tests can drive sub-second windows without long sleeps.
    #[test]
    fn parses_milliseconds() {
        assert_eq!(parse_budget_duration("500ms"), Duration::from_millis(500));
    }

    #[test]
    fn uppercase_input_is_normalized() {
        assert_eq!(parse_budget_duration("1H"), Duration::from_secs(3600));
        assert_eq!(
            parse_budget_duration("30D"),
            Duration::from_secs(30 * 24 * 3600)
        );
    }

    #[test]
    fn empty_returns_default_30_days() {
        assert_eq!(
            parse_budget_duration(""),
            Duration::from_secs(30 * 24 * 3600)
        );
    }

    #[test]
    fn budget_duration_is_valid_accepts_known_suffixes() {
        for ok in [
            "", "30s", "5m", "1h", "30d", "1mo", "2mo", "1w", "500ms", "  1H ",
        ] {
            assert!(budget_duration_is_valid(ok), "'{ok}' should be valid");
        }
    }

    #[test]
    fn budget_duration_is_valid_rejects_garbage() {
        for bad in ["1x", "garbage", "1.5h", "xxd", "abc"] {
            assert!(!budget_duration_is_valid(bad), "'{bad}' should be invalid");
        }
    }

    #[test]
    fn unparseable_returns_default() {
        assert_eq!(
            parse_budget_duration("garbage"),
            Duration::from_secs(30 * 24 * 3600)
        );
        assert_eq!(
            parse_budget_duration("xxd"),
            Duration::from_secs(30 * 24 * 3600)
        );
    }

    /// Malformed-but-non-empty inputs fall back to the 30-day default (and
    /// emit a `warn!`). These shapes previously fell through silently — a
    /// typo like `"1d "` vs `"1mo"` could turn into a 30x billing error.
    #[test]
    fn unparseable_non_empty_inputs_return_default() {
        let default = Duration::from_secs(30 * 24 * 3600);
        // Decimal values are not supported (integer + suffix only).
        assert_eq!(parse_budget_duration("1.5h"), default);
        // Numeric-only (no suffix).
        assert_eq!(parse_budget_duration("30"), default);
        // Space between number and unit.
        assert_eq!(parse_budget_duration("1 hour"), default);
        // Non-numeric prefix.
        assert_eq!(parse_budget_duration("xxd"), default);
        // Regression: a previously-accepted-but-wrong `"1w"` must NOT equal
        // the 30-day default now that the `w` (week) suffix is honored.
        assert_ne!(parse_budget_duration("1w"), default);
    }

    #[tokio::test]
    async fn scheduler_resets_spend_counter_after_window_expiry() {
        use crate::cache::InMemoryCache;

        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 42.0, None).await;
        assert_eq!(cache.counter_value("spend:abc").await, Some(42.0));

        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_millis(1),
        ));

        let scheduler = BudgetResetScheduler::new(
            cache.clone(),
            budget_hook,
            shared_state_with_budget(None, Some("1ms")),
        );

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;

        // The reset path removes the TTL-less counter entry entirely (matching
        // the Redis `delete` path). The counter is semantically zero for all
        // downstream purposes (every caller reads
        // `counter_value(...).unwrap_or(0.0)`), but the explicit `None` avoids
        // leaking a never-expiring zero row the sweep could never reclaim.
        assert_eq!(
            cache.counter_value("spend:abc").await,
            None,
            "spend counter must be removed after window expiry"
        );
    }

    #[tokio::test]
    async fn scheduler_resets_spend_counter_in_redis_layer() {
        use crate::cache::{DualCache, InMemoryCache};

        let memory = Arc::new(InMemoryCache::default());
        let dual: Arc<dyn CacheTrait> = Arc::new(DualCache::new(memory.clone(), None));

        let _ = memory.incr("spend:abc", 7.0, None).await;
        assert_eq!(dual.counter_value("spend:abc").await, Some(7.0));

        let budget_hook = Arc::new(BudgetHook::with_duration(
            dual.clone(),
            None,
            Duration::from_millis(1),
        ));

        let scheduler = BudgetResetScheduler::new(
            memory.clone(),
            budget_hook,
            shared_state_with_budget(None, Some("1ms")),
        )
        .with_redis_cache(dual.clone());

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;

        // The reset path runs `memory.reset_counter` then `dual.delete`. After
        // M9, `delete` also clears the in-memory counter map, so the counter
        // is gone entirely (`None`) rather than pinned at `Some(0.0)`. Both
        // states are semantically "0 spend" for budget checks (every caller
        // uses `counter_value(...).unwrap_or(0.0)`), but the explicit `None`
        // is the post-M9 contract — and it avoids leaking a never-expiring
        // zero row in the counters map.
        assert_eq!(
            memory.counter_value("spend:abc").await,
            None,
            "memory spend counter must be cleared after redis-layer reset"
        );
    }

    #[tokio::test]
    async fn scheduler_does_not_reset_non_spend_counters() {
        use crate::cache::InMemoryCache;

        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:abc", 10.0, None).await;

        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_millis(1),
        ));
        let scheduler = BudgetResetScheduler::new(
            cache.clone(),
            budget_hook,
            shared_state_with_budget(None, Some("1ms")),
        );

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;

        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(10.0),
            "non-spend counters must not be reset by the budget scheduler"
        );
    }

    /// Bug #30: per-key `budget_duration` is honored via the per-counter TTL
    /// that `BudgetHook::pre_call` attaches when first incrementing the spend
    /// counter. The scheduler is defense-in-depth for the GLOBAL window only;
    /// per-key expiry is owned by the cache. This test documents that contract:
    /// a counter created with a SHORT TTL expires on its own (without any
    /// scheduler tick), and one with a LONG TTL survives a scheduler tick that
    /// uses a shorter global duration.
    #[tokio::test]
    async fn per_key_ttl_expires_independently_of_scheduler_global_duration() {
        use crate::cache::InMemoryCache;

        let cache = Arc::new(InMemoryCache::default());

        // Simulate `BudgetHook::pre_call` for a per-key counter whose
        // configured duration (200ms) is much shorter than the global one.
        // The TTL is kept comfortably above scheduler/lock overhead so the
        // "still alive" assertion below is not a race against natural expiry.
        let per_key_ttl = Duration::from_millis(200);
        let _ = cache.incr("spend:perkey", 5.0, Some(per_key_ttl)).await;

        // Another counter with NO TTL — relies on the scheduler + the global
        // duration for reset.
        let _ = cache.incr("spend:nottl", 5.0, None).await;

        // Global budget_duration is long; the scheduler should NOT reset
        // either counter on this tick.
        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_secs(3600),
        ));
        let scheduler = BudgetResetScheduler::new(cache.clone(), budget_hook, shared_state(None));
        scheduler.tick_once().await;
        assert_eq!(
            cache.counter_value("spend:perkey").await,
            Some(5.0),
            "scheduler must not reset a counter whose global window has not elapsed"
        );
        assert_eq!(cache.counter_value("spend:nottl").await, Some(5.0));

        // Wait for the per-key TTL to expire. The cache itself must report
        // the counter as gone — no scheduler tick required.
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert_eq!(
            cache.counter_value("spend:perkey").await,
            None,
            "per-key TTL is the authoritative expiry path: the counter must \
             vanish once its own TTL elapses, independent of the scheduler"
        );
        // The no-TTL counter is still there (it has no per-key expiry).
        assert_eq!(cache.counter_value("spend:nottl").await, Some(5.0));
    }

    /// CRITICAL regression: a per-key counter whose own TTL is LONGER than
    /// the global `budget_duration` must NOT be reset when the global window
    /// elapses. Previously the scheduler compared only against the global
    /// duration, silently zeroing 30+ days of legitimate spend mid-window and
    /// granting free headroom for the rest of the real window.
    #[tokio::test]
    async fn per_key_longer_ttl_survives_global_window_expiry() {
        use crate::cache::InMemoryCache;

        let cache = Arc::new(InMemoryCache::default());
        // Per-key TTL of 1h; global budget_duration is 1ms. After the global
        // window elapses, the scheduler must still leave this counter alone.
        let _ = cache
            .incr("spend:longlived", 7.0, Some(Duration::from_secs(3600)))
            .await;

        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_millis(1),
        ));
        let scheduler = BudgetResetScheduler::new(
            cache.clone(),
            budget_hook,
            shared_state_with_budget(None, Some("1ms")),
        );

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;

        assert_eq!(
            cache.counter_value("spend:longlived").await,
            Some(7.0),
            "a counter whose per-key TTL outlives the global window must \
             not be prematurely zeroed"
        );
    }

    /// Router-scoped spend counters (`spend:dep:*`, `spend:provider:*`) carry
    /// their own TTLs (deployment / provider budgets) that are unrelated to
    /// the global key-budget duration. The scheduler must skip them so their
    /// cache TTL remains authoritative — resetting them on the global duration
    /// would either truncate a longer window or leave a stale one alive.
    #[tokio::test]
    async fn scheduler_skips_deployment_and_provider_spend_keys() {
        use crate::cache::InMemoryCache;

        let cache = Arc::new(InMemoryCache::default());
        // No per-key TTL — without the scope skip these would be reset since
        // the global budget_duration (1ms) elapses. The skip is what protects
        // them.
        let _ = cache
            .incr("spend:dep:gpt-4o:openai/gpt-4o", 10.0, None)
            .await;
        let _ = cache.incr("spend:provider:openai", 10.0, None).await;
        // A plain key-scoped counter with no TTL is still reset by the
        // scheduler (sanity that the skip is scope-specific, not blanket).
        let _ = cache.incr("spend:plainkey", 10.0, None).await;

        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_millis(1),
        ));
        let scheduler = BudgetResetScheduler::new(
            cache.clone(),
            budget_hook,
            shared_state_with_budget(None, Some("1ms")),
        );

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;

        assert_eq!(
            cache.counter_value("spend:dep:gpt-4o:openai/gpt-4o").await,
            Some(10.0),
            "deployment spend counter must not be reset by the key-budget scheduler"
        );
        assert_eq!(
            cache.counter_value("spend:provider:openai").await,
            Some(10.0),
            "provider spend counter must not be reset by the key-budget scheduler"
        );
        assert_eq!(
            cache.counter_value("spend:plainkey").await,
            None,
            "plain key-scoped counter is still reset on the global window"
        );
    }

    /// The first scheduler tick runs immediately on startup (before the first
    /// `TICK_INTERVAL` sleep) so stale TTL-less counters are cleaned in the
    /// first seconds of boot. Without the immediate tick, a fresh start would
    /// leave expired counters alive for up to 60s.
    #[tokio::test]
    async fn run_resets_stale_counter_on_first_tick_without_waiting_for_interval() {
        use crate::cache::InMemoryCache;

        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:stale", 5.0, None).await;
        // Ensure the global window (1ms) has elapsed before the scheduler
        // starts, so the first tick is eligible to reset.
        tokio::time::sleep(Duration::from_millis(2)).await;

        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_millis(1),
        ));
        let scheduler = BudgetResetScheduler::new(
            cache.clone(),
            budget_hook,
            shared_state_with_budget(None, Some("1ms")),
        );

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            scheduler.run(rx).await;
        });

        // Far shorter than TICK_INTERVAL (60s); only the immediate startup
        // tick can reset the counter this quickly.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            cache.counter_value("spend:stale").await,
            None,
            "first tick must run immediately on startup, resetting the stale \
             counter without waiting a full TICK_INTERVAL"
        );

        let _ = tx.send(());
        let _ = handle.await;
    }

    #[tokio::test]
    async fn scheduler_clears_alert_state_when_resetting_counter() {
        use crate::cache::InMemoryCache;
        use crate::hooks::AlertState;

        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 42.0, None).await;

        let alert_state = Arc::new(AlertState::new());
        alert_state.mark_fired("key:abc", 0.8).await;
        alert_state.mark_fired("key:abc", 1.0).await;
        alert_state.mark_fired("team:t1", 0.8).await;
        assert!(alert_state.was_fired("key:abc", 0.8).await);

        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_millis(1),
        ));
        let scheduler = BudgetResetScheduler::new(
            cache.clone(),
            budget_hook,
            shared_state_with_budget(Some(alert_state.clone()), Some("1ms")),
        );

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;

        assert_eq!(
            cache.counter_value("spend:abc").await,
            None,
            "spend counter must be reset"
        );
        assert!(
            !alert_state.was_fired("key:abc", 0.8).await,
            "resetting spend:abc must clear the key:abc alert scope"
        );
        assert!(!alert_state.was_fired("key:abc", 1.0).await);
        assert!(
            alert_state.was_fired("team:t1", 0.8).await,
            "unrelated scopes must be left untouched"
        );
    }

    /// The model budget counter `spend:model:{name}` is reset by the
    /// scheduler (same `spend:` prefix) and its alert scope (`model:{name}`)
    /// is cleared via `scope_for_spend_key`.
    #[tokio::test]
    async fn scheduler_resets_model_spend_counter_and_clears_alert_scope() {
        use crate::cache::InMemoryCache;
        use crate::hooks::AlertState;

        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:model:gpt-4o", 42.0, None).await;

        let alert_state = Arc::new(AlertState::new());
        alert_state.mark_fired("model:gpt-4o", 0.8).await;
        assert!(alert_state.was_fired("model:gpt-4o", 0.8).await);

        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_millis(1),
        ));
        let scheduler = BudgetResetScheduler::new(
            cache.clone(),
            budget_hook,
            shared_state_with_budget(Some(alert_state.clone()), Some("1ms")),
        );

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;

        assert_eq!(
            cache.counter_value("spend:model:gpt-4o").await,
            None,
            "per-model spend counter must be reset alongside other spend counters"
        );
        assert!(
            !alert_state.was_fired("model:gpt-4o", 0.8).await,
            "resetting spend:model:gpt-4o must clear the model:gpt-4o alert scope"
        );
    }

    /// Hot-reload divergence regression: when a reload swaps in a state with
    /// a NEW `AlertState` Arc, the scheduler must clear alerts on the state
    /// that is live NOW — not the startup snapshot it was constructed with.
    #[tokio::test]
    async fn scheduler_clears_current_alert_state_after_hot_reload_swap() {
        use crate::cache::InMemoryCache;

        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 42.0, None).await;

        let old_alert = Arc::new(AlertState::new());
        old_alert.mark_fired("key:abc", 0.8).await;
        let new_alert = Arc::new(AlertState::new());
        new_alert.mark_fired("key:abc", 0.8).await;

        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_millis(1),
        ));
        let shared = shared_state_with_budget(Some(old_alert.clone()), Some("1ms"));
        let scheduler = BudgetResetScheduler::new(cache.clone(), budget_hook, shared.clone());

        // Simulate a hot-reload: the watcher stores a state whose alerting
        // config changed, producing a brand-new `AlertState` Arc.
        let mut reloaded = shared.load_full().as_ref().clone();
        reloaded.alert_state = Some(new_alert.clone());
        shared.store(Arc::new(reloaded));

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;

        assert!(
            old_alert.was_fired("key:abc", 0.8).await,
            "the OLD alert state must be untouched — it is no longer live"
        );
        assert!(
            !new_alert.was_fired("key:abc", 0.8).await,
            "the scheduler must clear the CURRENT alert state after a hot-reload swap"
        );
    }

    /// LOW regression: the global `budget_duration` window must be re-read from
    /// the live shared config on every tick, so a hot-reload that changes
    /// `oxllm_settings.budget_duration` is honored by the defense-in-depth
    /// sweep (rather than staying frozen at the startup snapshot).
    #[tokio::test]
    async fn scheduler_uses_hot_reloaded_budget_duration() {
        use crate::cache::InMemoryCache;
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings};

        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 5.0, None).await;

        // Construct the scheduler with a LONG window (1h) and a shared state
        // whose config also says 1h. The first tick must NOT reset (window not
        // elapsed) even after the global window would have elapsed under a
        // short duration.
        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_secs(3600),
        ));
        let shared = shared_state_with_budget(None, Some("1h"));
        let scheduler = BudgetResetScheduler::new(cache.clone(), budget_hook, shared.clone());

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;
        assert_eq!(
            cache.counter_value("spend:abc").await,
            Some(5.0),
            "with a 1h window the counter must survive the tick"
        );

        // Hot-reload: swap in a state whose budget_duration is 1ms. The
        // scheduler was NOT reconstructed — it must pick up the new window from
        // the live shared config on the next tick.
        let mut reloaded = shared.load_full().as_ref().clone();
        reloaded.config = Arc::new(ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings {
                budget_duration: Some("1ms".to_string()),
                ..Default::default()
            },
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        });
        shared.store(Arc::new(reloaded));

        tokio::time::sleep(Duration::from_millis(5)).await;
        scheduler.tick_once().await;
        assert_eq!(
            cache.counter_value("spend:abc").await,
            None,
            "after a hot-reload to a 1ms window, the scheduler must use the \
             new duration and reset the elapsed counter"
        );
    }

    /// LOW regression: when a hot-reload REMOVES `budget_duration` from the
    /// live config, the scheduler's defense-in-depth sweep must fall back to
    /// the canonical 30-day default a fresh process would compute — NOT the
    /// stale startup snapshot. Otherwise the global reset cadence diverges
    /// from freshly-started peers for the rest of the process lifetime.
    #[tokio::test]
    async fn scheduler_uses_canonical_default_after_budget_duration_removed() {
        use crate::cache::InMemoryCache;
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings};

        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 5.0, None).await;

        // Construct with a SHORT (1h) budget_duration in both the hook and the
        // shared config. The first tick must NOT reset (1h window not elapsed).
        let budget_hook = Arc::new(BudgetHook::with_duration(
            cache.clone() as Arc<dyn CacheTrait>,
            None,
            Duration::from_secs(3600),
        ));
        let shared = shared_state_with_budget(None, Some("1h"));
        let scheduler = BudgetResetScheduler::new(cache.clone(), budget_hook, shared.clone());

        scheduler.tick_once().await;
        assert_eq!(
            cache.counter_value("spend:abc").await,
            Some(5.0),
            "with a 1h window the counter must survive the tick"
        );

        // Hot-reload: swap in a config whose `budget_duration` is `None`.
        // The scheduler must NOT keep using the stale 1h startup snapshot; it
        // must use the canonical 30-day default, under which the counter (only
        // moments old) still survives the tick.
        let mut reloaded = shared.load_full().as_ref().clone();
        reloaded.config = Arc::new(ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings {
                budget_duration: None,
                ..Default::default()
            },
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        });
        shared.store(Arc::new(reloaded));

        scheduler.tick_once().await;
        assert_eq!(
            cache.counter_value("spend:abc").await,
            Some(5.0),
            "after a hot-reload removing budget_duration, the scheduler must \
             use the 30-day canonical default (not the stale 1h startup \
             snapshot), so the counter survives the tick"
        );
    }
}
