//! Redis-optional distributed coordination.
//!
//! When Redis is configured (`cache.type: redis`), these primitives
//! share state across replicas. Without Redis, they degrade to
//! per-process behavior — the proxy still works, just without
//! fleet-wide coordination.

use std::sync::Arc;

use oxllm_core::Cache;

/// Wrapper that detects whether distributed coordination is available.
///
/// Constructed once at startup. The `distributed` flag is set when the
/// cache backend is a Redis-backed [`DualCache`](crate::DualCache) with a
/// working connection; consumers check [`Self::is_distributed`] to decide
/// whether to use the shared-store code path or the local-only fallback.
///
/// **Precondition of [`Self::with_distributed`]**: the wrapped cache MUST be
/// a live Redis-backed store. Setting `distributed = true` over a per-process
/// cache (e.g. [`InMemoryCache`](crate::cache::InMemoryCache)) compiles and
/// runs but provides ZERO cross-replica coordination — each replica's
/// "distributed" store is its own memory, so an attacker multiplies the
/// failure budget by the replica count. The type is `dyn Cache` (no type
/// tag), so this contract cannot be enforced at construction; the production
/// bootstrap ([`app::run`](crate::app::run)) honors it by constructing a
/// distributed coordinator ONLY from a real `redis_cache`.
#[derive(Clone)]
pub struct Coordinator {
    cache: Arc<dyn Cache>,
    distributed: bool,
}

impl Coordinator {
    /// Wrap a cache. Starts in local-only mode; call [`Self::with_distributed`]
    /// to enable the shared-store code path.
    pub fn new(cache: Arc<dyn Cache>) -> Self {
        Self {
            cache,
            distributed: false,
        }
    }

    /// Enable or disable distributed coordination. Set to `true` ONLY when the
    /// backing store is a live Redis connection (see the type-level
    /// precondition). The flag is not derivable from `dyn Cache`, so the caller
    /// is responsible for upholding this invariant.
    pub fn with_distributed(mut self, yes: bool) -> Self {
        if yes {
            tracing::debug!(
                distributed = true,
                "coordinator marked distributed; caller MUST ensure the wrapped \
                 cache is a live Redis-backed store (per-process caches give no \
                 cross-replica coordination)"
            );
        }
        self.distributed = yes;
        self
    }

    /// True when Redis-backed coordination is active.
    pub fn is_distributed(&self) -> bool {
        self.distributed
    }

    /// Access the underlying cache.
    pub fn cache(&self) -> &Arc<dyn Cache> {
        &self.cache
    }
}

impl std::fmt::Debug for Coordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Coordinator")
            .field("distributed", &self.distributed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_local_only() {
        let cache: Arc<dyn Cache> = Arc::new(crate::cache::InMemoryCache::default());
        let coord = Coordinator::new(cache);
        assert!(!coord.is_distributed());
    }

    #[test]
    fn with_distributed_enables_sharing() {
        let cache: Arc<dyn Cache> = Arc::new(crate::cache::InMemoryCache::default());
        let coord = Coordinator::new(cache).with_distributed(true);
        assert!(coord.is_distributed());
    }

    /// `with_distributed(false)` explicitly resets the flag back to local-only
    /// (e.g. after a builder step that tentatively enabled it).
    #[test]
    fn with_distributed_false_resets_to_local_only() {
        let cache: Arc<dyn Cache> = Arc::new(crate::cache::InMemoryCache::default());
        let coord = Coordinator::new(cache).with_distributed(true);
        assert!(coord.is_distributed());
        let coord = coord.with_distributed(false);
        assert!(
            !coord.is_distributed(),
            "with_distributed(false) must reset the flag to local-only"
        );
    }

    #[test]
    fn cache_accessor_returns_underlying_arc() {
        let cache: Arc<dyn Cache> = Arc::new(crate::cache::InMemoryCache::default());
        let coord = Coordinator::new(cache);
        let _ = coord.cache();
    }

    /// Latent-misuse contract (LOW): `with_distributed(true)` over a
    /// per-process cache cannot be auto-detected (`dyn Cache` carries no type
    /// tag), so the caller MUST uphold the Redis-backed precondition. The
    /// production bootstrap honors it by constructing a distributed
    /// coordinator ONLY from a real `redis_cache`. This test pins the
    /// documented behavior: the flag is settable regardless of the backing
    /// store, and the responsibility is the caller's — there is no silent
    /// refusal. A future typed constructor should make this unrepresentable;
    /// until then the guard is the doc + this regression test.
    #[test]
    fn with_distributed_is_caller_responsibility_over_non_redis_cache() {
        let cache: Arc<dyn Cache> = Arc::new(crate::cache::InMemoryCache::default());
        let coord = Coordinator::new(cache).with_distributed(true);
        assert!(
            coord.is_distributed(),
            "the flag is settable over any dyn Cache; ensuring the backing store is \
             Redis is the caller's documented precondition"
        );
    }
}
