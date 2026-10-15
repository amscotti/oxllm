//! Security middleware hardening.
//!
//! - `body_limit` config: `general_settings.max_request_body_size_mb`.
//! - [`smuggling_guard`]: rejects requests that carry both `Content-Length`
//!   and `Transfer-Encoding` (HTTP request smuggling protection).
//! - [`AuthBrake`]: per-IP auth-failure rate limiter (blocks after N fails).
//! - Secret redaction helpers ([`is_secret_key`], [`redact_header_value`],
//!   [`sanitize_url`]) keep credentials out of logs and URLs.
//! - [`sanitize_upstream_response`]: strips `server`/`x-request-id` headers
//!   and truncates upstream error bodies to 500 chars.

use std::collections::HashMap;
use std::net::{IpAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, Request, State};
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use futures_util::Stream;
use futures_util::StreamExt;
use parking_lot::Mutex;
use parking_lot::MutexGuard;

use crate::coordination::Coordinator;
use crate::error::ApiError;
use crate::state::SharedState;

/// Default request body size limit (10 MB) when config does not specify one.
pub const DEFAULT_BODY_LIMIT_BYTES: usize = 10 * 1024 * 1024;

/// Maximum upstream response body size (50 MB). Prevents a malicious or
/// misbehaving upstream from OOMing the proxy by streaming an unbounded body
/// into `response.bytes()`. Enforced in [`sanitize_upstream_response`] and the
/// non-streaming chat path via [`read_bounded_bytes`] (which aborts
/// mid-stream once the cap is exceeded).
pub const MAX_RESPONSE_BODY_BYTES: usize = 50 * 1024 * 1024;

/// Tighter cap than [`MAX_RESPONSE_BODY_BYTES`] for upstream *error* responses.
/// Error bodies are surfaced into our own error message and logged; 50 MB of
/// error text is a DoS vector and pointless. 10 KB is plenty to capture any
/// reasonable upstream error JSON. Shared by the chat and embeddings error
/// paths so the bounded-read contract has a single source of truth.
pub const MAX_ERROR_BODY_BYTES: usize = 10 * 1024;

/// Maximum nesting depth of a parsed JSON request body. A deeply nested JSON
/// document can cause quadratic CPU usage during parsing and traversal; the
/// body-size limit alone prevents the worst cases, and this check is
/// defense-in-depth against CPU-exhaustion DoS.
pub const MAX_JSON_DEPTH: usize = 128;

/// Resolve the configured body size limit (bytes) from `general_settings`.
/// A `None` config yields the 10 MB default; an explicit 0 is clamped up to a
/// 1 MB floor so a misconfiguration (`max_request_body_size_mb: 0`) does not
/// reject every POST outright (`RequestBodyLimitLayer::new(0)` denies all
/// bodies).
pub fn body_limit_bytes(max_mb: Option<u64>) -> usize {
    // `try_into` rather than `as usize` so a 32-bit target narrows a huge
    // config value to `usize::MAX` (saturating ceiling) instead of silently
    // wrapping to a much smaller value and under-limiting the body. Realistic
    // configs never approach this; the cast is defense-in-depth against the
    // silent truncation the review brief calls out.
    let mb: usize = max_mb.unwrap_or(10).try_into().unwrap_or(usize::MAX);
    mb.max(1).saturating_mul(1024 * 1024)
}

/// Request-smuggling guard: reject (400) when both `Content-Length` and
/// `Transfer-Encoding` are present.
pub async fn smuggling_guard(request: Request, next: Next) -> Response {
    let headers = request.headers();
    let has_cl = headers.contains_key("content-length");
    let has_te = headers.contains_key("transfer-encoding");
    if has_cl && has_te {
        return ApiError::bad_request(
            "ambiguous request: both 'Content-Length' and 'Transfer-Encoding' present",
        )
        .into_response();
    }
    next.run(request).await
}

/// One auth-failure record: count + the instant of the first failure in the
/// current window + the instant the IP was first blocked (when applicable).
#[derive(Debug, Clone, Copy)]
struct FailureRecord {
    count: u32,
    window_start: Instant,
    /// When the IP first crossed the failure threshold and entered the
    /// blocked state. `None` while the IP is still below the threshold. The
    /// IP remains blocked for `block_secs` from this instant, independent of
    /// the rolling failure window.
    blocked_at: Option<Instant>,
}

impl FailureRecord {
    fn reset(now: Instant) -> Self {
        Self {
            count: 0,
            window_start: now,
            blocked_at: None,
        }
    }
}

/// Number of `is_blocked` calls between sweeps of stale entries. Sweeping on
/// every call would make the hot path O(n) in the map size; doing it every
/// Nth call bounds the amortized cost while still evicting dead entries often
/// enough to prevent unbounded growth under IP-spoofing attacks (e.g. a
/// trusted proxy forwarding an attacker-controlled `X-Forwarded-For`).
const SWEEP_INTERVAL: u64 = 100;

/// Guards the [`AuthBrake`] map-cap warning so a botnet flood that keeps the
/// map at its cap does not spam one warn line per new IP.
static CAP_WARN_EMITTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Guards the distributed-brake backing-store outage warnings in
/// [`AuthBrake::is_blocked_distributed`] and
/// [`AuthBrake::record_failure_distributed`]. A sustained Redis outage would
/// otherwise emit one warn per blocked-check and one per recorded failure —
/// up to 3 warn lines per request under brute-force traffic. Warn once on
/// transition, then drop to `debug!` (the same one-shot contract as
/// [`CAP_WARN_EMITTED`]).
static DISTRIBUTED_CACHE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Emit the backing-store outage warning for the distributed brake at most
/// once per process (further outages are logged at debug). `fallback`
/// describes what degrades for this call site.
fn emit_distributed_outage_warn<E: std::fmt::Display>(error: &E, fallback: &str) {
    if !DISTRIBUTED_CACHE_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            error = %error,
            fallback,
            "distributed auth brake: backing store unavailable; degrading to the per-process map"
        );
    } else {
        tracing::debug!(
            error = %error,
            fallback,
            "distributed auth brake: backing store unavailable; degrading to the per-process map"
        );
    }
}

/// Emit the map-cap warning at most once per process (further evictions are
/// logged at debug). The one-shot keeps the hot eviction path free of log spam
/// while still surfacing the deployment condition to operators.
fn emit_cap_warn() {
    if !CAP_WARN_EMITTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            "auth brake tracked-IP map reached its cap; evicting oldest entries \
             to bound memory under a flood of distinct failing IPs"
        );
    }
}

/// Inner mutable state of [`AuthBrake`]. The `calls_since_sweep` counter
/// triggers periodic eviction so the `by_ip` map cannot grow without bound.
struct AuthBrakeInner {
    by_ip: HashMap<String, FailureRecord>,
    calls_since_sweep: u64,
}

/// A per-IP auth-failure brake. After `max_fails` failures within `window`,
/// the IP is blocked for `block_secs`. Uses a simple in-memory map with
/// periodic eviction of entries whose block AND failure window have both
/// elapsed (see `AuthBrake::sweep_locked`) plus a hard cap on the number of
/// tracked IPs (see [`AuthBrake::with_max_ips`]) so an in-flight botnet flood
/// cannot pin unbounded memory.
///
/// In a multi-replica deployment each replica maintains its own in-memory map;
/// when a distributed [`Coordinator`] is attached, failure counting and block
/// state are shared fleet-wide through the backing cache (Redis), so an
/// attacker cannot multiply the failure budget by the replica count. When the
/// shared store is unavailable the brake degrades to the per-process map for
/// the duration of the outage.
#[derive(Clone)]
pub struct AuthBrake {
    inner: Arc<Mutex<AuthBrakeInner>>,
    max_fails: u32,
    window: Duration,
    block_secs: u64,
    coordinator: Option<Coordinator>,
    max_ips: usize,
}

/// Default cap on the number of distinct IPs tracked in the local [`AuthBrake`]
/// map. A distributed botnet hammering bad credentials must not be able to pin
/// unbounded memory; beyond this many tracked IPs the oldest entries are
/// evicted to make room for new ones.
pub const DEFAULT_MAX_TRACKED_IPS: usize = 100_000;

impl AuthBrake {
    /// Build a brake with the given policy.
    pub fn new(max_fails: u32, window_secs: u64, block_secs: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AuthBrakeInner {
                by_ip: HashMap::new(),
                calls_since_sweep: 0,
            })),
            max_fails: max_fails.max(1),
            window: Duration::from_secs(window_secs.max(1)),
            // Clamp like the other fields: a 0-second block would make
            // `block_until == blocked_at`, silently disabling blocking.
            block_secs: block_secs.max(1),
            coordinator: None,
            max_ips: DEFAULT_MAX_TRACKED_IPS,
        }
    }

    /// Override the cap on the number of distinct tracked IPs (see
    /// [`DEFAULT_MAX_TRACKED_IPS`]). When the map is at the cap and a new IP is
    /// recorded, the entry with the oldest failure window is evicted so the map
    /// stays bounded under an in-flight botnet flood.
    pub fn with_max_ips(mut self, max_ips: usize) -> Self {
        self.max_ips = max_ips.max(1);
        self
    }

    /// Attach a [`Coordinator`]. When the coordinator is distributed
    /// (`is_distributed()` is `true`), failure counting and block state are
    /// shared across replicas via the backing cache (Redis); otherwise the
    /// local per-process map is used.
    pub fn with_coordinator(mut self, coord: Option<Coordinator>) -> Self {
        self.coordinator = coord;
        self
    }

    /// Default policy from the spec: 10 fails / 60s → block 300s.
    pub fn default_policy() -> Self {
        Self::new(10, 60, 300)
    }

    /// Returns the instant at which the IP's current block expires, or `None`
    /// when the IP is not currently blocked.
    fn block_until(record: &FailureRecord, block_secs: u64) -> Option<Instant> {
        record
            .blocked_at
            .map(|at| at + Duration::from_secs(block_secs))
    }

    /// Returns `true` when `record` is safe to evict: neither an active block
    /// nor a fresh failure window that could still trip the threshold. Keeping
    /// such entries would let an attacker pin unbounded memory by spamming
    /// distinct IPs (e.g. via a spoofed `X-Forwarded-For` behind a trusted
    /// proxy); evicting them reclaims that memory once they can no longer
    /// affect any in-flight decision.
    fn is_evictable(
        record: &FailureRecord,
        now: Instant,
        window: Duration,
        block_secs: u64,
    ) -> bool {
        if let Some(until) = Self::block_until(record, block_secs) {
            if now < until {
                return false;
            }
        }
        now.duration_since(record.window_start) >= window
    }

    /// Remove every entry whose block has expired AND whose failure window has
    /// elapsed. Called occasionally from [`Self::is_blocked`] to amortize the
    /// O(n) scan cost (every [`SWEEP_INTERVAL`]th call). The lock is already
    /// held by the caller.
    fn sweep_locked(&self, guard: &mut MutexGuard<'_, AuthBrakeInner>) {
        let now = Instant::now();
        let window = self.window;
        let block_secs = self.block_secs;
        let before = guard.by_ip.len();
        guard
            .by_ip
            .retain(|_, record| !Self::is_evictable(record, now, window, block_secs));
        let removed = before.saturating_sub(guard.by_ip.len());
        if removed > 0 {
            tracing::debug!(
                removed,
                remaining = guard.by_ip.len(),
                "auth brake sweep: evicted stale entries"
            );
        }
    }

    /// Remove the entry with the oldest failure window so a new IP can be
    /// recorded without growing the map past its hard cap. O(n) in the map
    /// size, but only reached when the map is already at the cap.
    ///
    /// Entries whose block is still active (`now < block_until`) are skipped
    /// when selecting the victim: dropping an actively-blocked IP would let
    /// it immediately re-offend, defeating the brake precisely when it
    /// matters most. The cap is still enforced — if every entry is actively
    /// blocked, the oldest by `window_start` is evicted anyway so the map
    /// never grows past `max_ips`. This mirrors the protection
    /// [`AuthBrake::is_evictable`] applies to the periodic sweep.
    fn evict_oldest_locked(
        by_ip: &mut HashMap<String, FailureRecord>,
        now: Instant,
        block_secs: u64,
    ) {
        let mut oldest_key: Option<String> = None;
        let mut oldest_start = Instant::now();
        let mut oldest_is_active_block = false;
        for (key, record) in by_ip.iter() {
            let is_active_block =
                Self::block_until(record, block_secs).is_some_and(|until| now < until);
            if oldest_key.is_none() {
                oldest_key = Some(key.clone());
                oldest_start = record.window_start;
                oldest_is_active_block = is_active_block;
                continue;
            }
            // Prefer evicting a non-blocked entry over a blocked one, even
            // when the blocked one is older: an active block is what the
            // brake exists to enforce.
            if is_active_block && !oldest_is_active_block {
                continue;
            }
            if !is_active_block && oldest_is_active_block {
                oldest_key = Some(key.clone());
                oldest_start = record.window_start;
                oldest_is_active_block = is_active_block;
                continue;
            }
            if record.window_start < oldest_start {
                oldest_key = Some(key.clone());
                oldest_start = record.window_start;
                oldest_is_active_block = is_active_block;
            }
        }
        if let Some(key) = oldest_key {
            by_ip.remove(&key);
        }
    }

    /// Returns `true` when the IP is currently in a blocked state. A block is
    /// active from the moment the failure threshold is crossed until
    /// `blocked_at + block_secs`, independent of the rolling failure window.
    ///
    /// When a distributed [`Coordinator`] is attached, the check is performed
    /// against the shared store (Redis) so every replica sees the same block
    /// state.
    pub async fn is_blocked(&self, ip: &str) -> bool {
        if let Some(coord) = &self.coordinator {
            if coord.is_distributed() {
                return self.is_blocked_distributed(ip).await;
            }
        }
        self.is_blocked_local(ip)
    }

    fn is_blocked_local(&self, ip: &str) -> bool {
        let mut guard = self.inner.lock();
        // Periodically reclaim memory from expired entries so an attacker
        // cannot grow the map without bound by hitting distinct IPs.
        guard.calls_since_sweep = guard.calls_since_sweep.saturating_add(1);
        if guard.calls_since_sweep >= SWEEP_INTERVAL {
            guard.calls_since_sweep = 0;
            self.sweep_locked(&mut guard);
        }
        let now = Instant::now();
        let window = self.window;
        let block_secs = self.block_secs;
        // Read-only fast path: do NOT create an entry merely to check status.
        // The previous `entry().or_insert(FailureRecord::reset(now))` grew the
        // map by one row for every distinct probing IP — a cheap unbounded
        // memory amplifier. Entries are created only in `record_failure`; an
        // IP we have never recorded a failure for is, by definition, not
        // blocked. We still mutate an existing entry in place to perform the
        // post-expiry reset (preserving prior behaviour for known IPs).
        let Some(entry) = guard.by_ip.get_mut(ip) else {
            return false;
        };
        if let Some(until) = Self::block_until(entry, block_secs) {
            if now < until {
                return true;
            }
            *entry = FailureRecord::reset(now);
            return false;
        }
        if now.duration_since(entry.window_start) >= window {
            *entry = FailureRecord::reset(now);
        }
        entry.count >= self.max_fails
    }

    /// Record an auth failure for an IP. When the failure count crosses the
    /// threshold, the IP is marked blocked for `block_secs`.
    ///
    /// When a distributed [`Coordinator`] is attached, the failure is recorded
    /// in the shared store (Redis) so the count is fleet-wide — an attacker
    /// cannot multiply the failure budget by the replica count.
    pub async fn record_failure(&self, ip: &str) {
        if let Some(coord) = &self.coordinator {
            if coord.is_distributed() {
                self.record_failure_distributed(ip).await;
                return;
            }
        }
        self.record_failure_local(ip);
    }

    fn record_failure_local(&self, ip: &str) {
        let mut guard = self.inner.lock();
        let now = Instant::now();
        let window = self.window;
        let block_secs = self.block_secs;
        if !guard.by_ip.contains_key(ip) && guard.by_ip.len() >= self.max_ips {
            // The map is at its hard cap and this is a new IP: evict the entry
            // with the oldest failure window so the map stays bounded under an
            // in-flight botnet flood (the periodic sweep cannot free space
            // while every distinct attacker IP has a fresh window). Entries
            // whose block is still active are preferentially preserved so the
            // evicted entry is, when possible, one that can no longer affect
            // an in-flight decision.
            Self::evict_oldest_locked(&mut guard.by_ip, now, block_secs);
            emit_cap_warn();
        }
        let entry = guard
            .by_ip
            .entry(ip.to_string())
            .or_insert(FailureRecord::reset(now));
        if let Some(until) = Self::block_until(entry, block_secs) {
            if now < until {
                return;
            }
            *entry = FailureRecord::reset(now);
        } else if now.duration_since(entry.window_start) >= window {
            *entry = FailureRecord::reset(now);
        }
        entry.count = entry.count.saturating_add(1);
        if entry.count >= self.max_fails && entry.blocked_at.is_none() {
            entry.blocked_at = Some(now);
        }
        // Log at debug, not warn: a brute-force attacker (millions of attempts
        // / spoofed XFF entries behind a trusted proxy) would otherwise flood
        // the log subsystem with one warn line per failure — a log-volume DoS.
        // The block transition is still surfaced at warn by the caller.
        tracing::debug!(
            ip = %ip,
            count = entry.count,
            max = self.max_fails,
            blocked = entry.blocked_at.is_some(),
            "auth failure recorded"
        );
    }

    async fn is_blocked_distributed(&self, ip: &str) -> bool {
        let Some(coord) = &self.coordinator else {
            return false;
        };
        let cache = coord.cache();
        let blocked_key = format!("authbrake:blocked:{ip}");
        // `try_counter_value` distinguishes a genuine miss (`Ok(None)`) from a
        // backing-store outage (`Err`). On an outage we fall back to the local
        // map rather than failing open: disabling blocking fleet-wide during a
        // Redis blip would hand an attacker an unlimited failure budget.
        match cache.try_counter_value(&blocked_key).await {
            Ok(Some(v)) => v >= 1.0,
            Ok(None) => false,
            Err(e) => {
                emit_distributed_outage_warn(&e, "local block state");
                self.is_blocked_local(ip)
            }
        }
    }

    /// Record a failure in the shared (Redis) store. The failure window is
    /// fixed, not sliding, matching the local path: the backing `incr`
    /// implementations (`InMemoryCache` and the Redis Lua `INCR_SCRIPT`) stamp
    /// the TTL only on first creation (`PTTL < 0`), so sustained probing
    /// extends the attacker's own block window but never the counting window —
    /// behaviourally identical to `record_failure_local`.
    ///
    /// The threshold crossing is atomic: the failure counter is incremented
    /// with [`Cache::incr_if_below`] capped at `max_fails`, so concurrent
    /// replicas cannot overshoot the counter, and the caller whose increment
    /// lands on (or past) the threshold sets the block. Setting the block also
    /// deletes the failure counter so a post-block failure starts from a clean
    /// window — otherwise a counting window longer than the block would
    /// instantly re-block on the first new failure (diverging from the local
    /// path, which resets the count on block expiry).
    ///
    /// A backing-store outage falls back to the local map (like the block
    /// check) so counting and blocking degrade to per-process behavior for the
    /// duration of the outage instead of silently disabling enforcement.
    async fn record_failure_distributed(&self, ip: &str) {
        let Some(coord) = &self.coordinator else {
            return;
        };
        let cache = coord.cache();
        if self.is_blocked_distributed(ip).await {
            return;
        }
        let fails_key = format!("authbrake:fails:{ip}");
        let blocked_key = format!("authbrake:blocked:{ip}");
        let count = match cache
            .incr_if_below(&fails_key, 1.0, self.max_fails as f64, Some(self.window))
            .await
        {
            Ok(oxllm_core::IncrResult::Allowed(v)) => v,
            Ok(oxllm_core::IncrResult::Rejected(current)) => current,
            Err(e) => {
                emit_distributed_outage_warn(&e, "local counting");
                self.record_failure_local(ip);
                return;
            }
        };
        if count >= self.max_fails as f64 {
            // Clean the failure window so a post-block failure starts fresh
            // (local parity); a longer counting window than the block
            // deliberately extends the block otherwise.
            let _ = cache.delete(&fails_key).await;
            let _ = cache
                .incr(
                    &blocked_key,
                    1.0,
                    Some(Duration::from_secs(self.block_secs)),
                )
                .await;
            tracing::warn!(
                ip = %ip,
                count = count,
                max = self.max_fails,
                "distributed auth brake: IP blocked"
            );
        }
    }

    /// Block duration in seconds (for the `Retry-After` header).
    pub fn block_secs(&self) -> u64 {
        self.block_secs
    }

    /// Remaining block time in seconds for `ip`, or `None` when `ip` is not
    /// currently blocked. Used to compute an accurate `Retry-After` header so
    /// a client blocked part-way through its window is not told to wait the
    /// full `block_secs` again. Falls back to [`Self::block_secs`] in
    /// distributed mode, where the shared store carries only a counter with
    /// a TTL, not the absolute `blocked_at` timestamp the local map records.
    pub async fn remaining_block_secs(&self, ip: &str) -> Option<u64> {
        if let Some(coord) = &self.coordinator {
            if coord.is_distributed() {
                // The distributed store carries a TTL'd counter, not an
                // absolute timestamp, so the precise remaining time is not
                // recoverable. Fall back to the full window.
                return self
                    .is_blocked_distributed(ip)
                    .await
                    .then_some(self.block_secs);
            }
        }
        let guard = self.inner.lock();
        let entry = guard.by_ip.get(ip)?;
        let until = Self::block_until(entry, self.block_secs)?;
        let now = Instant::now();
        (now < until).then(|| until.saturating_duration_since(now).as_secs().max(1))
    }

    /// Current number of tracked IPs. Exposed for tests so the eviction
    /// behavior can be asserted without sleeping for the full block window.
    #[cfg(test)]
    pub(crate) fn tracked_ips(&self) -> usize {
        let guard = self.inner.lock();
        guard.by_ip.len()
    }

    /// Force a sweep now. Exposed for tests so eviction can be exercised
    /// deterministically without depending on the call counter.
    #[cfg(test)]
    pub(crate) fn sweep_now(&self) {
        let mut guard = self.inner.lock();
        guard.calls_since_sweep = 0;
        self.sweep_locked(&mut guard);
    }
}

impl std::fmt::Debug for AuthBrake {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthBrake")
            .field("max_fails", &self.max_fails)
            .field("window", &self.window)
            .field("block_secs", &self.block_secs)
            .finish()
    }
}

/// Join every `X-Forwarded-For` header value (there may be more than one —
/// HTTP/1.1 permits duplicate headers) into a single comma-separated string,
/// matching RFC 7230 §3.2.2's equivalent-single-line rule. A misconfigured
/// proxy that passes duplicate `X-Forwarded-For` headers through without
/// merging them would otherwise cause the rightmost (trusted) value to live
/// in the second header, which a single `headers.get` read misses. Returns
/// `None` when no `X-Forwarded-For` header is present or every value fails
/// to render as a header string.
fn join_xff_headers(headers: &HeaderMap) -> Option<String> {
    let mut joined = String::new();
    for value in headers.get_all("x-forwarded-for") {
        let Ok(s) = value.to_str() else {
            continue;
        };
        if !joined.is_empty() {
            joined.push_str(", ");
        }
        joined.push_str(s);
    }
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// Parse a single `X-Forwarded-For` hop into an IP address. Tolerates the
/// RFC 7239-style bracketed IPv6 form (`[2001:db8::5]`), brackets plus port
/// (`[2001:db8::5]:443`), and the `ip:port` shape some proxies append for
/// IPv4 hops (`1.2.3.4:5678`); bare IPv4 / IPv6 entries parse directly.
/// Returns `None` for anything unparseable — the XFF walk then skips the
/// hop, never trusting it.
fn parse_xff_hop(raw: &str) -> Option<IpAddr> {
    let trimmed = raw.trim();
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return Some(ip);
    }
    let candidate: &str = if let Some(rest) = trimmed.strip_prefix('[') {
        let end = rest.find(']')?;
        &rest[..end]
    } else {
        // `ip:port`: strip the trailing `:port`. A bare IPv6 address would
        // already have parsed above, so more than one colon means the entry
        // is not a supported form.
        let colon = trimmed.rfind(':')?;
        if trimmed[..colon].contains(':') {
            return None;
        }
        &trimmed[..colon]
    };
    candidate.parse::<IpAddr>().ok()
}

/// Resolve the client IP for a request.
///
/// Forwarded headers (`X-Forwarded-For`, `X-Real-IP`) are honored only when
/// `trusted_proxies` is non-empty AND the connection's peer address falls
/// inside one of the trusted CIDR ranges. Otherwise the peer IP is used
/// directly (or `"unknown"` when no peer information is available — e.g. in
/// tests without `ConnectInfo`). This prevents an attacker from spoofing
/// forwarded headers to bypass per-IP auth rate limiting.
///
/// The peer is normalized via [`normalize_ip_for_ssrf`] so an IPv4-mapped
/// IPv6 presentation (`::ffff:a.b.c.d`, common on dual-stack listeners)
/// matches IPv4 trusted-proxy CIDRs, and every returned address is in
/// canonical (normalized) form.
pub fn client_ip(
    headers: &HeaderMap,
    peer: Option<std::net::SocketAddr>,
    trusted_proxies: &[ipnet::IpNet],
) -> String {
    let peer_ip = peer.map(|sa| normalize_ip_for_ssrf(sa.ip()));
    let peer_is_trusted = match peer_ip {
        Some(ip) => trusted_proxies.iter().any(|cidr| cidr.contains(&ip)),
        None => false,
    };
    if peer_is_trusted {
        if let Some(joined) = join_xff_headers(headers) {
            // Walk the XFF chain right-to-left, skipping entries that are
            // themselves trusted proxies, and use the first non-trusted,
            // IP-shaped entry: that is the real client. In a multi-tier trusted
            // chain (client → P1 → P2 → oxllm, with both P1 and P2 inside
            // `trusted_proxy_cidrs`), the rightmost entry is P1 (the inner
            // proxy), not the client; using it would collapse every client
            // behind P1 into one auth-brake bucket and, if P1's range happened
            // to be in `allowed_ips`, let every client behind P1 pass the IP
            // allowlist. The leftmost entry is trivially spoofable (a client
            // can prepend any value to XFF), so the walk stops at the first
            // non-trusted hop and never reaches client-controlled prepended
            // values. Non-IP-shaped hops are skipped rather than trusted; a
            // hop that never parses cannot be attributed.
            for hop in joined.split(',').rev() {
                let Some(ip) = parse_xff_hop(hop).map(normalize_ip_for_ssrf) else {
                    continue;
                };
                let trusted = trusted_proxies.iter().any(|cidr| cidr.contains(&ip));
                if !trusted {
                    return ip.to_string();
                }
            }
        }
        if let Some(xri) = headers.get("x-real-ip").and_then(|v| v.to_str().ok()) {
            let trimmed = xri.trim();
            // Validate the shape exactly like the XFF branch: a non-trusted
            // proxy may forward rather than overwrite `X-Real-IP`, letting a
            // client inject an arbitrary value. Varying that value per request
            // would key distinct auth-brake entries and so evade per-IP
            // blocking. Only trust a value that parses as an IP address;
            // otherwise fall through to the peer IP. A value that IS a trusted
            // proxy (e.g. the previous hop in a multi-tier chain that forwards
            // rather than overwrites the header) is skipped too, mirroring the
            // XFF walk's trust-skipping logic: returning a trusted-proxy IP
            // would collapse every client behind it into one brake bucket and
            // could satisfy an IP allowlist keyed on the proxy range.
            if let Ok(parsed) = trimmed.parse::<std::net::IpAddr>() {
                let ip = normalize_ip_for_ssrf(parsed);
                let trusted = trusted_proxies.iter().any(|cidr| cidr.contains(&ip));
                if !trusted {
                    return ip.to_string();
                }
            }
        }
    }
    match peer_ip {
        Some(ip) => ip.to_string(),
        None => "unknown".to_string(),
    }
}

/// Resolve the per-client IP used for auth-brake enforcement, or `None` when
/// the request cannot be attributed to a real client without collapsing many
/// callers into one shared bucket.
///
/// Mirrors [`client_ip`] but returns `None` in three cases where keying the
/// brake would cause a global self-DoS:
///
/// - **No peer address** (server not started with
///   `into_make_service_with_connect_info`): `client_ip` would fall back to the
///   shared `"unknown"` string, collapsing every peerless client into one
///   bucket.
/// - **Peer in a private / shared range but not a trusted proxy**: when the
///   operator has not configured `trusted_proxy_cidrs` (the default), every
///   request behind a reverse proxy / LB resolves to the proxy's own address.
///   Keying the brake on that address would let a single failing client block
///   every other client behind the same proxy for `block_secs`. Rather than
///   enforce on an address that is shared by many callers, refuse to enforce
///   (return `None`), exactly like the peerless case. Operators who want the
///   brake behind a proxy must configure `trusted_proxy_cidrs` so the
///   rightmost `X-Forwarded-For` entry (the real client) is used.
/// - **Trusted peer that yields no usable forwarded entry**: the XFF walk
///   found no non-trusted IP-shaped hop (no `X-Forwarded-For` at all — an L4
///   / TCP proxy cannot add one — garbage XFF, an all-trusted hop chain, or a
///   garbage `X-Real-IP`), so `client_ip` falls back to the peer itself —
///   the proxy's address, shared by every client behind it. Same refusal:
///   skip enforcement rather than block the whole proxy.
pub fn auth_brake_ip(
    headers: &HeaderMap,
    peer: Option<std::net::SocketAddr>,
    trusted_proxies: &[ipnet::IpNet],
) -> Option<String> {
    let peer_ip = peer.map(|sa| normalize_ip_for_ssrf(sa.ip()));
    let peer_is_trusted = match peer_ip {
        Some(ip) => trusted_proxies.iter().any(|cidr| cidr.contains(&ip)),
        None => false,
    };
    if !peer_is_trusted {
        let ip = peer_ip?;
        if is_shared_peer_address(&ip) {
            return None;
        }
        return Some(ip.to_string());
    }
    let resolved = client_ip(headers, peer, trusted_proxies);
    // Detect the fallback: a forwarded entry equal to the peer would itself be
    // a trusted proxy and therefore skipped by the walk, so `resolved` can
    // equal the peer only when no usable entry was found.
    if peer_ip.is_some_and(|p| resolved == p.to_string()) {
        return None;
    }
    Some(resolved)
}

/// Whether a peer address is a "shared" range — loopback, RFC1918 private,
/// link-local, or CGNAT — that a reverse proxy / load balancer (or a private
/// network) would present for many distinct clients. Reuses the SSRF private
/// CIDR table after IPv4-mapped IPv6 normalization.
fn is_shared_peer_address(ip: &IpAddr) -> bool {
    let ip = normalize_ip_for_ssrf(*ip);
    !check_ip_allowed(ip)
}

/// Reject JSON payloads whose nesting depth exceeds `max_depth`. Recursion
/// depth counts objects and arrays as one level each (scalars are depth 0).
/// Returns `Err(ApiError::bad_request)` when the depth is exceeded.
pub fn check_json_depth(value: &serde_json::Value, max_depth: usize) -> Result<(), ApiError> {
    /// Recursively compute the depth, short-circuiting as soon as it exceeds
    /// `cap`. Returning `None` signals "already over the cap" so the caller
    /// can stop traversing the rest of the tree — a document far over the
    /// limit must not pay full traversal cost (and an unbounded tree must not
    /// stack-overflow before being rejected).
    fn depth_of(value: &serde_json::Value, cap: usize) -> Option<usize> {
        match value {
            serde_json::Value::Object(map) => {
                // One level of nesting costs one unit of `cap`. If `cap` is
                // already 0 then this container alone pushes us over.
                let child_cap = cap.checked_sub(1)?;
                let mut max_child = 0;
                for (_, v) in map {
                    let d = depth_of(v, child_cap)?;
                    if d > max_child {
                        max_child = d;
                    }
                }
                Some(1 + max_child)
            }
            serde_json::Value::Array(arr) => {
                let child_cap = cap.checked_sub(1)?;
                let mut max_child = 0;
                for v in arr {
                    let d = depth_of(v, child_cap)?;
                    if d > max_child {
                        max_child = d;
                    }
                }
                Some(1 + max_child)
            }
            _ => Some(0),
        }
    }
    // The tree is over the limit iff any subtree already exceeds it; detect
    // that via the short-circuit and reject without computing full depth.
    if depth_of(value, max_depth).is_none() {
        return Err(ApiError::bad_request(format!(
            "request body nesting depth exceeds the maximum of {max_depth}"
        )));
    }
    Ok(())
}

/// Strip transport fingerprints and gateway-owned caller-spend snapshots.
/// Upstream response-cost/token receipts remain available for chained billing.
pub fn strip_leaking_headers(headers: &mut HeaderMap) {
    headers.remove("server");
    headers.remove("x-request-id");
    headers.remove("content-length");
    for name in crate::pipeline::KEY_SPEND_HEADERS {
        headers.remove(name);
    }
}

/// Strip every hop-by-hop header listed in RFC 7230 §6.1 (`connection`,
/// `keep-alive`, `proxy-authenticate`, `proxy-authorization`, `te`, `trailer`,
/// `transfer-encoding`, `upgrade`) so an upstream value is never forwarded to
/// the downstream client. Hop-by-hop headers describe a single transport hop;
/// relaying them on a response whose body has been re-framed (e.g. a chunked
/// upstream now buffered as a fixed `Vec<u8>`) creates a protocol mismatch.
/// Applied uniformly to the streaming relay, the buffered JSON success, and
/// the error paths of [`sanitize_upstream_response`].
pub fn strip_hop_by_hop_headers(headers: &mut HeaderMap) {
    headers.remove("transfer-encoding");
    headers.remove("connection");
    headers.remove("keep-alive");
    headers.remove("proxy-authenticate");
    headers.remove("proxy-authorization");
    headers.remove("te");
    headers.remove("trailer");
    headers.remove("upgrade");
}

/// Truncate a string to at most `max_chars` characters, appending an ellipsis
/// when truncation occurred.
pub fn truncate_error_message(msg: &str, max_chars: usize) -> String {
    if msg.chars().count() <= max_chars {
        return msg.to_string();
    }
    let mut out: String = msg.chars().take(max_chars).collect();
    out.push('…');
    out
}

/// Wrap an upstream byte stream so the relayed body can never exceed `cap`
/// bytes. Each chunk is forwarded while the running total stays under the cap;
/// the chunk that pushes the total over the cap is replaced with an `Err`
/// (which axum surfaces as a mid-stream body error, tearing down the
/// downstream connection) and the stream then ends. This bounds proxy memory
/// and egress bandwidth for a malicious / misbehaving upstream that streams an
/// unbounded body on the non-buffered relay path.
///
/// The relay is intentionally truncated at the cap: the over-cap chunk's bytes
/// are dropped and the downstream is cut there, so a client receives at most
/// `cap` bytes followed by a teardown. A genuine upstream transport error that
/// arrives after the cap is still surfaced (relayed as `Err`) rather than
/// silently swallowed.
fn capped_byte_stream<S>(
    stream: S,
    cap: usize,
) -> impl Stream<Item = Result<axum::body::Bytes, std::io::Error>>
where
    S: Stream<Item = Result<axum::body::Bytes, reqwest::Error>>,
{
    stream.scan(0usize, move |total, item| {
        let result = match item {
            Err(e) => {
                // Surface an upstream transport error even after the cap was
                // exceeded (the previous `None` swallowed it).
                Some(Err(std::io::Error::other(e)))
            }
            Ok(_) if *total > cap => None,
            Ok(bytes) => {
                *total = total.saturating_add(bytes.len());
                if *total > cap {
                    tracing::warn!(
                        accumulated = *total,
                        cap,
                        "upstream streaming body exceeded cap; aborting relay"
                    );
                    Some(Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "upstream response too large: streamed body exceeds limit",
                    )))
                } else {
                    Some(Ok(bytes))
                }
            }
        };
        std::future::ready(result)
    })
}

/// Apply sanitization to an upstream response: strip leaking headers and
/// truncate the body when it's an error (status >= 400) and over 500 chars.
///
/// Enforces [`MAX_RESPONSE_BODY_BYTES`]: responses that declare a larger
/// `Content-Length` (or whose streamed body exceeds the cap) are rejected with
/// an error so a malicious upstream cannot OOM the proxy.
///
/// Upstream *error* bodies are never rejected for size: an oversized error
/// payload (Cloudflare challenge pages and other SaaS HTML block pages
/// routinely exceed the cap) is truncated to [`MAX_ERROR_BODY_BYTES`] and the
/// upstream status is preserved — re-mapping the provider's real
/// 401/403/429/503 to an oxllm 500 because its error page is large would hide
/// the actionable status and body from the client (the reference surfaces the
/// upstream error status regardless of body size). See
/// `read_error_body_capped`.
pub async fn sanitize_upstream_response(response: reqwest::Response) -> Result<Response, ApiError> {
    let status = response.status();
    let media_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| {
            ct.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        })
        .unwrap_or_default();
    let json_or_unknown =
        media_type.is_empty() || media_type == "application/json" || media_type.ends_with("+json");
    let must_buffer = !status.is_success() || json_or_unknown;

    // Successful non-JSON payloads (batch result JSONL, file downloads,
    // audio/octet streams) do not need body inspection. Relay them
    // incrementally instead of pinning the entire upstream body in proxy
    // memory. Errors and JSON stay on the bounded buffered path so the body
    // guard and error truncation still apply. SSE success bodies are NOT
    // buffered here: `text/event-stream` is neither empty nor JSON, so for a
    // 2xx it takes the streaming relay branch below (transformed chat paths
    // own their own SSE pump).
    //
    // The relay is bounded by [`MAX_RESPONSE_BODY_BYTES`]: an upstream that
    // *advertises* more than the cap via `Content-Length` is rejected up front
    // (header-only check, no buffering), and a chunked / lying upstream is
    // aborted mid-stream by [`capped_byte_stream`] as soon as the running
    // total exceeds the cap — the downstream connection is torn down rather
    // than letting a malicious upstream pin unbounded proxy memory and egress
    // bandwidth.
    if status.is_success() && !must_buffer {
        check_content_length(response.headers(), MAX_RESPONSE_BODY_BYTES)?;
        let mut headers = response.headers().clone();
        strip_leaking_headers(&mut headers);
        // Strip every hop-by-hop header listed in RFC 7230 §6.1 so an
        // upstream `proxy-authenticate` challenge or a stray `upgrade` from a
        // non-WebSocket relay is not forwarded to the downstream client. The
        // buffered JSON / error paths rebuild the response from scratch and
        // are handled by the shared `strip_hop_by_hop_headers` call below.
        strip_hop_by_hop_headers(&mut headers);
        let capped = capped_byte_stream(response.bytes_stream(), MAX_RESPONSE_BODY_BYTES);
        let mut resp = Response::new(axum::body::Body::from_stream(capped));
        *resp.status_mut() = status;
        for (k, v) in headers.iter() {
            if let Ok(name) = axum::http::HeaderName::from_bytes(k.as_str().as_bytes()) {
                resp.headers_mut().append(name, v.clone());
            }
        }
        return Ok(resp);
    }

    // Buffered path (errors + JSON). Bounded by [`read_bounded_bytes`]: the
    // body is streamed chunk-by-chunk and aborted mid-stream once the running
    // total exceeds the cap, so a chunked / no-Content-Length upstream cannot
    // pin an unbounded body in proxy heap before the cap check fires (the
    // `Content-Length` pre-check inside is header-only). Errors are capped at
    // the tighter [`MAX_ERROR_BODY_BYTES`]; JSON successes at
    // [`MAX_RESPONSE_BODY_BYTES`].
    let cap = if status.is_success() {
        MAX_RESPONSE_BODY_BYTES
    } else {
        MAX_ERROR_BODY_BYTES
    };
    let mut headers = response.headers().clone();
    strip_leaking_headers(&mut headers);
    // Strip hop-by-hop headers on the buffered path too: the body is now a
    // fixed `Vec<u8>`, so a surviving `transfer-encoding: chunked` would
    // describe a framing the client no longer receives. Same RFC 7230 §6.1
    // set as the streaming relay branch above.
    strip_hop_by_hop_headers(&mut headers);
    let bytes = if status.is_success() {
        read_bounded_bytes(response, cap).await?
    } else {
        // Truncate, don't fail: the upstream status must survive an oversized
        // error body (see the function doc).
        read_error_body_capped(response, cap).await
    };

    if status.is_success() {
        let mut resp = Response::new(axum::body::Body::from(bytes));
        *resp.status_mut() = status;
        for (k, v) in headers.iter() {
            if let Ok(name) = axum::http::HeaderName::from_bytes(k.as_str().as_bytes()) {
                resp.headers_mut().append(name, v.clone());
            }
        }
        return Ok(resp);
    }

    // The error body was decoded to text above while the raw bytes were read;
    // reqwest performs no auto-decompression here, so a gzip/deflate
    // `content-encoding` would describe compressed bytes that the client no
    // longer receives. Drop it (and the now-stale `content-length`, already
    // stripped) so the surfaced body matches its headers.
    headers.remove("content-encoding");
    let body_text = String::from_utf8_lossy(&bytes).to_string();
    let truncated = truncate_error_message(&body_text, 500);
    let mut resp = Response::new(axum::body::Body::from(truncated));
    *resp.status_mut() = status;
    for (k, v) in headers.iter() {
        if let Ok(name) = axum::http::HeaderName::from_bytes(k.as_str().as_bytes()) {
            resp.headers_mut().append(name, v.clone());
        }
    }
    Ok(resp)
}

/// Reject a response when its `Content-Length` header exceeds `max_bytes`
/// (callers pass [`MAX_RESPONSE_BODY_BYTES`] for full bodies and a tighter
/// error cap where relevant). Responses without the header (chunked / unknown
/// length) are allowed through and bounded by [`check_bytes_limit`] after the
/// body is read.
pub fn check_content_length(headers: &HeaderMap, max_bytes: usize) -> Result<(), ApiError> {
    if let Some(cl) = headers.get("content-length").and_then(|v| v.to_str().ok()) {
        if let Ok(len) = cl.trim().parse::<usize>() {
            if len > max_bytes {
                return Err(ApiError::internal(format!(
                    "upstream response too large: Content-Length {len} exceeds limit \
                     {max_bytes} bytes"
                )));
            }
        }
    }
    Ok(())
}

/// Reject a body when its actual length exceeds [`MAX_RESPONSE_BODY_BYTES`].
/// This is the defense-in-depth check for responses whose `Content-Length` was
/// missing or lying.
pub fn check_bytes_limit(bytes: &[u8]) -> Result<(), ApiError> {
    if bytes.len() > MAX_RESPONSE_BODY_BYTES {
        return Err(ApiError::internal(format!(
            "upstream response too large: body {} bytes exceeds limit \
             {MAX_RESPONSE_BODY_BYTES} bytes",
            bytes.len()
        )));
    }
    Ok(())
}

/// Read an upstream response body into memory, bounded by `cap` bytes.
///
/// This is the streaming equivalent of `response.bytes().await` followed by
/// [`check_bytes_limit`]: it performs a `Content-Length` pre-check (rejecting
/// an advertised size above `cap` without any I/O), then
/// streams the body chunk-by-chunk, aborting mid-stream as soon as the running
/// total exceeds `cap`. A malicious or compromised upstream that streams a
/// multi-GB body is therefore rejected after roughly `cap` bytes rather than
/// pinning the entire payload in proxy heap before the post-read limit fires.
///
/// Callers that want the broad response cap pass [`MAX_RESPONSE_BODY_BYTES`];
/// callers reading an *error* body pass the tighter [`MAX_ERROR_BODY_BYTES`],
/// matching the chat error path.
pub async fn read_bounded_bytes(
    response: reqwest::Response,
    cap: usize,
) -> Result<Vec<u8>, ApiError> {
    check_content_length(response.headers(), cap)?;
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    let mut oversized = false;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if buf.len().saturating_add(bytes.len()) > cap {
                    oversized = true;
                    break;
                }
                buf.extend_from_slice(&bytes);
            }
            Err(e) => {
                // The reqwest detail can carry internal topology; log it
                // server-side and surface a generic message (the same
                // redaction contract as the chat network-error path).
                tracing::error!(
                    error.kind = "network",
                    detail = %e,
                    "failed to read upstream body"
                );
                return Err(ApiError::internal("failed to read upstream body"));
            }
        }
    }
    if oversized || buf.len() > cap {
        return Err(ApiError::internal(format!(
            "upstream body exceeds {cap} byte limit"
        )));
    }
    Ok(buf)
}

/// Read up to `cap` bytes of an upstream *error* body, truncating instead of
/// failing when the body is larger (the error-path counterpart of
/// [`read_bounded_bytes`]). Oversized upstream error payloads are routine
/// (Cloudflare challenge pages, SaaS HTML block pages); failing the read
/// would re-map the provider's real status to an oxllm 500 with a generic
/// message, hiding the actionable status and body from the client and from
/// retry/alerting semantics. The body is cut at exactly `cap` bytes (the
/// surfaced message is truncated further downstream), mirroring what the
/// ≤cap path already does.
///
/// A transport error mid-read surfaces whatever partial bytes were already
/// collected (with the reqwest detail logged server-side only, matching
/// [`read_bounded_bytes`]'s redaction contract): the upstream status plus a
/// partial body is more truthful than a generic 500.
pub(crate) async fn read_error_body_capped(response: reqwest::Response, cap: usize) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::with_capacity(cap);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if buf.len() >= cap {
                    break;
                }
                let remaining = cap - buf.len();
                if bytes.len() > remaining {
                    buf.extend_from_slice(&bytes[..remaining]);
                    // debug, not warn: an upstream serving oversized error
                    // bodies does so for every request during an incident,
                    // and per-request warns would flood the log (the same
                    // log-volume hazard the auth-brake one-shots guard
                    // against).
                    tracing::debug!(
                        cap,
                        "upstream error body exceeds cap; truncating to preserve the upstream status"
                    );
                    break;
                }
                buf.extend_from_slice(&bytes);
            }
            Err(e) => {
                tracing::error!(
                    error.kind = "network",
                    detail = %e,
                    "failed to read upstream error body; surfacing partial body"
                );
                break;
            }
        }
    }
    buf
}

/// Whether a header or body value is "secret-bearing" and must be redacted
/// from info-level logs. Over-matches on purpose (safer to over-redact).
pub fn is_secret_key(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("authorization")
        || lower.contains("api_key")
        || lower.contains("apikey")
        || lower.contains("api-key")
        || lower.contains("password")
        || lower.contains("secret")
        || lower.contains("token")
}

/// Render endpoint context without URL credentials, secret query parameters,
/// fragments, or known webhook capability paths. Query-name matching ignores
/// ASCII case and percent encoding. Malformed/opaque non-SQLite URLs fail
/// closed rather than echoing a possibly credential-bearing raw input.
pub fn sanitize_url(url: &str) -> String {
    if url
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("sqlite:"))
    {
        // SQLite DSNs name files, not network authorities. An '@' or '=' in a
        // filename is ordinary endpoint context, not username/password data.
        return oxllm_core::redaction::redact_connection_text(url);
    }
    let Ok(mut parsed) = url::Url::parse(url) else {
        return "[invalid URL redacted]".to_string();
    };
    if parsed.host_str().is_none() {
        return "[invalid URL redacted]".to_string();
    }
    let _ = parsed.set_username("");
    let _ = parsed.set_password(None);
    parsed.set_fragment(None);
    let query = parsed.query().map(oxllm_core::redaction::redact_url_query);
    parsed.set_query(query.as_deref().filter(|query| !query.is_empty()));
    if parsed
        .host_str()
        .is_some_and(oxllm_core::redaction::is_capability_url_host)
        && parsed.path() != "/"
    {
        parsed.set_path("/***");
    }
    parsed.to_string()
}

/// Log-safe rendering for AWS SigV4 / other presigned URLs.
///
/// [`sanitize_url`] now removes signing material as well: every `X-Amz-*` /
/// `X-Goog-*` query pair, plus credentials, tokens and signatures, regardless
/// of case or encoded names. This entry point retains the same fail-closed
/// behavior for malformed inputs and preserves non-secret routing parameters.
pub fn sanitize_signed_url(url: &str) -> String {
    sanitize_url(url)
}

/// Redact a header value if its name is secret-bearing. Returns `[REDACTED]`.
pub fn redact_header_value(name: &str, value: &str) -> String {
    if is_secret_key(name) {
        "[REDACTED]".to_string()
    } else {
        value.to_string()
    }
}

/// Capture the actual tracing output of a synchronous regression test without
/// changing the process-global subscriber or other test threads.
#[cfg(test)]
pub(crate) fn capture_test_logs<F, T>(f: F) -> (T, String)
where
    F: FnOnce() -> T,
{
    #[derive(Clone)]
    struct Writer(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let bytes = Arc::new(Mutex::new(Vec::new()));
    let writer = Writer(bytes.clone());
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .without_time()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let result = tracing::subscriber::with_default(subscriber, f);
    let log = String::from_utf8_lossy(&bytes.lock()).into_owned();
    (result, log)
}

/// One-shot gate shared by the outer guard and standalone auth backstop.
/// A deployment behind a reverse proxy without `trusted_proxy_cidrs` (or with
/// a trusted proxy that forwards no usable `X-Forwarded-For`) skips per-IP
/// enforcement on **every** request; a per-request warn would flood the log.
/// Mirrors [`CAP_WARN_EMITTED`]: warn once per process, then `debug!`.
static BRAKE_SKIP_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn log_auth_brake_skip(
    peer: Option<std::net::SocketAddr>,
    trusted_proxies: &[ipnet::IpNet],
) {
    let reason = match peer {
        None => "missing_peer",
        Some(peer)
            if trusted_proxies
                .iter()
                .any(|cidr| cidr.contains(&normalize_ip_for_ssrf(peer.ip()))) =>
        {
            "missing_forwarded_client"
        }
        Some(_) => "untrusted_shared_peer",
    };
    if !BRAKE_SKIP_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            reason,
            "auth brake active but this request has no attributable client IP; \
             skipping per-IP enforcement to avoid blocking a shared peer. \
             Configure trusted_proxy_cidrs (and XFF forwarding on the proxy), \
             or connect-info serving when reason is missing_peer."
        );
    } else {
        tracing::debug!(reason, "auth brake skip: no attributable client IP");
    }
}

/// Auth middleware wrapper that applies the auth brake: blocks requests from
/// IPs that have exceeded the failure budget.
///
/// Requests whose client IP cannot be attributed without collapsing many
/// callers into one shared bucket (no peer, an untrusted private/proxy peer,
/// or a trusted proxy without a usable `X-Forwarded-For`) skip enforcement
/// entirely — see [`auth_brake_ip`]. The first such skip is surfaced with a
/// one-shot warn so an operator silently running without per-IP enforcement
/// behind a proxy can see it.
pub async fn auth_brake_guard(
    State(shared): State<SharedState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    let state = shared.load_full();
    let Some(brake) = &state.auth_brake else {
        return next.run(request).await;
    };
    let peer = request
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0);
    let Some(ip) = auth_brake_ip(&headers, peer, &state.trusted_proxies) else {
        // Cannot attribute the request to a real client; skip enforcement
        // rather than key the brake on a shared address (global self-DoS).
        log_auth_brake_skip(peer, &state.trusted_proxies);
        return next.run(request).await;
    };
    if brake.is_blocked(&ip).await {
        tracing::warn!(ip = %ip, "auth brake: IP blocked");
        let mut err = ApiError::rate_limited("too many auth failures; blocked").into_response();
        // Report the *remaining* block time, not the full window: a client
        // blocked part-way through its window should not be told to wait the
        // full `block_secs` again. Falls back to `block_secs` when the
        // remaining time cannot be recovered (distributed mode).
        let retry_after = brake
            .remaining_block_secs(&ip)
            .await
            .unwrap_or_else(|| brake.block_secs());
        if let Ok(v) = axum::http::HeaderValue::from_str(&retry_after.to_string()) {
            err.headers_mut().insert("retry-after", v);
        }
        return err;
    }
    next.run(request).await
}

/// IP-allowlist middleware. When `general_settings.allowed_ips` is configured,
/// requests whose resolved client IP does not fall inside any of the listed
/// CIDR ranges are rejected with 403. When unset, all IPs are allowed. An
/// explicitly empty list (`allowed_ips: []`) denies all requests — the reference
/// parity: `_check_valid_ip` treats an empty allowlist as matching no one
/// (fail-closed), so the setting can never be read as "allow everyone".
///
/// The client IP is resolved via the same `trusted_proxy_cidrs`-aware logic
/// used by the auth brake, so an `X-Forwarded-For` header is only honored
/// when the peer is a trusted proxy. NOTE: without `trusted_proxy_cidrs`, the
/// allowlist checks the **peer** address, not the client — behind a proxy/LB
/// that collapses every client to the peer, so an allowlist that happens to
/// include the proxy's own range lets every client behind it pass. A startup
/// warning flags that configuration (see `state::build`); operators who want
/// the allowlist keyed on the real client must configure
/// `trusted_proxy_cidrs`.
pub async fn allowed_ips_guard(
    State(shared): State<SharedState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    let state = shared.load_full();
    let Some(allowed) = state.allowed_ips.as_deref() else {
        return next.run(request).await;
    };
    let peer = request
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0);
    let ip_str = client_ip(&headers, peer, &state.trusted_proxies);
    if ip_allowed(&ip_str, allowed) {
        next.run(request).await
    } else {
        tracing::warn!(ip = %ip_str, "allowed_ips: client IP rejected");
        ApiError::forbidden("client IP not in allowlist").into_response()
    }
}

/// Returns `true` when `ip_str` parses as an IP address that is contained in
/// any of the `allowed` CIDR ranges (after IPv4-mapped IPv6 normalization, so
/// a dual-stack peer presentation `::ffff:a.b.c.d` matches IPv4 CIDRs).
/// Unparseable inputs are rejected, as is any input when `allowed` is empty
/// (deny all).
pub(crate) fn ip_allowed(ip_str: &str, allowed: &[ipnet::IpNet]) -> bool {
    let Ok(ip) = ip_str.parse::<std::net::IpAddr>() else {
        return false;
    };
    let ip = normalize_ip_for_ssrf(ip);
    allowed.iter().any(|cidr| cidr.contains(&ip))
}

/// Parse a list of CIDR strings into `IpNet`s, skipping unparseable entries
/// with a warning (never a hard error, per the unknown-config-keys rule).
pub fn parse_cidr_list(raw: &[String]) -> Vec<ipnet::IpNet> {
    raw.iter()
        .filter_map(|s| match s.parse::<ipnet::IpNet>() {
            Ok(net) => Some(net),
            Err(e) => {
                tracing::error!(cidr = %s, error = %e, "ignoring unparseable CIDR");
                None
            }
        })
        .collect()
}

/// Private / loopback / link-local CIDR ranges that upstream URLs must not
/// resolve to. This is the runtime equivalent of the `PRIVATE_CIDRS` list
/// used at config-load time in `config/loader.rs`. Kept in `security.rs`
/// (rather than re-exported from loader) so the runtime SSRF check has no
/// dependency on the config module and can be invoked from the upstream
/// pipeline without pulling config-only types into the hot path.
///
/// See [`check_ip_allowed`] and [`is_private_host`] for the runtime checks
/// that consume this list. See [`validate_upstream_ip`] for the
/// request-time entry point.
///
/// Note: the reference enforces this as a default-deny via Python's
/// `ipaddress.ip_address.is_global` (RFC 6890) plus an explicit exception
/// list for cloud-internal IPs that *are* globally routable. oxllm uses an
/// explicit allow/deny list instead; [`CLOUD_METADATA_EXCEPTIONS`] holds the
/// globally-routable cloud-fabric IPs that oxllm also blocks, and is folded
/// into `PRIVATE_CIDR_NETS` so the hot path consults a single table.
/// The IPv4 entries cover the IANA special-purpose registry (including the
/// 6to4 relay anycast `192.88.99.0/24` and the AS112 relay anycasts
/// `192.31.196.0/24` / `192.52.193.0/24`, matching `is_global`'s exclusion
/// of the newer registry entries).
pub const PRIVATE_CIDRS: &[&str] = &[
    "0.0.0.0/8",
    "10.0.0.0/8",
    "127.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.0.0/16",
    "100.64.0.0/10",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.31.196.0/24",
    "192.52.193.0/24",
    "192.88.99.0/24",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
    "2001:db8::/32",
];

/// Globally-routable cloud-fabric IPs that are reachable from inside a cloud
/// VM via special in-fabric routing and therefore must be blocked by the
/// SSRF guard even though they are not part of any RFC 6890 private range.
/// Mirrors the reference cloud-metadata exception list,
/// extended with the Alibaba Cloud
/// metadata service (a shared gap in the reference).
///
/// - **`168.63.129.16/32`** — Azure Wire Server. Reachable from inside any
///   Azure VM; provides guest-agent / VM management services.
/// - **`100.100.100.200/32`** — Alibaba Cloud metadata service (ECS
///   meta-server). Globally routable, reachable from inside any Alibaba ECS
///   instance.
pub const CLOUD_METADATA_EXCEPTIONS: &[&str] = &["168.63.129.16/32", "100.100.100.200/32"];

/// Map IPv4-mapped (`::ffff:x.x.x.x`) and IPv4-compatible (`::x.x.x.x`,
/// deprecated) IPv6 addresses to their IPv4 form so the private-CIDR checks
/// apply. Other addresses are returned unchanged.
/// This mirrors the helper used at config-load time: an attacker behind a
/// DNS rebinding can return an IPv4-mapped / IPv4-compatible IPv6 that
/// bypasses a naive IPv4-only private-range check.
pub fn normalize_ip_for_ssrf(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return IpAddr::V4(v4);
            }
            // IPv4-compatible IPv6 (`::a.b.c.d`, RFC 4291 deprecated): the low
            // 32 bits embed an IPv4 address. Normalize it so the private-range
            // checks apply. `::` / `::1` fall through to `0.0.0.x` (private),
            // which is correct for SSRF purposes — they are not connectable
            // destinations either way.
            let segs = v6.segments();
            if segs[..6].iter().all(|&s| s == 0) {
                return IpAddr::V4(std::net::Ipv4Addr::new(
                    (segs[6] >> 8) as u8,
                    (segs[6] & 0xff) as u8,
                    (segs[7] >> 8) as u8,
                    (segs[7] & 0xff) as u8,
                ));
            }
            ip
        }
        other => other,
    }
}

/// The parsed form of [`PRIVATE_CIDRS`] plus [`CLOUD_METADATA_EXCEPTIONS`].
/// Parsing 10+ CIDR strings on every request is pure waste on the upstream
/// hot path (`is_private_host_async` → `check_ip_allowed` runs per
/// request), so the parsed `IpNet`s are computed once and reused.
/// Unparseable entries are skipped (they are compile-time constants here,
/// so a parse failure is a programming error worth a warning).
static PRIVATE_CIDR_NETS: std::sync::LazyLock<Vec<ipnet::IpNet>> = std::sync::LazyLock::new(|| {
    PRIVATE_CIDRS
        .iter()
        .chain(CLOUD_METADATA_EXCEPTIONS.iter())
        .filter_map(|s| match s.parse::<ipnet::IpNet>() {
            Ok(net) => Some(net),
            Err(e) => {
                tracing::error!(cidr = *s, error = %e, "PRIVATE_CIDRS entry failed to parse");
                None
            }
        })
        .collect()
});

/// Returns `true` when `ip` falls inside any of the [`PRIVATE_CIDRS`] ranges
/// (after IPv4-mapped IPv6 normalization). This is the runtime SSRF check;
/// it complements the config-load-time `validate_api_base` in
/// `config/loader.rs` by re-checking immediately before the upstream HTTP
/// request, closing the DNS-rebinding window between config load and
/// request execution.
///
/// A NAT64 well-known prefix address (`64:ff9b::/96`) is evaluated through
/// its embedded IPv4 address: on a NAT64-enabled host the embedded private
/// IPv4 remains reachable, so the prefix must not launder a private
/// destination past the table.
pub fn check_ip_allowed(ip: IpAddr) -> bool {
    let ip = normalize_ip_for_ssrf(ip);
    if let IpAddr::V6(v6) = ip {
        let segs = v6.segments();
        if segs[0] == 0x64 && segs[1] == 0xff9b && segs[2..6].iter().all(|&s| s == 0) {
            return check_ip_allowed(IpAddr::V4(std::net::Ipv4Addr::new(
                (segs[6] >> 8) as u8,
                (segs[6] & 0xff) as u8,
                (segs[7] >> 8) as u8,
                (segs[7] & 0xff) as u8,
            )));
        }
    }
    for cidr in PRIVATE_CIDR_NETS.iter() {
        if cidr.contains(&ip) {
            return false;
        }
    }
    true
}

/// Resolve `host` and return `true` iff every resolved IP address is
/// allowed by [`check_ip_allowed`] (i.e. none of them is private /
/// loopback / link-local). A host that fails to resolve returns `false`
/// (fail-closed) to prevent a DNS-rebinding attack from smuggling in a
/// private IP after the lookup.
///
/// This is the runtime equivalent of the config-load-time `api_base` check,
/// intended to be called from the upstream HTTP path immediately before
/// the request goes on the wire. Callers that need to permit private IPs
/// (e.g. local Ollama deployments) must opt out explicitly — see the
/// `allow_private_api_base` setting in `general_settings`.
pub fn is_private_host(host: &str) -> bool {
    // Literal IP hosts skip DNS resolution entirely.
    let trimmed = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return !check_ip_allowed(ip);
    }
    match (host, 80u16).to_socket_addrs() {
        Ok(addrs) => {
            let collected: Vec<IpAddr> = addrs.map(|sa| sa.ip()).collect();
            if collected.is_empty() {
                // No results — fail-closed.
                return true;
            }
            collected.iter().any(|ip| !check_ip_allowed(*ip))
        }
        Err(_) => true,
    }
}

/// Best-effort runtime DNS cache for [`is_private_host_cached`]. Maps a
/// hostname to `(cached_at, is_private)`. Entries older than
/// [`DNS_CACHE_TTL`] are treated as misses and re-resolved. The cache is
/// process-local; in a multi-replica deployment each replica maintains its
/// own view, which is fine because the SSRF decision is per-request.
///
/// This is intentionally a simple `Mutex<HashMap>` rather than a full
/// cached DNS resolver: it exists only to keep one client hammering the same
/// upstream host from re-issuing `getaddrinfo` on every single request.
///
/// Because `is_private_host_async` is called with **user-supplied** hostnames
/// on the image-URL path, distinct hosts accumulate forever without an
/// eviction path — an authenticated caller could grow the map without bound
/// by submitting distinct `image_url` hosts. The inner struct therefore
/// carries a call counter that triggers a periodic sweep (mirroring
/// [`AuthBrake::sweep_locked`]) so stale entries are reclaimed, plus a hard
/// cap ([`DEFAULT_MAX_DNS_CACHE_ENTRIES`]) with oldest-entry eviction
/// (mirroring [`AuthBrake::evict_oldest_locked`]) so the map stays bounded
/// even under a burst of fresh distinct-host inserts where no entry is yet
/// stale enough for the sweep to reclaim.
static DNS_CACHE: std::sync::OnceLock<Mutex<DnsCacheInner>> = std::sync::OnceLock::new();

/// TTL for entries in [`DNS_CACHE`]. 10 seconds is short enough that a
/// DNS-rebinding attack launched seconds after a benign lookup is still
/// caught, but long enough to amortize the lookup across a burst of
/// requests to the same upstream.
const DNS_CACHE_TTL: Duration = Duration::from_secs(10);

/// Number of `is_private_host_async` cache-miss insertions between sweeps of
/// stale [`DNS_CACHE`] entries. Sweeping on every insert would make the hot
/// path O(n) in the map size; doing it every Nth insert bounds the amortized
/// cost while still evicting dead entries often enough to prevent unbounded
/// growth under distinct-host probing (mirroring [`SWEEP_INTERVAL`] for the
/// auth brake).
const DNS_CACHE_SWEEP_INTERVAL: u64 = 100;

/// Hard cap on the number of distinct hostnames tracked in [`DNS_CACHE`],
/// mirroring [`DEFAULT_MAX_TRACKED_IPS`] for [`AuthBrake`]. The TTL-based
/// sweep ([`dns_cache_sweep_locked`]) reclaims entries older than
/// [`DNS_CACHE_TTL`], but every entry survives at least `DNS_CACHE_TTL`
/// before it is sweepable, so under sustained distinct-host probing (the
/// primary user-controlled entry point is `is_private_host_async` called with
/// the hostname of a user-supplied `image_url`) the map grows to roughly
/// `rate × DNS_CACHE_TTL` before reaching steady-state turnover. Beyond this
/// cap the entry with the oldest `cached_at` is evicted to make room for a
/// new host (see [`dns_cache_evict_oldest_locked`]), matching how
/// [`AuthBrake`] bounds its `by_ip` map.
const DEFAULT_MAX_DNS_CACHE_ENTRIES: usize = 100_000;

#[derive(Default)]
struct DnsCacheInner {
    entries: HashMap<String, (Instant, bool)>,
    inserts_since_sweep: u64,
}

fn dns_cache() -> &'static Mutex<DnsCacheInner> {
    DNS_CACHE.get_or_init(|| Mutex::new(DnsCacheInner::default()))
}

/// Remove every [`DNS_CACHE`] entry older than [`DNS_CACHE_TTL`]. Such
/// entries are already treated as misses on lookup, so dropping them is
/// lossless — they are simply dead weight pinning memory under distinct-host
/// probing. Called occasionally from [`is_private_host_async`] to amortize
/// the O(n) scan cost (every [`DNS_CACHE_SWEEP_INTERVAL`]th insert).
fn dns_cache_sweep_locked(inner: &mut DnsCacheInner) {
    let now = Instant::now();
    let before = inner.entries.len();
    inner
        .entries
        .retain(|_, (cached_at, _)| now.duration_since(*cached_at) < DNS_CACHE_TTL);
    let removed = before.saturating_sub(inner.entries.len());
    if removed > 0 {
        tracing::debug!(
            removed,
            remaining = inner.entries.len(),
            "ssrf dns cache sweep: evicted stale entries"
        );
    }
}

/// Remove the entry with the oldest `cached_at` so a new hostname can be
/// inserted without growing [`DNS_CACHE`] past [`DEFAULT_MAX_DNS_CACHE_ENTRIES`].
/// O(n) in the map size, but only reached when the map is already at the cap
/// (the periodic sweep handles normal-case reclamation). Mirrors
/// [`AuthBrake::evict_oldest_locked`].
fn dns_cache_evict_oldest_locked(inner: &mut DnsCacheInner) {
    let mut oldest_key: Option<String> = None;
    let mut oldest_at = Instant::now();
    for (key, (cached_at, _)) in inner.entries.iter() {
        if oldest_key.is_none() || *cached_at < oldest_at {
            oldest_key = Some(key.clone());
            oldest_at = *cached_at;
        }
    }
    if let Some(key) = oldest_key {
        inner.entries.remove(&key);
    }
}

/// Best-effort DNS resolution for the SSRF check. A DNS failure is NOT
/// treated as "private": the upstream request will fail on its own, so we
/// only block on a *positive* private resolution. This is the fail-open
/// counterpart to [`is_private_host`]'s fail-closed behavior, used on the
/// request hot path where a transient resolver outage must not take every
/// upstream offline.
///
/// This performs a synchronous `getaddrinfo` and must never be called from
/// an async context; see [`is_private_host_async`], which wraps it in
/// [`tokio::task::spawn_blocking`].
fn resolve_is_private_best_effort(host: &str) -> bool {
    match (host, 80u16).to_socket_addrs() {
        Ok(addrs) => {
            let collected: Vec<IpAddr> = addrs.map(|sa| sa.ip()).collect();
            if collected.is_empty() {
                false
            } else {
                collected.iter().any(|ip| !check_ip_allowed(*ip))
            }
        }
        Err(_) => false,
    }
}

/// Synchronous, cache-only lookup for use where blocking DNS is unacceptable.
/// Returns the cached verdict when a fresh (`DNS_CACHE_TTL`) entry exists,
/// and `false` (allow) on a miss or for an unresolvable hostname. **It does
/// not perform DNS resolution** — a cache miss does not block the request,
/// it just means the decision is deferred to a later
/// [`is_private_host_async`] call (or, in the worst case, to the upstream
/// HTTP request itself, which will fail naturally if the host is bad).
///
/// Literal IP hosts skip the cache entirely and are evaluated directly
/// against [`check_ip_allowed`] (they don't vary over time so caching would
/// be pointless).
///
/// Callers on the async hot path should prefer [`is_private_host_async`],
/// which performs the blocking DNS lookup on a `spawn_blocking` thread
/// instead of stalling the executor. This synchronous variant is kept for
/// contexts that cannot await (and for tests).
pub fn is_private_host_cached(host: &str) -> bool {
    // Literal IP hosts skip DNS resolution (and the cache) entirely.
    let trimmed = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return !check_ip_allowed(ip);
    }

    let now = Instant::now();
    let cache = dns_cache().lock();
    if let Some((cached_at, result)) = cache.entries.get(host) {
        if now.duration_since(*cached_at) < DNS_CACHE_TTL {
            return *result;
        }
    }
    // Cache miss: do NOT do a synchronous `getaddrinfo` here — that would
    // stall the executor thread and every other task parked on it. Fail
    // open (allow the request); the async variant / the upstream HTTP call
    // will catch a genuinely bad host.
    false
}

/// Async, cached, best-effort SSRF check for the upstream hot path. Returns
/// `true` when `host` resolves to at least one private / loopback /
/// link-local IP address.
///
/// This is the async-safe replacement for [`is_private_host_cached`] on the
/// request path. It:
///
/// 1. Resolves literal IP hosts synchronously (no DNS, no cache).
/// 2. Returns the cached verdict on a cache hit (respecting
///    `DNS_CACHE_TTL`).
/// 3. On a cache miss, offloads the blocking `getaddrinfo` syscall to
///    [`tokio::task::spawn_blocking`] so a slow resolver cannot stall the
///    async executor thread (and the hundreds of other tasks parked on it).
///    The result is cached and returned.
///
/// Like [`is_private_host_cached`] (and unlike [`is_private_host`]), this is
/// **best-effort**: a DNS failure returns `false` (allow the request). The
/// upstream HTTP call will then fail naturally on its own — we don't double
/// as a hostname validator here. Only a successful DNS lookup that returns a
/// known-private IP blocks the request.
///
/// This fail-open behavior is deliberately kept even on the request hot path
/// because the *authoritative* SSRF enforcement now lives at the connection
/// layer in [`SsrfResolver`] (installed on the shared `reqwest::Client` when
/// `allow_private_api_base` is unset). That resolver re-resolves at connect
/// time and rejects private addresses, so a transient resolver outage that
/// makes this pre-check fail-open no longer takes SSRF offline — the connect
/// still cannot reach a private IP. This pre-check is therefore
/// defense-in-depth: a fast, cacheable early reject for the common case,
/// short-circuiting before the request goes on the wire, but never the sole
/// enforcement point. [`is_private_host`] / [`validate_upstream_ip`] remain
/// available as standalone **fail-closed** helpers.
pub async fn is_private_host_async(host: &str) -> bool {
    // Literal IP hosts skip DNS resolution (and the cache) entirely.
    let trimmed = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return !check_ip_allowed(ip);
    }

    let now = Instant::now();
    {
        let cache = dns_cache().lock();
        if let Some((cached_at, result)) = cache.entries.get(host) {
            if now.duration_since(*cached_at) < DNS_CACHE_TTL {
                return *result;
            }
        }
    }

    // Cache miss: move the synchronous `getaddrinfo` off the async executor
    // thread. A slow resolver (5s+ per call) would otherwise stall the
    // worker and every other task parked on it. `spawn_blocking` parks the
    // future without blocking the thread.
    let host_owned = host.to_string();
    let result = tokio::task::spawn_blocking(move || resolve_is_private_best_effort(&host_owned))
        .await
        .unwrap_or(false);

    let now = Instant::now();
    let mut cache = dns_cache().lock();
    // Enforce the hard cap: the TTL sweep can only reclaim entries already
    // older than `DNS_CACHE_TTL`, so a burst of fresh distinct-host inserts
    // would otherwise grow the map to `rate × DNS_CACHE_TTL` before any entry
    // becomes sweepable. When at the cap and inserting a new host, evict the
    // oldest entry by `cached_at` (mirroring `AuthBrake::evict_oldest_locked`).
    // An upsert of an existing host never grows the map, so it skips eviction.
    if cache.entries.len() >= DEFAULT_MAX_DNS_CACHE_ENTRIES && !cache.entries.contains_key(host) {
        dns_cache_evict_oldest_locked(&mut cache);
    }
    cache.entries.insert(host.to_string(), (now, result));
    // Periodically reclaim memory from stale entries so an attacker cannot
    // grow the map without bound by submitting distinct `image_url` hosts.
    cache.inserts_since_sweep = cache.inserts_since_sweep.saturating_add(1);
    if cache.inserts_since_sweep >= DNS_CACHE_SWEEP_INTERVAL {
        cache.inserts_since_sweep = 0;
        dns_cache_sweep_locked(&mut cache);
    }
    result
}

/// A [`reqwest::dns::Resolve`] that drops any resolved address falling inside
/// the [`PRIVATE_CIDRS`] ranges (after IPv4-mapped-IPv6 normalization via
/// [`check_ip_allowed`]), so a connection is never opened to a private /
/// loopback / link-local IP.
///
/// This closes the DNS-rebinding TOCTOU between the best-effort pre-check
/// ([`is_private_host_async`]) and the actual HTTP fetch: an attacker who
/// controls authoritative DNS can answer the pre-check's lookup with a public
/// IP and `reqwest`'s lookup moments later with a private IP (e.g. cloud
/// metadata `169.254.169.254`). By moving the private-range check *inside* the
/// resolver, the verdict and the connect share one resolution — there is no
/// second lookup to rebind against. If every resolved address is private,
/// resolution fails and the request errors rather than connecting.
///
/// Installed on the shared `reqwest::Client` (see `state::build`) whenever
/// `general_settings.allow_private_api_base` is not set, so both the upstream
/// POST path and the user-supplied image-URL download path are protected at
/// the connection layer regardless of the preceding best-effort pre-check.
#[derive(Debug, Default, Clone, Copy)]
pub struct SsrfResolver {
    allow_private: bool,
    ipv4_only: bool,
}

impl SsrfResolver {
    /// Build a resolver that rejects private ranges and permits both families.
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn with_options(allow_private: bool, ipv4_only: bool) -> Self {
        Self {
            allow_private,
            ipv4_only,
        }
    }
}

impl reqwest::dns::Resolve for SsrfResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        let allow_private = self.allow_private;
        let ipv4_only = self.ipv4_only;
        Box::pin(async move {
            let host_for_lookup = host.clone();
            let addrs = match tokio::task::spawn_blocking(move || {
                (host_for_lookup.as_str(), 0u16).to_socket_addrs()
            })
            .await
            {
                Ok(Ok(addrs)) => addrs,
                Ok(Err(e)) => return Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                Err(e) => return Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
            };
            let filtered: Vec<std::net::SocketAddr> = addrs
                .filter(|sa| {
                    (allow_private || check_ip_allowed(sa.ip())) && (!ipv4_only || sa.is_ipv4())
                })
                .collect();
            if filtered.is_empty() {
                tracing::warn!(
                    host = %host,
                    ipv4_only,
                    "DNS resolver: no address matches the configured network policy"
                );
                return Err(if ipv4_only {
                    format!("no permitted IPv4 address for '{host}'")
                } else {
                    format!("all addresses for '{host}' are in a blocked private range")
                }
                .into());
            }
            Ok(Box::new(filtered.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// Build an HTTP client for operator-configured guardrail / webhook-hook
/// targets, with the SSRF-aware DNS resolver
/// ([`SsrfResolver`]) installed unless `allow_private` is set (matching the
/// shared-upstream-client opt-in in `state::build`).
///
/// These targets are reachable from request-time content (MCP / custom
/// guardrail bodies carry client text), so the connect-time private-range
/// filter closes the DNS-rebinding gap that a config-load-only validation and
/// a cache-only pre-check leave open: an attacker who can rebind the target's
/// DNS after config load cannot steer the connection to a private address.
pub fn build_ssrf_http_client(
    allow_private: bool,
    timeout: Duration,
) -> reqwest::Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none());
    if !allow_private {
        builder = builder.dns_resolver(std::sync::Arc::new(SsrfResolver::new()));
    }
    builder.build()
}

/// Validate that the hostname embedded in an upstream URL does not resolve
/// to a private / loopback / link-local IP. Returns `Ok(())` when the URL
/// is safe to call and `Err(message)` otherwise.
///
/// This is a standalone **fail-closed** SSRF helper: an unparseable URL or an
/// unresolvable host returns `Err`. It is *not* on the live request hot path
/// (which relies on the [`SsrfResolver`] installed on the `reqwest::Client`
/// for connection-layer enforcement, plus the best-effort
/// [`is_private_host_async`] pre-check). It remains available as a
/// reusable, synchronous, fail-closed check for ad-hoc callers and tests.
///
/// Config-load validation (`config::loader::validate_api_base`) runs once at
/// startup; between then and the actual HTTP request, DNS may change (DNS
/// rebinding), so a runtime check is required to close the gap. Callers that
/// need to bypass the check for legitimate private upstreams (e.g. a local
/// Ollama instance) must do so based on the deployment's
/// `allow_private_api_base` setting before relying on this function.
pub fn validate_upstream_ip(url: &str) -> Result<(), String> {
    let parsed = url::Url::parse(url).map_err(|e| format!("invalid URL: {e}"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| "upstream URL is missing a host".to_string())?;
    if is_private_host(host) {
        return Err(format!(
            "upstream host '{host}' resolves to a private / loopback / link-local IP"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_response_body_bytes_constant_is_50mb() {
        assert_eq!(MAX_RESPONSE_BODY_BYTES, 50 * 1024 * 1024);
    }

    #[test]
    fn check_content_length_rejects_oversized() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-length",
            format!("{}", MAX_RESPONSE_BODY_BYTES + 1).parse().unwrap(),
        );
        let result = check_content_length(&headers, MAX_RESPONSE_BODY_BYTES);
        assert!(result.is_err(), "oversized Content-Length must be rejected");
    }

    #[test]
    fn check_content_length_honors_caller_cap() {
        // The pre-check must compare against the caller's cap, not only the
        // global 50 MB limit: an error-body reader passing the tighter 10 KB
        // cap needs the header pre-check to fire for a 32 KB advertisement.
        let mut headers = HeaderMap::new();
        headers.insert("content-length", "32768".parse().unwrap());
        assert!(
            check_content_length(&headers, MAX_ERROR_BODY_BYTES).is_err(),
            "32 KB Content-Length must trip the 10 KB error cap pre-check"
        );
        assert!(
            check_content_length(&headers, MAX_RESPONSE_BODY_BYTES).is_ok(),
            "the same header is within the global cap"
        );
    }

    #[test]
    fn check_content_length_allows_within_limit() {
        let mut headers = HeaderMap::new();
        headers.insert("content-length", "1024".parse().unwrap());
        assert!(check_content_length(&headers, MAX_RESPONSE_BODY_BYTES).is_ok());
    }

    #[test]
    fn check_content_length_ignores_missing_header() {
        let headers = HeaderMap::new();
        assert!(check_content_length(&headers, MAX_RESPONSE_BODY_BYTES).is_ok());
    }

    #[test]
    fn check_bytes_limit_rejects_oversized_body() {
        let big = vec![0u8; MAX_RESPONSE_BODY_BYTES + 1];
        let result = check_bytes_limit(&big);
        assert!(
            result.is_err(),
            "body over the limit must be rejected (defense-in-depth)"
        );
    }

    #[test]
    fn check_bytes_limit_allows_small_body() {
        let small = vec![0u8; 256];
        assert!(check_bytes_limit(&small).is_ok());
    }

    #[tokio::test]
    async fn sanitize_upstream_response_streams_binary_success() {
        let payload = b"binary-payload".to_vec();
        let http_resp = axum::http::Response::builder()
            .status(200)
            .header("content-type", "application/octet-stream")
            .header("content-length", payload.len().to_string())
            .header("server", "upstream")
            .header("x-request-id", "leak")
            .body(axum::body::Bytes::from(payload.clone()))
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp)
            .await
            .expect("binary success should stream, not buffer/reject");
        assert_eq!(result.status(), axum::http::StatusCode::OK);
        assert!(result.headers().get("server").is_none());
        assert!(result.headers().get("x-request-id").is_none());
        assert!(result.headers().get("content-length").is_none());
        let bytes = axum::body::to_bytes(result.into_body(), 1024)
            .await
            .expect("read relayed body");
        assert_eq!(bytes.as_ref(), payload.as_slice());
    }

    #[tokio::test]
    async fn sanitize_upstream_response_rejects_oversized_streaming_content_length() {
        // A successful non-JSON response that advertises a Content-Length over
        // the cap must be rejected up front on the relay path too (no
        // buffering), closing the streaming-branch DoS hole where a malicious
        // upstream could stream an unbounded body.
        let oversized = MAX_RESPONSE_BODY_BYTES + 1;
        let http_resp = axum::http::Response::builder()
            .status(200)
            .header("content-type", "application/octet-stream")
            .header("content-length", oversized.to_string())
            .body(axum::body::Bytes::from_static(b"{}"))
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp).await;
        assert!(
            result.is_err(),
            "streaming path must reject oversized Content-Length before relaying"
        );
    }

    #[tokio::test]
    async fn capped_byte_stream_aborts_over_cap() {
        // The relay cap must abort a chunked / unknown-length upstream as soon
        // as the running total exceeds the cap, surfacing a mid-stream error
        // and never forwarding more than `cap` bytes — so a malicious upstream
        // cannot pin unbounded proxy memory / egress on the streaming path.
        use futures_util::StreamExt;
        let chunk = axum::body::Bytes::from(vec![0u8; 1024 * 1024]);
        let cap = 5 * 1024 * 1024; // 5 MiB
        let input = futures_util::stream::iter(
            (0..60).map(move |_| Ok::<axum::body::Bytes, reqwest::Error>(chunk.clone())),
        );
        let mut capped = capped_byte_stream(input, cap);
        let mut total = 0usize;
        let mut saw_error = false;
        while let Some(item) = capped.next().await {
            match item {
                Ok(b) => total += b.len(),
                Err(_) => saw_error = true,
            }
        }
        assert!(
            saw_error,
            "stream must surface an error once the cap is exceeded"
        );
        assert!(
            total <= cap,
            "forwarded bytes ({total}) must never exceed the cap ({cap})"
        );
    }

    #[tokio::test]
    async fn sanitize_upstream_response_rejects_oversized_content_length() {
        let oversized = MAX_RESPONSE_BODY_BYTES + 1;
        let http_resp = axum::http::Response::builder()
            .status(200)
            .header("content-length", oversized.to_string())
            .body(axum::body::Bytes::from_static(b"{}"))
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp).await;
        assert!(
            result.is_err(),
            "response declaring Content-Length over the limit must error before buffering"
        );
    }

    #[test]
    fn body_limit_default() {
        assert_eq!(body_limit_bytes(None), 10 * 1024 * 1024);
    }

    #[test]
    fn body_limit_custom() {
        assert_eq!(body_limit_bytes(Some(50)), 50 * 1024 * 1024);
    }

    #[test]
    fn body_limit_zero_treated_as_one_mb_floor() {
        assert_eq!(
            body_limit_bytes(Some(0)),
            1024 * 1024,
            "0 must be clamped up to a 1 MB floor, not 0 (which rejects all bodies)"
        );
    }

    #[test]
    fn truncate_short_unchanged() {
        assert_eq!(truncate_error_message("hi", 500), "hi");
    }

    #[test]
    fn truncate_long_adds_ellipsis() {
        let long = "x".repeat(600);
        let t = truncate_error_message(&long, 500);
        assert_eq!(t.chars().count(), 501);
        assert!(t.ends_with('…'));
    }

    #[test]
    fn strip_leaking_headers_removes_server_and_request_id() {
        let mut h = HeaderMap::new();
        h.insert("server", "nginx".parse().unwrap());
        h.insert("x-request-id", "abc".parse().unwrap());
        h.insert(crate::pipeline::KEY_SPEND_HEADER, "99".parse().unwrap());
        h.insert(
            crate::pipeline::LITELLM_KEY_SPEND_HEADER,
            "88".parse().unwrap(),
        );
        h.insert(
            crate::pipeline::UPSTREAM_RESPONSE_COST_HEADER,
            "0.125".parse().unwrap(),
        );
        h.insert(
            crate::pipeline::UPSTREAM_TOTAL_TOKENS_HEADER,
            "12".parse().unwrap(),
        );
        h.insert("x-keep", "yes".parse().unwrap());
        strip_leaking_headers(&mut h);
        assert!(h.get("server").is_none());
        assert!(h.get("x-request-id").is_none());
        for name in crate::pipeline::KEY_SPEND_HEADERS {
            assert!(h.get(name).is_none());
        }
        assert_eq!(
            h.get(crate::pipeline::UPSTREAM_RESPONSE_COST_HEADER)
                .unwrap(),
            "0.125"
        );
        assert_eq!(
            h.get(crate::pipeline::UPSTREAM_TOTAL_TOKENS_HEADER)
                .unwrap(),
            "12"
        );
        assert!(h.get("x-keep").is_some());
    }

    #[test]
    fn secret_keys_detected() {
        assert!(is_secret_key("authorization"));
        assert!(is_secret_key("Authorization"));
        assert!(is_secret_key("api_key"));
        assert!(is_secret_key("api-key"));
        assert!(is_secret_key("x-api-key"));
        assert!(is_secret_key("password"));
        assert!(is_secret_key("client_secret"));
        assert!(is_secret_key("access_token"));
        assert!(!is_secret_key("content-type"));
        assert!(!is_secret_key("accept"));
    }

    #[test]
    fn redact_secret_header() {
        assert_eq!(
            redact_header_value("authorization", "Bearer sk-x"),
            "[REDACTED]"
        );
        assert_eq!(redact_header_value("content-type", "json"), "json");
    }

    #[test]
    fn sanitize_url_strips_userinfo() {
        assert_eq!(
            sanitize_url("redis://user:pass@host:6379/0"),
            "redis://host:6379/0"
        );
    }

    #[test]
    fn sanitize_url_strips_password_only() {
        assert_eq!(
            sanitize_url("postgres://:secret@db.example.com:5432/app"),
            "postgres://db.example.com:5432/app"
        );
    }

    #[test]
    fn sanitize_url_preserves_url_without_userinfo() {
        assert_eq!(
            sanitize_url("wss://api.openai.com/v1/realtime?model=gpt-4o"),
            "wss://api.openai.com/v1/realtime?model=gpt-4o"
        );
    }

    #[test]
    fn sanitize_url_fails_closed_for_malformed_or_opaque_inputs() {
        for raw in [
            "not a url at all",
            "postgres://alice:password@[broken/db?api_key=hidden",
            "https://alice:password@host:bad-port/path?token=hidden",
            "alice:password@host/path?api_key=hidden",
            "https:/alice:password@/path?token=hidden",
        ] {
            assert_eq!(sanitize_url(raw), "[invalid URL redacted]");
            assert_eq!(sanitize_signed_url(raw), "[invalid URL redacted]");
        }
    }

    #[test]
    fn sanitize_url_redacts_encoded_query_secrets_and_preserves_endpoint_context() {
        let raw = "postgres://alice:password@db.example:5432/app?sslmode=require&PaSS%77ord=query-password&API%5FKEY=query-key&%61ccess_token=query-token&application_name=proxy#token=fragment-secret";
        let safe = sanitize_url(raw);
        assert_eq!(
            safe,
            "postgres://db.example:5432/app?sslmode=require&application_name=proxy"
        );
        for secret in [
            "alice",
            "password",
            "query-key",
            "query-token",
            "fragment-secret",
        ] {
            assert!(!safe.contains(secret));
        }
        assert_eq!(
            sanitize_url("https://host/path?model=gpt-4o;API%5FKEY=hidden&api-version=2"),
            "https://host/path?model=gpt-4o&api-version=2"
        );
    }

    #[test]
    fn sanitize_url_preserves_sqlite_file_paths() {
        for raw in [
            "sqlite:/var/lib/oxllm/data.db",
            "sqlite::memory:",
            "sqlite://relative@data.sqlite",
            "sqlite:/var/password=data.sqlite",
        ] {
            assert_eq!(sanitize_url(raw), raw);
        }
        assert_eq!(
            sanitize_url("sqlite:/data.sqlite?mode=rwc&PaSS%77ord=hidden"),
            "sqlite:/data.sqlite?mode=rwc"
        );
    }

    #[test]
    fn sanitize_url_hides_slack_capability_paths() {
        let safe = sanitize_url("https://HOOKS.SLACK.COM/services/team/channel/live-capability");
        assert_eq!(safe, "https://hooks.slack.com/***");
        assert!(!safe.contains("live-capability"));
    }

    #[test]
    fn sanitize_signed_url_drops_aws_security_token() {
        // A presigned Bedrock WS URL carrying a real-shaped STS session token.
        // STS tokens start with "IQoJ" — the redactor must remove the whole
        // `X-Amz-Security-Token` pair so the live credential is never logged.
        let url = "wss://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-sonic-v1:0/invoke-with-response-stream-websocket?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIATEST/20260731/us-east-1/bedrock/aws4_request&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAoaCXVzLWVhc3QtMSJGMEQ&X-Amz-Signature=deadbeef&X-Amz-Date=20260731T000000Z";
        let redacted = sanitize_signed_url(url);
        assert!(
            !redacted.contains("X-Amz-Security-Token"),
            "session token pair must be dropped: {redacted}"
        );
        assert!(
            !redacted.contains("IQoJ"),
            "STS token prefix must not survive redaction: {redacted}"
        );
        assert!(
            !redacted.contains("X-Amz-Signature"),
            "signature pair must be dropped: {redacted}"
        );
        assert!(
            !redacted.contains("X-Amz-Credential"),
            "credential pair must be dropped: {redacted}"
        );
    }

    #[test]
    fn sanitize_signed_url_preserves_non_aws_query_pairs() {
        let url = "wss://host/path?model=gpt-4o&X-Amz-Signature=abc&keep=1";
        let redacted = sanitize_signed_url(url);
        assert!(redacted.contains("model=gpt-4o"));
        assert!(redacted.contains("keep=1"));
        assert!(!redacted.contains("X-Amz-"));
    }

    #[test]
    fn sanitize_signed_url_strips_userinfo_too() {
        let url = "wss://user:pass@host/path?X-Amz-Signature=abc";
        let redacted = sanitize_signed_url(url);
        assert!(!redacted.contains("user:pass"));
        assert!(!redacted.contains("X-Amz-"));
    }

    #[test]
    fn sanitize_signed_url_removes_mixed_case_and_encoded_signing_material() {
        let raw = "wss://alice:password@host/path?model=gpt-4o&x-aMz-SiGnAtUrE=signature&%58-Amz-Credential=credential&x-amz-security-token=temporary&AWSAccessKeyId=access-id&Signature=legacy&sig=azure&keep=1";
        assert_eq!(
            sanitize_signed_url(raw),
            "wss://host/path?model=gpt-4o&keep=1"
        );
    }

    #[test]
    fn database_error_display_and_debug_chains_are_log_safe() {
        let error = anyhow::anyhow!(
            "driver rejected PostGreSQL://alice:first-secret@db/app?PaSS%77ord=query-secret"
        )
        .context("retry mysql://bob:second-secret@backup/app?api%5Fkey=key-secret");
        for raw in [format!("{error:#}"), format!("{error:?}")] {
            let safe = oxllm_core::redaction::redact_connection_text(&raw);
            for secret in [
                "alice",
                "first-secret",
                "query-secret",
                "bob",
                "second-secret",
                "key-secret",
            ] {
                assert!(!safe.contains(secret), "leaked {secret}: {safe}");
            }
            assert!(safe.contains("db/app"));
            assert!(safe.contains("backup/app"));
        }
    }

    #[test]
    fn client_ip_prefers_xff_when_peer_trusted() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "1.2.3.4, 5.6.7.8".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.1:1234".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "5.6.7.8");
    }

    #[test]
    fn client_ip_uses_rightmost_xff_entry() {
        // The leftmost XFF entry is client-controlled (a client can prepend
        // any value), so it must NOT be used to resolve the client IP —
        // otherwise an attacker could spoof it to bypass the IP allowlist and
        // the auth-brake. The rightmost entry is the one appended by the
        // trusted proxy itself and is therefore the trustworthy value.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "spoofed-ip, 5.6.7.8".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.1:1234".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "5.6.7.8",
            "must use the rightmost (proxy-appended) XFF entry, not the spoofable leftmost"
        );
    }

    #[test]
    fn client_ip_falls_back_when_xff_entry_is_invalid() {
        // An unparseable XFF entry must not be trusted; fall back to the peer
        // address rather than forwarding garbage to the allowlist/brake.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "not-an-ip".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.9:4321".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "10.0.0.9",
            "invalid XFF entry must fall back to the peer address"
        );
    }

    #[test]
    fn client_ip_falls_back_to_x_real_ip_when_peer_trusted() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-real-ip", "9.9.9.9".parse().unwrap());
        let peer: std::net::SocketAddr = "10.1.2.3:5555".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "9.9.9.9");
    }

    #[test]
    fn client_ip_rejects_non_ip_x_real_ip_and_falls_back_to_peer() {
        // A non-trusted hop may forward rather than overwrite `X-Real-IP`, so
        // a client can inject an arbitrary value. A garbage value must NOT be
        // trusted: it would key distinct auth-brake entries and so let an
        // attacker evade per-IP blocking by varying it per request. Fall back
        // to the peer address (mirroring the XFF branch's IP-shape check).
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-real-ip", "garbage-not-an-ip".parse().unwrap());
        let peer: std::net::SocketAddr = "10.1.2.3:5555".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "10.1.2.3",
            "a non-IP X-Real-IP value must fall back to the peer, not be trusted"
        );
    }

    #[test]
    fn client_ip_x_real_ip_chosen_over_peer_when_valid_ip() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-real-ip", "4.4.4.4".parse().unwrap());
        let peer: std::net::SocketAddr = "10.9.9.9:1".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "4.4.4.4");
    }

    #[test]
    fn client_ip_x_real_ip_skips_trusted_proxy_value() {
        // A trusted proxy that forwards (rather than overwrites) `X-Real-IP`
        // with another trusted-proxy IP must not be attributed as the client.
        // Mirror the XFF walk's trust-skipping logic: a trusted-proxy value
        // would collapse every client behind it into one auth-brake bucket
        // and could satisfy an IP allowlist keyed on the proxy range. Fall
        // through to the peer address instead.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-real-ip", "10.0.0.9".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "10.0.0.5",
            "an X-Real-IP that is itself a trusted proxy must fall through to the peer"
        );
    }

    #[test]
    fn client_ip_x_real_ip_uses_non_trusted_public_value() {
        // A valid public X-Real-IP (not a trusted proxy) is still honored:
        // only trusted-proxy values are skipped.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-real-ip", "203.0.113.10".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "203.0.113.10");
    }

    #[test]
    fn client_ip_ignores_xff_when_peer_untrusted() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "1.2.3.4".parse().unwrap());
        h.insert("x-real-ip", "9.9.9.9".parse().unwrap());
        let peer: std::net::SocketAddr = "203.0.113.5:1234".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "203.0.113.5");
    }

    #[test]
    fn client_ip_ignores_xff_when_no_trusted_proxies_configured() {
        let trusted: Vec<ipnet::IpNet> = vec![];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "1.2.3.4".parse().unwrap());
        h.insert("x-real-ip", "9.9.9.9".parse().unwrap());
        let peer: std::net::SocketAddr = "203.0.113.5:1234".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "203.0.113.5",
            "with no trusted proxies, peer IP wins over forwarded headers"
        );
    }

    #[test]
    fn client_ip_unknown_when_no_peer_and_no_trusted_proxies() {
        let trusted: Vec<ipnet::IpNet> = vec![];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "1.2.3.4".parse().unwrap());
        assert_eq!(client_ip(&h, None, &trusted), "unknown");
    }

    #[test]
    fn client_ip_uses_peer_when_trusted_but_no_forwarded_headers() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let h = HeaderMap::new();
        let peer: std::net::SocketAddr = "10.5.5.5:9876".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "10.5.5.5");
    }

    #[test]
    fn client_ip_supports_ipv6_trusted_cidr() {
        let trusted: Vec<ipnet::IpNet> = vec!["fd00::/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "1.2.3.4".parse().unwrap());
        let peer: std::net::SocketAddr = "[fd00::1]:443".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "1.2.3.4");
    }

    // ---- auth_brake_ip: skip un-attributable / shared buckets ---------------

    #[test]
    fn auth_brake_ip_none_when_no_peer() {
        let trusted: Vec<ipnet::IpNet> = vec![];
        let h = HeaderMap::new();
        assert!(auth_brake_ip(&h, None, &trusted).is_none());
    }

    #[test]
    fn auth_brake_ip_skips_untrusted_private_peer() {
        // No trusted_proxies configured + a private-range peer (a reverse
        // proxy / LB in front): keying the brake on the proxy's address would
        // collapse every client behind it into one bucket. Must be skipped.
        let trusted: Vec<ipnet::IpNet> = vec![];
        for addr in [
            "10.0.0.5:8080",
            "192.168.1.10:443",
            "172.20.5.5:443",
            "100.64.0.1:443",
            "169.254.10.10:443",
            "127.0.0.1:8080",
        ] {
            let peer: std::net::SocketAddr = addr.parse().unwrap();
            let mut h = HeaderMap::new();
            h.insert("x-forwarded-for", "198.51.100.1".parse().unwrap());
            assert!(
                auth_brake_ip(&h, Some(peer), &trusted).is_none(),
                "untrusted private-range peer {addr} must skip brake enforcement"
            );
        }
    }

    #[test]
    fn auth_brake_ip_attributes_public_peer() {
        let trusted: Vec<ipnet::IpNet> = vec![];
        let h = HeaderMap::new();
        let peer: std::net::SocketAddr = "8.8.8.8:4321".parse().unwrap();
        assert_eq!(
            auth_brake_ip(&h, Some(peer), &trusted).as_deref(),
            Some("8.8.8.8")
        );
    }

    #[test]
    fn auth_brake_ip_uses_xff_when_peer_is_trusted_proxy() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "198.51.100.1".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.5:8080".parse().unwrap();
        assert_eq!(
            auth_brake_ip(&h, Some(peer), &trusted).as_deref(),
            Some("198.51.100.1"),
            "a trusted proxy's XFF client must be attributable"
        );
    }

    #[test]
    fn auth_brake_ip_skips_when_trusted_peer_sends_no_xff() {
        // A trusted L4 / TCP proxy cannot add X-Forwarded-For; `client_ip`
        // falls back to the proxy's own address, which every client behind it
        // shares. Keying the brake on it would let one failing client block
        // them all — the exact self-DoS this function refuses — so
        // enforcement must be skipped.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let h = HeaderMap::new();
        let peer: std::net::SocketAddr = "10.0.0.5:8080".parse().unwrap();
        assert!(
            auth_brake_ip(&h, Some(peer), &trusted).is_none(),
            "trusted peer without X-Forwarded-For must skip brake enforcement"
        );
    }

    #[test]
    fn auth_brake_ip_skips_when_xff_chain_is_all_trusted_hops() {
        // Every XFF hop is itself a trusted proxy: no client can be
        // attributed, and the fallback (the peer) is shared — skip.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "10.0.0.9, 10.0.0.10".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert!(
            auth_brake_ip(&h, Some(peer), &trusted).is_none(),
            "an all-trusted-hop XFF chain must skip brake enforcement"
        );
    }

    #[test]
    fn auth_brake_ip_skips_when_only_garbage_forwarded_headers() {
        // Garbage XFF plus a garbage X-Real-IP: neither yields an attributable
        // client, and the peer fallback is the shared proxy address — skip.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "not-an-ip".parse().unwrap());
        h.insert("x-real-ip", "garbage-not-an-ip".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert!(
            auth_brake_ip(&h, Some(peer), &trusted).is_none(),
            "garbage-only forwarded headers must skip brake enforcement"
        );
    }

    #[test]
    fn auth_brake_ip_skips_private_peer_even_with_xff_when_untrusted() {
        // The peer is private and NOT a configured trusted proxy: the XFF
        // header is client-controllable and must not be honoured; the private
        // peer would be a shared bucket, so enforcement is skipped.
        let trusted: Vec<ipnet::IpNet> = vec!["203.0.113.0/24".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "198.51.100.1".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.5:8080".parse().unwrap();
        assert!(auth_brake_ip(&h, Some(peer), &trusted).is_none());
    }

    #[test]
    fn auth_brake_ip_attributes_untrusted_public_peer_ignoring_xff() {
        // Public peer, no matching trusted proxy: XFF is ignored, the public
        // peer itself is the attributable client.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "8.8.8.8".parse().unwrap());
        let peer: std::net::SocketAddr = "1.1.1.1:1234".parse().unwrap();
        assert_eq!(
            auth_brake_ip(&h, Some(peer), &trusted).as_deref(),
            Some("1.1.1.1")
        );
    }

    #[test]
    fn json_depth_shallow_ok() {
        let v = serde_json::json!({"a": [1, 2, {"b": "c"}]});
        assert!(check_json_depth(&v, 128).is_ok());
    }

    #[test]
    fn json_depth_scalar_ok() {
        let v = serde_json::json!(42);
        assert!(check_json_depth(&v, 0).is_ok());
    }

    #[test]
    fn json_depth_rejects_deeply_nested() {
        let mut v = serde_json::Value::Null;
        for _ in 0..200 {
            v = serde_json::Value::Array(vec![v]);
        }
        let err = check_json_depth(&v, 128).expect_err("should reject depth > 128");
        let resp = err.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn json_depth_exactly_at_limit_ok() {
        let mut v = serde_json::Value::Null;
        for _ in 0..128 {
            v = serde_json::Value::Array(vec![v]);
        }
        assert!(check_json_depth(&v, 128).is_ok());
    }

    #[test]
    fn json_depth_one_over_limit_rejects() {
        let mut v = serde_json::Value::Null;
        for _ in 0..129 {
            v = serde_json::Value::Array(vec![v]);
        }
        assert!(check_json_depth(&v, 128).is_err());
    }

    #[tokio::test]
    async fn auth_brake_blocks_after_threshold() {
        let brake = AuthBrake::new(3, 60, 300);
        assert!(!brake.is_blocked("1.2.3.4").await);
        brake.record_failure("1.2.3.4").await;
        brake.record_failure("1.2.3.4").await;
        assert!(
            !brake.is_blocked("1.2.3.4").await,
            "2 fails < 3 should not block"
        );
        brake.record_failure("1.2.3.4").await;
        assert!(brake.is_blocked("1.2.3.4").await, "3 fails should block");
    }

    #[tokio::test]
    async fn auth_brake_isolates_ips() {
        let brake = AuthBrake::new(2, 60, 300);
        brake.record_failure("1.1.1.1").await;
        brake.record_failure("1.1.1.1").await;
        assert!(brake.is_blocked("1.1.1.1").await);
        assert!(!brake.is_blocked("2.2.2.2").await);
    }

    #[tokio::test]
    async fn auth_brake_block_persists_beyond_failure_window() {
        let brake = AuthBrake::new(2, 1, 300);
        brake.record_failure("9.9.9.9").await;
        brake.record_failure("9.9.9.9").await;
        assert!(brake.is_blocked("9.9.9.9").await);
        std::thread::sleep(std::time::Duration::from_millis(1_300));
        assert!(
            brake.is_blocked("9.9.9.9").await,
            "block must outlive the 1s failure window (block_secs=300)"
        );
    }

    #[tokio::test]
    async fn auth_brake_unblocks_after_block_secs() {
        let brake = AuthBrake::new(1, 60, 1);
        brake.record_failure("8.8.8.8").await;
        assert!(brake.is_blocked("8.8.8.8").await);
        std::thread::sleep(std::time::Duration::from_millis(1_300));
        assert!(
            !brake.is_blocked("8.8.8.8").await,
            "IP must be unblocked after block_secs elapses"
        );
    }

    #[tokio::test]
    async fn auth_brake_reblock_uses_block_secs_window() {
        let brake = AuthBrake::new(1, 60, 1);
        brake.record_failure("7.7.7.7").await;
        assert!(brake.is_blocked("7.7.7.7").await);
        std::thread::sleep(std::time::Duration::from_millis(1_300));
        assert!(!brake.is_blocked("7.7.7.7").await);
        brake.record_failure("7.7.7.7").await;
        assert!(brake.is_blocked("7.7.7.7").await, "fresh failures re-block");
    }

    /// L10: `is_blocked` must NOT create map entries for IPs that are merely
    /// being checked. The previous `entry().or_insert(...)` grew the map by one
    /// row per distinct probing IP — a cheap unbounded memory amplifier. After
    /// the fix, an unknown IP returns `false` read-only and leaves `tracked_ips`
    /// unchanged; entries are created only by `record_failure`.
    #[tokio::test]
    async fn is_blocked_does_not_create_entry_for_unknown_ip() {
        let brake = AuthBrake::new(3, 60, 60);
        assert_eq!(brake.tracked_ips(), 0, "no entries to start");

        // Checking several never-seen IPs must not create any entries.
        assert!(!brake.is_blocked("203.0.113.1").await);
        assert!(!brake.is_blocked("203.0.113.2").await);
        assert!(!brake.is_blocked("203.0.113.3").await);
        assert_eq!(
            brake.tracked_ips(),
            0,
            "status checks must not create entries; only record_failure should"
        );

        // A real failure still creates an entry (positive-path sanity check).
        brake.record_failure("203.0.113.1").await;
        assert_eq!(brake.tracked_ips(), 1);
        assert!(
            !brake.is_blocked("203.0.113.1").await,
            "one failure is below threshold"
        );
    }

    #[tokio::test]
    async fn auth_brake_sweep_evicts_expired_entries() {
        let brake = AuthBrake::new(3, 1, 1);
        brake.record_failure("1.1.1.1").await;
        brake.record_failure("2.2.2.2").await;
        assert_eq!(brake.tracked_ips(), 2, "two distinct IPs recorded");

        brake.sweep_now();
        assert_eq!(
            brake.tracked_ips(),
            2,
            "entries within the failure window must not be evicted"
        );

        std::thread::sleep(std::time::Duration::from_millis(1_300));
        brake.sweep_now();
        assert_eq!(
            brake.tracked_ips(),
            0,
            "expired entries must be evicted to prevent unbounded growth"
        );
    }

    #[tokio::test]
    async fn auth_brake_sweep_keeps_active_block() {
        let brake = AuthBrake::new(1, 1, 300);
        brake.record_failure("9.9.9.9").await;
        assert!(brake.is_blocked("9.9.9.9").await);

        std::thread::sleep(std::time::Duration::from_millis(1_300));
        brake.sweep_now();
        assert_eq!(
            brake.tracked_ips(),
            1,
            "actively-blocked IP must survive the sweep"
        );
        assert!(brake.is_blocked("9.9.9.9").await);
    }

    #[tokio::test]
    async fn auth_brake_sweep_triggers_periodically_from_is_blocked() {
        let brake = AuthBrake::new(1, 1, 1);
        for i in 0..500u32 {
            brake.record_failure(&format!("10.0.{i}.{i}")).await;
        }
        assert_eq!(brake.tracked_ips(), 500);

        std::thread::sleep(std::time::Duration::from_millis(1_300));

        // L10: `is_blocked` no longer creates map entries, so to verify a
        // still-fresh entry survives the sweep we must create it via
        // `record_failure` (the only entry-creating path after the fix). The
        // 500 expired entries above were recorded >1s ago and must be evicted.
        brake.record_failure("77.77.77.77").await;

        for _ in 0..600 {
            let _ = brake.is_blocked("77.77.77.77").await;
        }
        assert_eq!(
            brake.tracked_ips(),
            1,
            "periodic sweep from is_blocked must reclaim expired entries; \
             only the still-fresh dummy IP should remain"
        );
    }

    fn distributed_brake(max_fails: u32, window_secs: u64, block_secs: u64) -> AuthBrake {
        let cache: std::sync::Arc<dyn oxllm_core::Cache> =
            std::sync::Arc::new(crate::cache::InMemoryCache::default());
        let coord = crate::coordination::Coordinator::new(cache).with_distributed(true);
        AuthBrake::new(max_fails, window_secs, block_secs).with_coordinator(Some(coord))
    }

    #[tokio::test]
    async fn distributed_auth_brake_local_fallback_matches_local_behavior() {
        let cache: std::sync::Arc<dyn oxllm_core::Cache> =
            std::sync::Arc::new(crate::cache::InMemoryCache::default());
        let coord = crate::coordination::Coordinator::new(cache);
        let brake = AuthBrake::new(3, 60, 300).with_coordinator(Some(coord));
        assert!(!brake.is_blocked("1.2.3.4").await);
        brake.record_failure("1.2.3.4").await;
        brake.record_failure("1.2.3.4").await;
        assert!(!brake.is_blocked("1.2.3.4").await);
        brake.record_failure("1.2.3.4").await;
        assert!(brake.is_blocked("1.2.3.4").await);
    }

    #[tokio::test]
    async fn distributed_auth_brake_blocks_after_threshold() {
        let brake = distributed_brake(3, 60, 300);
        assert!(!brake.is_blocked("1.2.3.4").await);
        brake.record_failure("1.2.3.4").await;
        brake.record_failure("1.2.3.4").await;
        assert!(
            !brake.is_blocked("1.2.3.4").await,
            "2 fails < 3 should not block in distributed mode"
        );
        brake.record_failure("1.2.3.4").await;
        assert!(
            brake.is_blocked("1.2.3.4").await,
            "3 fails should block in distributed mode"
        );
    }

    #[tokio::test]
    async fn distributed_auth_brake_isolates_ips() {
        let brake = distributed_brake(2, 60, 300);
        brake.record_failure("1.1.1.1").await;
        brake.record_failure("1.1.1.1").await;
        assert!(brake.is_blocked("1.1.1.1").await);
        assert!(!brake.is_blocked("2.2.2.2").await);
    }

    #[tokio::test]
    async fn distributed_auth_brake_block_expires_after_ttl() {
        let brake = distributed_brake(1, 60, 1);
        brake.record_failure("8.8.8.8").await;
        assert!(brake.is_blocked("8.8.8.8").await);
        tokio::time::sleep(std::time::Duration::from_millis(1_300)).await;
        assert!(
            !brake.is_blocked("8.8.8.8").await,
            "distributed block must expire after block_secs TTL"
        );
    }

    #[tokio::test]
    async fn distributed_auth_brake_shares_state_via_cache() {
        let cache: std::sync::Arc<dyn oxllm_core::Cache> =
            std::sync::Arc::new(crate::cache::InMemoryCache::default());
        let coord = crate::coordination::Coordinator::new(cache.clone()).with_distributed(true);
        let brake_a = AuthBrake::new(2, 60, 300).with_coordinator(Some(coord.clone()));
        let brake_b = AuthBrake::new(2, 60, 300).with_coordinator(Some(coord));

        brake_a.record_failure("5.5.5.5").await;
        brake_b.record_failure("5.5.5.5").await;
        assert!(
            brake_a.is_blocked("5.5.5.5").await,
            "failures recorded by separate brake instances over the same cache \
             must accumulate fleet-wide"
        );
    }

    #[tokio::test]
    async fn smuggling_guard_rejects_both_headers() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let app = Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(smuggling_guard));

        let mut req = Request::builder()
            .method("GET")
            .uri("/")
            .body(Body::empty())
            .unwrap();
        req.headers_mut()
            .insert("content-length", "10".parse().unwrap());
        req.headers_mut()
            .insert("transfer-encoding", "chunked".parse().unwrap());

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn smuggling_guard_allows_single_header() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let app = Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(smuggling_guard));

        let mut req = Request::builder()
            .method("GET")
            .uri("/")
            .body(Body::empty())
            .unwrap();
        req.headers_mut()
            .insert("content-length", "10".parse().unwrap());

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
    }

    #[test]
    fn ip_allowed_matches_inside_cidr() {
        let allowed: Vec<ipnet::IpNet> =
            parse_cidr_list(&["10.0.0.0/8".to_string(), "192.168.1.0/24".to_string()]);
        assert!(ip_allowed("10.5.5.5", &allowed));
        assert!(ip_allowed("192.168.1.50", &allowed));
    }

    #[test]
    fn ip_allowed_rejects_outside_cidr() {
        let allowed: Vec<ipnet::IpNet> = parse_cidr_list(&["10.0.0.0/8".to_string()]);
        assert!(!ip_allowed("203.0.113.5", &allowed));
        assert!(!ip_allowed("192.168.1.1", &allowed));
    }

    #[test]
    fn ip_allowed_rejects_unparseable_ip() {
        let allowed: Vec<ipnet::IpNet> = parse_cidr_list(&["10.0.0.0/8".to_string()]);
        assert!(!ip_allowed("not-an-ip", &allowed));
        assert!(!ip_allowed("unknown", &allowed));
    }

    #[test]
    fn parse_cidr_list_skips_invalid_entries() {
        let parsed = parse_cidr_list(&[
            "10.0.0.0/8".to_string(),
            "not-a-cidr".to_string(),
            "192.168.0.0/16".to_string(),
        ]);
        assert_eq!(parsed.len(), 2, "invalid entry should be skipped");
    }

    #[test]
    fn ip_allowed_supports_ipv6() {
        let allowed: Vec<ipnet::IpNet> = parse_cidr_list(&["fd00::/8".to_string()]);
        assert!(ip_allowed("fd00::1", &allowed));
        assert!(!ip_allowed("2001:4860:4860::8888", &allowed));
    }

    #[test]
    fn ip_allowed_normalizes_ipv4_mapped_v6() {
        // A dual-stack listener presents IPv4 peers as `::ffff:a.b.c.d`; the
        // allowlist must match them against IPv4 CIDRs so operators do not
        // need to duplicate every range in v6 form.
        let allowed: Vec<ipnet::IpNet> = parse_cidr_list(&["10.0.0.0/8".to_string()]);
        assert!(
            ip_allowed("::ffff:10.0.0.5", &allowed),
            "an IPv4-mapped allowlisted peer must pass"
        );
        assert!(
            !ip_allowed("::ffff:192.168.1.1", &allowed),
            "an IPv4-mapped non-allowlisted peer must be rejected"
        );
    }

    #[test]
    fn ip_allowed_empty_list_denies_all() {
        // IP-filter parity: an empty list denies every
        // address — an explicit `allowed_ips: []` must never read as
        // "allow everyone".
        let allowed: Vec<ipnet::IpNet> = vec![];
        assert!(!ip_allowed("8.8.8.8", &allowed));
        assert!(!ip_allowed("10.0.0.1", &allowed));
        assert!(!ip_allowed("unknown", &allowed));
    }

    #[test]
    fn check_ip_allowed_rejects_private_ranges() {
        // Every entry in PRIVATE_CIDRS must trip the runtime check.
        assert!(!check_ip_allowed("10.0.0.1".parse().unwrap()));
        assert!(!check_ip_allowed("127.0.0.1".parse().unwrap()));
        assert!(!check_ip_allowed("172.16.5.5".parse().unwrap()));
        assert!(!check_ip_allowed("192.168.1.1".parse().unwrap()));
        assert!(!check_ip_allowed("169.254.169.254".parse().unwrap()));
        assert!(!check_ip_allowed("100.64.0.1".parse().unwrap()));
        assert!(!check_ip_allowed("0.0.0.0".parse().unwrap()));
        assert!(!check_ip_allowed("::1".parse().unwrap()));
        assert!(!check_ip_allowed("fc00::1".parse().unwrap()));
        assert!(!check_ip_allowed("fe80::1".parse().unwrap()));
    }

    #[test]
    fn check_ip_allowed_accepts_public_addresses() {
        assert!(check_ip_allowed("8.8.8.8".parse().unwrap()));
        assert!(check_ip_allowed("1.1.1.1".parse().unwrap()));
        assert!(check_ip_allowed("2001:4860:4860::8888".parse().unwrap()));
    }

    #[test]
    fn check_ip_allowed_normalizes_ipv4_mapped_v6() {
        // An IPv4-mapped IPv6 representation of a private IPv4 address must
        // be detected as private (DNS rebinding / NAT64 bypass).
        let mapped: IpAddr = "::ffff:10.0.0.1".parse().unwrap();
        assert!(!check_ip_allowed(mapped));
        let mapped_pub: IpAddr = "::ffff:8.8.8.8".parse().unwrap();
        assert!(check_ip_allowed(mapped_pub));
    }

    #[test]
    fn private_cidr_nets_cached_and_cover_all_constants() {
        // The parsed cache must contain exactly one IpNet per parseable
        // constant in PRIVATE_CIDRS + CLOUD_METADATA_EXCEPTIONS (so the hot
        // path never re-parses them).
        assert_eq!(
            PRIVATE_CIDR_NETS.len(),
            PRIVATE_CIDRS.len() + CLOUD_METADATA_EXCEPTIONS.len(),
            "every PRIVATE_CIDRS + CLOUD_METADATA_EXCEPTIONS constant must parse \
             (a mismatch means a bad constant)"
        );
        // Every private literal must still be rejected via the cached path.
        for cidr in PRIVATE_CIDR_NETS.iter() {
            assert!(cidr.contains(&cidr.network()));
        }
        // Sanity: the same static slice is handed back on every call
        // (structurally a OnceLock-style cache reuses one allocation).
        let ptr1 = PRIVATE_CIDR_NETS.as_ptr();
        let ptr2 = PRIVATE_CIDR_NETS.as_ptr();
        assert_eq!(ptr1, ptr2, "parsed CIDRs must be reused, not recomputed");
    }

    #[test]
    fn is_private_host_recognizes_literal_ips() {
        assert!(is_private_host("127.0.0.1"));
        assert!(is_private_host("10.1.2.3"));
        assert!(is_private_host("192.168.0.1"));
        assert!(!is_private_host("8.8.8.8"));
        assert!(is_private_host("[::1]"));
        // Bracketed IPv6 stripping handled by trim.
        assert!(is_private_host("::1"));
    }

    #[test]
    fn validate_upstream_ip_rejects_private_literal() {
        let err = validate_upstream_ip("http://127.0.0.1/admin").expect_err("private");
        assert!(err.contains("private"), "got: {err}");
        let err = validate_upstream_ip("http://10.0.0.1/").expect_err("private");
        assert!(err.contains("private"));
    }

    #[test]
    fn validate_upstream_ip_accepts_public_literal() {
        validate_upstream_ip("https://api.openai.com/v1/chat/completions")
            .expect("public IP literals must pass");
        validate_upstream_ip("https://1.1.1.1/").expect("public IP literals must pass");
    }

    #[test]
    fn validate_upstream_ip_rejects_bad_urls() {
        assert!(validate_upstream_ip("not a url").is_err());
        assert!(validate_upstream_ip("file:///etc/passwd").is_err());
    }

    #[tokio::test]
    async fn ssrf_resolver_rejects_private_literal_host() {
        use reqwest::dns::Resolve;
        // C05/HIGH: the resolver must drop any resolved address in a private
        // range at connection time. A literal IP host skips DNS, so this
        // deterministically exercises the private-range filter without network
        // access. The cloud-metadata IP is the canonical SSRF target.
        let resolver = SsrfResolver::new();
        for host in ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1"] {
            let name: reqwest::dns::Name = host.parse().unwrap();
            let result = resolver.resolve(name).await;
            assert!(
                result.is_err(),
                "resolver must reject private host '{host}' at connection time"
            );
        }
    }

    #[tokio::test]
    async fn ssrf_resolver_allows_public_literal_host() {
        use reqwest::dns::Resolve;
        let resolver = SsrfResolver::new();
        let name: reqwest::dns::Name = "8.8.8.8".parse().unwrap();
        let addrs = resolver
            .resolve(name)
            .await
            .expect("public literal host must resolve");
        let collected: Vec<std::net::SocketAddr> = addrs.collect();
        assert!(
            collected.iter().any(|a| a.ip().to_string() == "8.8.8.8"),
            "public host must be returned by the resolver, got {collected:?}"
        );
    }

    #[test]
    fn is_private_host_cached_recognizes_literal_private_ips() {
        // Literal IPs skip DNS and the cache; they're evaluated directly.
        assert!(is_private_host_cached("127.0.0.1"));
        assert!(is_private_host_cached("10.1.2.3"));
        assert!(is_private_host_cached("192.168.0.1"));
        assert!(!is_private_host_cached("8.8.8.8"));
    }

    #[test]
    fn is_private_host_cached_fails_open_on_cache_miss() {
        // H1: after moving DNS off the sync hot path, `is_private_host_cached`
        // is a cache-only lookup. A cache miss must NOT do a blocking DNS
        // call; it fails open (returns false). The async variant owns the
        // actual lookup.
        // Use a hostname that is guaranteed not to be in the cache (it
        // doesn't need to resolve — the sync function no longer tries).
        let miss = is_private_host_cached("never-cached-host-for-tests.example");
        assert!(
            !miss,
            "cache miss must fail open (allow) without a blocking DNS call"
        );
    }

    #[tokio::test]
    async fn is_private_host_async_recognizes_literal_private_ips() {
        // Literal IPs skip DNS and the cache; they're evaluated directly.
        assert!(is_private_host_async("127.0.0.1").await);
        assert!(is_private_host_async("10.1.2.3").await);
        assert!(is_private_host_async("192.168.0.1").await);
        assert!(!is_private_host_async("8.8.8.8").await);
    }

    #[tokio::test]
    async fn is_private_host_async_best_effort_for_unresolvable_host() {
        // An unresolvable hostname must NOT be reported as private
        // (best-effort: allow the request; the upstream call will fail
        // naturally). RFC 6761 reserves `.invalid` so it never resolves.
        let result = is_private_host_async("nonexistent-host-invalid-tld.invalid").await;
        assert!(
            !result,
            "unresolvable host must be allowed (best-effort), got blocked"
        );
    }

    #[tokio::test]
    async fn is_private_host_async_caches_result_for_subsequent_sync_lookup() {
        // H1: after an async lookup populates the cache, the synchronous
        // cache-only `is_private_host_cached` must observe the cached value
        // without doing its own DNS resolution. A `.invalid` host resolves
        // to "not private" (best-effort); the cached entry should reflect
        // that when read back synchronously.
        let host = "another-unresolvable-host-for-cache-test.invalid";
        assert!(!is_private_host_async(host).await);
        assert!(
            !is_private_host_cached(host),
            "sync lookup must read the value cached by the async lookup"
        );
    }

    #[test]
    fn dns_cache_sweep_evicts_stale_entries() {
        // Backdate several entries past DNS_CACHE_TTL, then sweep.
        let stale = match Instant::now().checked_sub(Duration::from_secs(60)) {
            Some(t) => t,
            None => return,
        };
        {
            let mut cache = dns_cache().lock();
            for i in 0..50 {
                cache
                    .entries
                    .insert(format!("stale-sweep-test-{i}.invalid"), (stale, false));
            }
        }
        {
            let mut cache = dns_cache().lock();
            dns_cache_sweep_locked(&mut cache);
            let remaining = cache
                .entries
                .keys()
                .filter(|k| k.starts_with("stale-sweep-test-"))
                .count();
            assert_eq!(remaining, 0, "stale entries must be evicted by the sweep");
        }
    }

    #[test]
    fn dns_cache_sweep_keeps_fresh_entries() {
        let fresh = Instant::now();
        {
            let mut cache = dns_cache().lock();
            for i in 0..20 {
                cache
                    .entries
                    .insert(format!("fresh-sweep-test-{i}.invalid"), (fresh, false));
            }
        }
        {
            let mut cache = dns_cache().lock();
            dns_cache_sweep_locked(&mut cache);
            let remaining = cache
                .entries
                .keys()
                .filter(|k| k.starts_with("fresh-sweep-test-"))
                .count();
            assert_eq!(remaining, 20, "fresh entries must survive the sweep");
        }
    }

    #[test]
    fn dns_cache_periodic_sweep_keeps_map_bounded() {
        // Distinct-host probing on the image-URL path must not grow
        // [`DNS_CACHE`] without bound. Simulate many inserts (each already
        // stale) driving the periodic sweep that runs every
        // [`DNS_CACHE_SWEEP_INTERVAL`] inserts, and assert the map stays
        // bounded.
        let stale = match Instant::now().checked_sub(Duration::from_secs(60)) {
            Some(t) => t,
            None => return,
        };
        for i in 0..(DNS_CACHE_SWEEP_INTERVAL * 3) as u32 {
            let key = format!("bounded-direct-test-{i}.invalid");
            let mut cache = dns_cache().lock();
            cache.entries.insert(key, (stale, false));
            cache.inserts_since_sweep = cache.inserts_since_sweep.saturating_add(1);
            if cache.inserts_since_sweep >= DNS_CACHE_SWEEP_INTERVAL {
                cache.inserts_since_sweep = 0;
                dns_cache_sweep_locked(&mut cache);
            }
        }
        let remaining = {
            let cache = dns_cache().lock();
            cache
                .entries
                .keys()
                .filter(|k| k.starts_with("bounded-direct-test-"))
                .count()
        };
        assert!(
            remaining < DNS_CACHE_SWEEP_INTERVAL as usize,
            "periodic sweep must keep the cache bounded; still had {remaining} stale entries"
        );
    }

    #[test]
    fn dns_cache_evict_oldest_removes_oldest_entry() {
        // The TTL sweep reclaims entries older than `DNS_CACHE_TTL`, but
        // every entry survives at least that long before it is sweepable, so
        // under sustained distinct-host probing the map grows to
        // `rate × DNS_CACHE_TTL` before reaching steady-state turnover. The
        // hard cap evicts the oldest entry by `cached_at` to make room for a
        // new host. Exercise the eviction helper on a LOCAL `DnsCacheInner`
        // (the helper takes `&mut DnsCacheInner`) so the assertion is not
        // perturbed by other tests mutating the shared global cache in
        // parallel, and because the 100k cap is impractical to fill here.
        let oldest = match Instant::now().checked_sub(Duration::from_secs(60)) {
            Some(t) => t,
            None => return,
        };
        let fresh = Instant::now();
        let mut inner = DnsCacheInner::default();
        inner
            .entries
            .insert("evict-fresh-1.invalid".to_string(), (fresh, false));
        inner
            .entries
            .insert("evict-oldest.invalid".to_string(), (oldest, false));
        inner
            .entries
            .insert("evict-fresh-2.invalid".to_string(), (fresh, false));
        dns_cache_evict_oldest_locked(&mut inner);
        assert_eq!(inner.entries.len(), 2, "exactly one entry must be evicted");
        assert!(
            !inner.entries.contains_key("evict-oldest.invalid"),
            "the entry with the oldest cached_at must be the eviction victim"
        );
        assert!(
            inner.entries.contains_key("evict-fresh-1.invalid")
                && inner.entries.contains_key("evict-fresh-2.invalid"),
            "the fresh entries must survive"
        );
    }

    #[test]
    fn dns_cache_evict_oldest_is_noop_on_empty_map() {
        // Eviction must be a no-op (not panic) when the map is empty.
        let mut inner = DnsCacheInner::default();
        dns_cache_evict_oldest_locked(&mut inner);
        assert!(inner.entries.is_empty());
    }

    #[tokio::test]
    async fn allowed_ips_guard_blocks_untrusted_peer() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let mut config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                allowed_ips: Some(vec!["10.0.0.0/8".to_string()]),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        config.oxllm_settings.drop_params = None;
        let shared = crate::app::build_shared_state(config).expect("state");

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                allowed_ips_guard,
            ));

        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::FORBIDDEN,
            "no peer info → 'unknown' IP is not in 10.0.0.0/8"
        );
    }

    // ---- C05 HIGH-1: buffered response path is bounded mid-stream ---------

    #[tokio::test]
    async fn sanitize_upstream_response_aborts_chunked_json_over_cap() {
        // A chunked (no Content-Length) JSON success body over the cap must be
        // rejected by `read_bounded_bytes` mid-stream — not buffered in full
        // and only then rejected (the old `response.bytes().await` path would
        // pin the entire body in proxy heap before the post-read check fired).
        let body =
            reqwest::Body::wrap_stream(futures_util::stream::iter((0..60u32).map(|_| {
                Ok::<_, std::io::Error>(axum::body::Bytes::from(vec![0u8; 1024 * 1024]))
            })));
        let http_resp = axum::http::Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(body)
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp).await;
        assert!(
            result.is_err(),
            "chunked JSON success body over the cap must be rejected, not buffered"
        );
    }

    #[tokio::test]
    async fn sanitize_upstream_response_truncates_chunked_error_over_error_cap() {
        // A chunked error body over the tighter 10 KB cap is truncated (not
        // rejected): the upstream status must survive even when the error
        // payload is huge, mirroring the ≤cap path's status preservation.
        let body = vec![b'x'; 32 * 1024];
        let http_resp = axum::http::Response::builder()
            .status(500)
            .header("content-type", "application/json")
            .body(axum::body::Bytes::from(body))
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp)
            .await
            .expect("oversized chunked error body must yield a truncated response, not an error");
        assert_eq!(
            result.status(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        let bytes = axum::body::to_bytes(result.into_body(), 1024 * 1024)
            .await
            .expect("read body");
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.starts_with('x'));
        assert!(
            text.chars().count() <= 501,
            "surfaced error body must stay truncated to 500 chars + ellipsis, got {}",
            text.chars().count()
        );
    }

    #[tokio::test]
    async fn sanitize_upstream_response_preserves_status_for_oversized_error_body() {
        // A 429 whose body exceeds MAX_ERROR_BODY_BYTES (Cloudflare challenge
        // pages and other SaaS HTML block pages routinely do) must surface the
        // upstream 429 with the first ~500 chars — NOT an oxllm 500 from the
        // mid-stream abort. The honestly-advertised Content-Length over the
        // cap must not be rejected up front on the error path either.
        let html: String = "<html>rate limited</html>".repeat(2048); // 48 KB
        let http_resp = axum::http::Response::builder()
            .status(429)
            .header("content-type", "text/html")
            .header("content-length", html.len().to_string())
            .body(axum::body::Bytes::from(html.clone()))
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp)
            .await
            .expect("oversized error body must truncate, not fail");
        assert_eq!(
            result.status(),
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            "the upstream status must survive the oversized body"
        );
        let bytes = axum::body::to_bytes(result.into_body(), 1024 * 1024)
            .await
            .expect("read body");
        let text = String::from_utf8_lossy(&bytes).to_string();
        assert!(
            text.starts_with(&html[..html.len().min(500)]),
            "the surfaced body must be the beginning of the upstream body"
        );
        assert_eq!(text.chars().count(), 501, "500 chars + ellipsis");
        assert!(text.ends_with('…'));
    }

    #[tokio::test]
    async fn sanitize_upstream_response_error_path_drops_content_encoding() {
        // The error body is decoded to text via from_utf8_lossy while reqwest
        // performs no auto-decompression, so a `content-encoding` header would
        // describe bytes the client no longer receives. It must be dropped.
        let payload = vec![0x1f, 0x8b, 0x08, 0x00, 0x01, 0x02, 0x03]; // gzip magic + junk
        let http_resp = axum::http::Response::builder()
            .status(502)
            .header("content-type", "application/json")
            .header("content-encoding", "gzip")
            .body(axum::body::Bytes::from(payload))
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp)
            .await
            .expect("error path must produce a response");
        assert_eq!(result.status(), axum::http::StatusCode::BAD_GATEWAY);
        assert!(
            result.headers().get("content-encoding").is_none(),
            "surfaced error body must not advertise a stale content-encoding"
        );
    }

    // ---- C05 MEDIUM: multi-tier trusted proxy chain ------------------------

    #[test]
    fn client_ip_multi_tier_trusted_chain_skips_inner_proxy() {
        // client → P1 (10.0.0.9) → P2 (192.168.0.5) → oxllm, both proxies
        // trusted. The rightmost XFF entry is P1, not the client; the walk
        // must skip P1 and return the real client.
        let trusted: Vec<ipnet::IpNet> = vec![
            "10.0.0.0/8".parse().unwrap(),
            "192.168.0.0/16".parse().unwrap(),
        ];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "8.8.8.8, 10.0.0.9".parse().unwrap());
        let peer: std::net::SocketAddr = "192.168.0.5:443".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "8.8.8.8",
            "must attribute the real client, not the inner trusted proxy"
        );
    }

    #[test]
    fn client_ip_all_trusted_hops_fall_back_to_peer() {
        // Every hop in the chain is a trusted proxy: there is no client to
        // attribute, so fall back to the peer address.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "10.0.0.9".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "10.0.0.5");
    }

    #[test]
    fn client_ip_multi_tier_skips_non_ip_and_trusted_hops() {
        // A garbage (spoofable) leftmost entry plus a trusted inner hop: the
        // walk skips both and lands on the real client.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert(
            "x-forwarded-for",
            "garbage, 10.0.0.9, 8.8.8.8".parse().unwrap(),
        );
        let peer: std::net::SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "8.8.8.8");
    }

    #[test]
    fn client_ip_parses_bracketed_ipv6_xff_entry() {
        // RFC 7239-style bracketed IPv6 hops must be attributed, not skipped:
        // a trusted proxy emitting bracketed IPv6 for its IPv6 clients would
        // otherwise collapse them all onto the peer bucket.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert(
            "x-forwarded-for",
            "[2001:db8::5], 10.0.0.9".parse().unwrap(),
        );
        let peer: std::net::SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "2001:db8::5",
            "bracketed IPv6 XFF entries must parse"
        );
    }

    #[test]
    fn client_ip_parses_bracketed_ipv6_xff_entry_with_port() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert(
            "x-forwarded-for",
            "[2001:db8::5]:443, 10.0.0.9".parse().unwrap(),
        );
        let peer: std::net::SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "2001:db8::5");
    }

    #[test]
    fn client_ip_parses_ipv4_port_xff_entry() {
        // Some proxies append `ip:port` per hop; the IPv4 form must parse.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "1.2.3.4:5678, 10.0.0.9".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert_eq!(client_ip(&h, Some(peer), &trusted), "1.2.3.4");
    }

    #[test]
    fn parse_xff_hop_still_skips_unparseable_entries() {
        // Tolerating brackets/ports must not start trusting garbage: anything
        // that still fails to parse is skipped (fail-safe).
        assert!(parse_xff_hop("garbage").is_none());
        assert!(parse_xff_hop("unknown").is_none());
        assert!(parse_xff_hop("[broken").is_none());
        assert!(parse_xff_hop("1.2.3.4.5:80").is_none());
        assert!(parse_xff_hop("2001:db8::5").is_some());
        assert!(parse_xff_hop("1.2.3.4").is_some());
        assert!(parse_xff_hop(" 1.2.3.4 ").is_some());
    }

    #[test]
    fn client_ip_honors_ipv4_mapped_peer_for_trusted_cidrs() {
        // On a dual-stack listener an IPv4 peer is presented as
        // `::ffff:a.b.c.d`; the trusted-proxy check must normalize so an IPv4
        // trusted CIDR matches and the XFF walk runs.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "198.51.100.7".parse().unwrap());
        let peer: std::net::SocketAddr = "[::ffff:10.0.0.5]:443".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "198.51.100.7",
            "an IPv4-mapped peer must match the IPv4 trusted-proxy CIDR"
        );
    }

    #[test]
    fn client_ip_normalizes_ipv4_mapped_peer_fallback() {
        // Without forwarded headers the peer itself is the client; the
        // mapped form must come back canonical (IPv4) so downstream CIDR
        // checks and map keys see one representation.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let h = HeaderMap::new();
        let peer: std::net::SocketAddr = "[::ffff:203.0.113.9]:443".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "203.0.113.9",
            "the mapped peer must be normalized to its IPv4 form"
        );
    }

    #[test]
    fn auth_brake_ip_attributes_ipv4_mapped_public_peer() {
        // A mapped public peer is attributable and not shared: the brake keys
        // on the normalized IPv4 form.
        let trusted: Vec<ipnet::IpNet> = vec![];
        let h = HeaderMap::new();
        let peer: std::net::SocketAddr = "[::ffff:8.8.8.8]:443".parse().unwrap();
        assert_eq!(
            auth_brake_ip(&h, Some(peer), &trusted).as_deref(),
            Some("8.8.8.8")
        );
    }

    #[test]
    fn auth_brake_ip_skips_ipv4_mapped_shared_peer() {
        // A mapped private peer (untrusted) is a shared address — skip, same
        // as the plain IPv4 form.
        let trusted: Vec<ipnet::IpNet> = vec![];
        let h = HeaderMap::new();
        let peer: std::net::SocketAddr = "[::ffff:10.0.0.5]:443".parse().unwrap();
        assert!(auth_brake_ip(&h, Some(peer), &trusted).is_none());
    }

    // ---- C05 MEDIUM/HIGH-3: distributed brake semantics --------------------

    #[tokio::test]
    async fn distributed_auth_brake_window_longer_than_block_no_instant_reblock() {
        // With a counting window (10s) longer than the block (1s), a post-block
        // failure must start from a clean window (local parity): the fails
        // counter is deleted when the block is set, so the first new failure
        // after unblocking cannot instantly re-block.
        let brake = distributed_brake(2, 10, 1);
        brake.record_failure("6.6.6.6").await;
        brake.record_failure("6.6.6.6").await;
        assert!(brake.is_blocked("6.6.6.6").await);
        tokio::time::sleep(Duration::from_millis(1_300)).await;
        assert!(
            !brake.is_blocked("6.6.6.6").await,
            "block must expire after block_secs"
        );
        brake.record_failure("6.6.6.6").await;
        assert!(
            !brake.is_blocked("6.6.6.6").await,
            "one fresh failure must NOT instantly re-block (window > block)"
        );
    }

    #[tokio::test]
    async fn distributed_auth_brake_concurrent_failures_bounded_at_threshold() {
        // The threshold crossing is atomic via `incr_if_below`: 64 concurrent
        // failures must never push the shared fails counter past `max_fails`
        // (the old incr + post-check would overshoot by the concurrency).
        let cache: std::sync::Arc<dyn oxllm_core::Cache> =
            std::sync::Arc::new(crate::cache::InMemoryCache::default());
        let coord = crate::coordination::Coordinator::new(cache.clone()).with_distributed(true);
        let brake = AuthBrake::new(5, 60, 300).with_coordinator(Some(coord));
        let mut handles = Vec::new();
        for _ in 0..64 {
            let brake = brake.clone();
            handles.push(tokio::spawn(async move {
                brake.record_failure("7.7.7.7").await;
            }));
        }
        for h in handles {
            h.await.expect("task");
        }
        if let Some(v) = cache
            .try_counter_value("authbrake:fails:7.7.7.7")
            .await
            .expect("cache read")
        {
            assert!(
                v <= 5.0,
                "fails counter must never overshoot max_fails, got {v}"
            );
        }
        assert!(
            brake.is_blocked("7.7.7.7").await,
            "threshold crossing must block"
        );
    }

    struct FailingCache;

    #[async_trait::async_trait]
    impl oxllm_core::Cache for FailingCache {
        async fn get(&self, _key: &str) -> Option<String> {
            None
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(
            &self,
            _key: &str,
            _by: f64,
            _ttl: Option<Duration>,
        ) -> Result<f64, oxllm_core::Error> {
            Err(oxllm_core::Error::Network("simulated outage".to_string()))
        }
        async fn try_counter_value(&self, _key: &str) -> Result<Option<f64>, oxllm_core::Error> {
            Err(oxllm_core::Error::Network("simulated outage".to_string()))
        }
    }

    #[tokio::test]
    async fn distributed_auth_brake_falls_back_to_local_on_cache_error() {
        // C05/HIGH-3: a backing-store outage must degrade to the local per-IP
        // map (counting + blocking), not silently disable enforcement.
        let coord = crate::coordination::Coordinator::new(std::sync::Arc::new(FailingCache))
            .with_distributed(true);
        let brake = AuthBrake::new(2, 60, 300).with_coordinator(Some(coord));
        brake.record_failure("1.2.3.4").await;
        brake.record_failure("1.2.3.4").await;
        assert!(
            brake.is_blocked("1.2.3.4").await,
            "must fall back to the local map and still block during an outage"
        );
        assert_eq!(brake.tracked_ips(), 1);
    }

    // ---- C05 MEDIUM: tracked-IP map hard cap -------------------------------

    #[tokio::test]
    async fn auth_brake_caps_tracked_ips() {
        let brake = AuthBrake::new(3, 60, 300).with_max_ips(4);
        for i in 0..20u32 {
            brake.record_failure(&format!("10.0.{i}.{i}")).await;
        }
        assert!(
            brake.tracked_ips() <= 4,
            "tracked-IP map must never grow past the cap (got {})",
            brake.tracked_ips()
        );
    }

    // ---- C05 LOW: constructor / normalization hardening --------------------

    #[tokio::test]
    async fn auth_brake_clamps_zero_block_secs() {
        // A 0-second block would make `block_until == blocked_at`, silently
        // disabling blocking. Clamped to >= 1s.
        let brake = AuthBrake::new(1, 60, 0);
        brake.record_failure("8.8.8.8").await;
        assert!(
            brake.is_blocked("8.8.8.8").await,
            "block_secs=0 must be clamped so blocking still engages"
        );
        std::thread::sleep(std::time::Duration::from_millis(1_300));
        assert!(!brake.is_blocked("8.8.8.8").await);
    }

    #[test]
    fn check_ip_allowed_rejects_ipv4_compatible_v6() {
        // The deprecated IPv4-compatible IPv6 form (`::a.b.c.d`) must be
        // normalized so the private-range checks apply.
        assert!(!check_ip_allowed("::10.0.0.1".parse().unwrap()));
        assert!(!check_ip_allowed("::127.0.0.1".parse().unwrap()));
        assert!(!check_ip_allowed("::169.254.169.254".parse().unwrap()));
        // An IPv4-compatible form of a public IPv4 stays allowed.
        assert!(check_ip_allowed("::8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn check_ip_allowed_rejects_newly_added_reserved_ranges() {
        for ip in [
            "198.18.0.1",
            "198.51.100.7",
            "203.0.113.5",
            "192.0.2.1",
            "192.0.0.1",
            "224.0.0.1",
            "240.0.0.1",
            "192.88.99.1",
            "192.31.196.1",
            "192.52.193.1",
        ] {
            assert!(
                !check_ip_allowed(ip.parse().unwrap()),
                "{ip} must be rejected as a reserved range"
            );
        }
        assert!(!check_ip_allowed("2001:db8::1".parse().unwrap()));
    }

    #[test]
    fn check_ip_allowed_rejects_nat64_embedded_private() {
        // The NAT64 well-known prefix embeds an IPv4 address in the low 32
        // bits; on a NAT64-enabled host the embedded private IPv4 stays
        // reachable, so it must be evaluated against the same table. `64:ff9b
        // ::a00:1` embeds 10.0.0.1; `64:ff9b::808:808` embeds 8.8.8.8.
        assert!(!check_ip_allowed("64:ff9b::a00:1".parse().unwrap()));
        assert!(!check_ip_allowed("64:ff9b::7f00:1".parse().unwrap()));
        assert!(check_ip_allowed("64:ff9b::808:808".parse().unwrap()));
    }

    #[test]
    fn check_ip_allowed_rejects_ipv6_multicast() {
        // The reference blocks multicast via `ip.is_multicast` for both IPv4 and
        // IPv6; oxllm's `PRIVATE_CIDRS` covers IPv4 (`224.0.0.0/4`) and must
        // also cover the IPv6 multicast range `ff00::/8` to keep parity with
        // the default-deny posture. Practical risk is low (TCP to multicast
        // is unusual) but it is a concrete defense-in-depth gap.
        assert!(!check_ip_allowed("ff00::1".parse().unwrap()));
        assert!(!check_ip_allowed("ff02::1".parse().unwrap()));
        assert!(!check_ip_allowed("ff0e::1234".parse().unwrap()));
        assert!(!check_ip_allowed("fffe::abcd".parse().unwrap()));
    }

    // ---- C05 LOW: capped_byte_stream surfaces upstream errors --------------

    #[tokio::test]
    async fn capped_byte_stream_surfaces_upstream_error_after_cap() {
        use futures_util::StreamExt;
        // Obtain a genuine reqwest error (connection refused on loopback).
        let upstream_err = reqwest::get("http://127.0.0.1:1/").await.unwrap_err();
        let input = futures_util::stream::iter(vec![
            Ok::<axum::body::Bytes, reqwest::Error>(axum::body::Bytes::from_static(b"abcd")),
            Ok(axum::body::Bytes::from_static(b"efgh")),
            Err(upstream_err),
        ]);
        let mut capped = capped_byte_stream(input, 4);
        let mut items = Vec::new();
        while let Some(item) = capped.next().await {
            items.push(item);
        }
        assert_eq!(items.len(), 3, "chunk, cap-abort, upstream-error");
        assert!(items[0].is_ok());
        assert!(
            items[1].is_err(),
            "cap-abort must surface a mid-stream error"
        );
        let upstream_source = items[2]
            .as_ref()
            .unwrap_err()
            .get_ref()
            .and_then(|src| src.downcast_ref::<reqwest::Error>());
        assert!(
            upstream_source.is_some(),
            "an upstream error after the cap must be surfaced, not swallowed"
        );
    }

    // ---- C05 LOW: auth-failure log volume ---------------------------------

    /// A minimal subscriber that counts WARN+ events, to prove per-failure
    /// auth recording no longer logs at warn (a brute-force attacker would
    /// otherwise flood the log subsystem).
    struct WarnCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl tracing::Subscriber for WarnCounter {
        fn enabled(&self, _m: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _span: &tracing::span::Id, _event: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            // tracing's Level ordering is reversed from severity (DEBUG > WARN
            // > ERROR), so "WARN or more severe" is `level <= WARN`.
            if *event.metadata().level() <= tracing::Level::WARN {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        fn enter(&self, _span: &tracing::span::Id) {}
        fn exit(&self, _span: &tracing::span::Id) {}
        fn clone_span(&self, _span: &tracing::span::Id) -> tracing::span::Id {
            tracing::span::Id::from_u64(2)
        }
        fn try_close(&self, _id: tracing::span::Id) -> bool {
            true
        }
    }

    #[test]
    fn repeated_auth_failures_do_not_log_warn_per_failure() {
        let warns = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let subscriber = WarnCounter(warns.clone());
        // with_default scopes the subscriber to this thread only, so parallel
        // test threads are unaffected.
        tracing::subscriber::with_default(subscriber, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let brake = AuthBrake::new(10, 60, 300);
                for i in 0..25u32 {
                    brake.record_failure(&format!("198.18.{i}.{i}")).await;
                }
            });
        });
        assert_eq!(
            warns.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "per-failure recording must not log at WARN+ (log-volume DoS)"
        );
    }

    // ---- C05 MEDIUM: allowed_ips peer semantics ----------------------------

    #[tokio::test]
    async fn allowed_ips_guard_peer_is_authoritative_for_untrusted_private_peer() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        // allowed_ips = the peer's private range, no trusted_proxy_cidrs, and a
        // spoofed public X-Forwarded-For: the allowlist keys on the PEER
        // (forwarded headers are client-controllable and ignored for an
        // untrusted peer). This documents the footgun flagged by the startup
        // warning — an allowlist that includes the peer's range lets every
        // client behind that peer pass, so operators must configure
        // `trusted_proxy_cidrs` to check the real client.
        let config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                allowed_ips: Some(vec!["10.0.0.0/8".to_string()]),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let shared = crate::app::build_shared_state(config).expect("state");

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                allowed_ips_guard,
            ));

        // Peer inside the allowlist → allowed, spoofed XFF ignored.
        let peer: std::net::SocketAddr = "10.0.0.5:8080".parse().unwrap();
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("x-forwarded-for", "8.8.8.8")
            .extension(axum::extract::ConnectInfo(peer))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "an allowlisted peer passes even with a spoofed XFF (peer is authoritative)"
        );

        // Peer OUTSIDE the allowlist → rejected, spoofed XFF still ignored.
        let peer: std::net::SocketAddr = "192.168.1.50:8080".parse().unwrap();
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("x-forwarded-for", "8.8.8.8")
            .extension(axum::extract::ConnectInfo(peer))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::FORBIDDEN,
            "a non-allowlisted peer must be rejected regardless of spoofed XFF"
        );
    }

    // ---- C05 MEDIUM: SSRF cloud-fabric IP blocklist ------------------------

    #[test]
    fn check_ip_allowed_rejects_cloud_metadata_exceptions() {
        // Globally-routable cloud-fabric IPs reachable from inside a cloud VM
        // must be blocked by the SSRF guard even though they are not in any
        // RFC 6890 private range. Mirrors the reference
        // `_CLOUD_METADATA_EXCEPTIONS` (Azure Wire Server) plus the Alibaba
        // Cloud metadata service.
        assert!(
            !check_ip_allowed(IpAddr::from([168, 63, 129, 16])),
            "Azure Wire Server (168.63.129.16) must be blocked"
        );
        assert!(
            !check_ip_allowed(IpAddr::from([100, 100, 100, 200])),
            "Alibaba Cloud metadata (100.100.100.200) must be blocked"
        );
    }

    #[test]
    fn cloud_metadata_exceptions_constant_parses() {
        for s in CLOUD_METADATA_EXCEPTIONS {
            assert!(
                s.parse::<ipnet::IpNet>().is_ok(),
                "CLOUD_METADATA_EXCEPTIONS entry '{s}' must parse"
            );
        }
    }

    // ---- C05 MEDIUM: evict_oldest preserves actively-blocked IPs -----------

    #[tokio::test]
    async fn evict_oldest_preserves_actively_blocked_ip() {
        // Cap=4, threshold=3, window=1s, block=300s. Fill 3 slots with fresh,
        // non-blocked entries (one failure each, below threshold). Directly
        // install one actively-blocked entry whose window_start is backdated
        // well past `window` but whose block is still active. Recording a 5th
        // distinct IP triggers eviction; the actively-blocked IP must survive
        // (the pre-fix behavior would evict it as the oldest-window entry).
        let brake = AuthBrake::new(3, 1, 300).with_max_ips(4);
        brake.record_failure("10.0.0.1").await;
        brake.record_failure("10.0.0.2").await;
        brake.record_failure("10.0.0.3").await;
        let old_window_start = Instant::now()
            .checked_sub(Duration::from_secs(60))
            .expect("Instant can go back 60s");
        {
            let mut guard = brake.inner.lock();
            guard.by_ip.insert(
                "10.0.0.99".to_string(),
                FailureRecord {
                    count: 3,
                    window_start: old_window_start,
                    blocked_at: Some(Instant::now()),
                },
            );
        }
        assert_eq!(brake.tracked_ips(), 4, "map is at the cap");

        // Trigger eviction with one more distinct IP.
        brake.record_failure("10.0.0.4").await;
        assert_eq!(
            brake.tracked_ips(),
            4,
            "cap must still be enforced after eviction"
        );
        assert!(
            brake.is_blocked("10.0.0.99").await,
            "actively-blocked IP must survive eviction despite the oldest window_start"
        );
    }

    #[tokio::test]
    async fn evict_oldest_still_enforces_cap_when_all_entries_blocked() {
        // When every entry is actively blocked, the cap must still be
        // enforced: the oldest-window blocked entry is evicted rather than
        // growing the map past max_ips.
        let brake = AuthBrake::new(1, 60, 300).with_max_ips(2);
        brake.record_failure("10.0.0.1").await; // blocked
        brake.record_failure("10.0.0.2").await; // blocked
        assert_eq!(brake.tracked_ips(), 2);
        assert!(brake.is_blocked("10.0.0.1").await);
        assert!(brake.is_blocked("10.0.0.2").await);
        // Third distinct IP forces an eviction even though both are blocked.
        brake.record_failure("10.0.0.3").await;
        assert!(
            brake.tracked_ips() <= 2,
            "cap must hold even when all entries are actively blocked"
        );
    }

    // ---- C05 LOW: remaining_block_secs -------------------------------------

    #[tokio::test]
    async fn remaining_block_secs_reports_time_left_not_full_window() {
        let brake = AuthBrake::new(1, 60, 10);
        brake.record_failure("9.9.9.9").await;
        assert!(brake.is_blocked("9.9.9.9").await);
        tokio::time::sleep(Duration::from_millis(1_300)).await;
        let remaining = brake
            .remaining_block_secs("9.9.9.9")
            .await
            .expect("blocked IP should report a remaining time");
        assert!(
            remaining < brake.block_secs(),
            "remaining ({remaining}) must be < full block_secs ({}) after sleeping",
            brake.block_secs()
        );
        assert!(remaining >= 1, "remaining must stay positive while blocked");
    }

    #[tokio::test]
    async fn remaining_block_secs_none_for_unblocked_ip() {
        let brake = AuthBrake::new(3, 60, 300);
        assert!(
            brake.remaining_block_secs("1.2.3.4").await.is_none(),
            "unblocked IP must report no remaining block time"
        );
    }

    // ---- C05 LOW: duplicate XFF headers are merged -------------------------

    #[test]
    fn client_ip_merges_duplicate_xff_headers() {
        // RFC 7230 §3.2.2 permits duplicate headers; a misconfigured proxy
        // that passes two `X-Forwarded-For` headers through without merging
        // would leave the trusted (rightmost) value in the second header.
        // `headers.get` returns only the first; the merge must consider both.
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "spoofed".parse().unwrap());
        h.append("x-forwarded-for", "5.6.7.8".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.1:1234".parse().unwrap();
        assert_eq!(
            client_ip(&h, Some(peer), &trusted),
            "5.6.7.8",
            "duplicate XFF headers must be merged so the rightmost across both is chosen"
        );
    }

    #[test]
    fn join_xff_headers_returns_none_when_absent() {
        let h = HeaderMap::new();
        assert!(join_xff_headers(&h).is_none());
    }

    #[test]
    fn join_xff_headers_joins_multiple_values() {
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", "1.1.1.1".parse().unwrap());
        h.append("x-forwarded-for", "2.2.2.2".parse().unwrap());
        assert_eq!(join_xff_headers(&h).as_deref(), Some("1.1.1.1, 2.2.2.2"));
    }

    // ---- C05 LOW: body_limit_bytes saturating ceiling ----------------------

    #[test]
    fn body_limit_huge_value_does_not_wrap() {
        // A config value that overflows usize on a 32-bit target must saturate
        // to usize::MAX rather than silently wrapping to a small value. On a
        // 64-bit target `u64::MAX` still saturates the subsequent multiply.
        let huge = body_limit_bytes(Some(u64::MAX));
        assert_eq!(
            huge,
            usize::MAX,
            "huge config must saturate to usize::MAX, not wrap to a smaller limit"
        );
    }

    // ---- C05 LOW: streaming relay strips all hop-by-hop headers ------------

    #[tokio::test]
    async fn sanitize_upstream_response_strips_all_hop_by_hop_headers_on_relay() {
        // RFC 7230 §6.1 lists connection, keep-alive, proxy-authenticate,
        // proxy-authorization, te, trailer, transfer-encoding, upgrade as
        // hop-by-hop. The relay must strip all of them, not just the first
        // three, so an upstream `upgrade` / `proxy-authenticate` never reaches
        // the downstream client.
        let payload = b"binary-payload".to_vec();
        let http_resp = axum::http::Response::builder()
            .status(200)
            .header("content-type", "application/octet-stream")
            .header("content-length", payload.len().to_string())
            .header("upgrade", "h2c")
            .header("proxy-authenticate", "Basic realm=\"x\"")
            .header("proxy-authorization", "Bearer x")
            .header("te", "trailers")
            .header("trailer", "X-Foo")
            .body(axum::body::Bytes::from(payload.clone()))
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp)
            .await
            .expect("binary success should stream");
        for h in [
            "upgrade",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "connection",
            "keep-alive",
        ] {
            assert!(
                result.headers().get(h).is_none(),
                "hop-by-hop header '{h}' must be stripped from the relayed response"
            );
        }
        // A non-hop-by-hop header must survive.
        assert!(result.headers().get("content-type").is_some());
    }

    // ---- C05 LOW: buffered JSON + error paths strip hop-by-hop headers ------

    #[tokio::test]
    async fn sanitize_upstream_response_strips_hop_by_hop_headers_on_buffered_json() {
        // The buffered JSON success path rebuilds the response from a fixed
        // `Vec<u8>`; a surviving upstream `transfer-encoding: chunked` would
        // describe a framing the client no longer receives. It must strip
        // every RFC 7230 §6.1 hop-by-hop header, exactly like the relay path.
        let payload = b"{\"ok\":true}".to_vec();
        let http_resp = axum::http::Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .header("content-length", payload.len().to_string())
            .header("transfer-encoding", "chunked")
            .header("connection", "keep-alive")
            .header("keep-alive", "timeout=5")
            .header("upgrade", "h2c")
            .header("proxy-authenticate", "Basic realm=\"x\"")
            .header("proxy-authorization", "Bearer x")
            .header("te", "trailers")
            .header("trailer", "X-Foo")
            .body(axum::body::Bytes::from(payload.clone()))
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp)
            .await
            .expect("buffered JSON success should produce a response");
        assert_eq!(result.status(), axum::http::StatusCode::OK);
        for h in [
            "transfer-encoding",
            "connection",
            "keep-alive",
            "upgrade",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
        ] {
            assert!(
                result.headers().get(h).is_none(),
                "hop-by-hop header '{h}' must be stripped from the buffered JSON response"
            );
        }
        // A non-hop-by-hop header must survive.
        assert!(result.headers().get("content-type").is_some());
    }

    #[tokio::test]
    async fn sanitize_upstream_response_strips_hop_by_hop_headers_on_error_path() {
        // The error path rebuilds the response from a truncated body too, so a
        // surviving `transfer-encoding` / `connection` would likewise mismatch
        // the re-framed body.
        let payload = br#"{"error":"boom"}"#.to_vec();
        let http_resp = axum::http::Response::builder()
            .status(502)
            .header("content-type", "application/json")
            .header("content-length", payload.len().to_string())
            .header("transfer-encoding", "chunked")
            .header("connection", "keep-alive")
            .header("upgrade", "h2c")
            .body(axum::body::Bytes::from(payload.clone()))
            .expect("build http response");
        let resp: reqwest::Response = http_resp.into();
        let result = sanitize_upstream_response(resp)
            .await
            .expect("error path must produce a response");
        assert_eq!(result.status(), axum::http::StatusCode::BAD_GATEWAY);
        for h in ["transfer-encoding", "connection", "keep-alive", "upgrade"] {
            assert!(
                result.headers().get(h).is_none(),
                "hop-by-hop header '{h}' must be stripped from the error response"
            );
        }
    }

    // ---- C05 LOW: multi-valued upstream response headers --------------------

    #[tokio::test]
    async fn sanitize_upstream_response_preserves_multi_valued_headers() {
        // The three rebuild loops must use `append`, not `insert`: iterating a
        // HeaderMap yields every value of a multi-valued header, and `insert`
        // would keep only the last one — silently dropping all but the final
        // `set-cookie` on pass-through targets that set several cookies (the
        // common case for auth endpoints). Exercise all three paths: relay,
        // buffered JSON, and error.
        let build = |status: u16, content_type: &str| {
            axum::http::Response::builder()
                .status(status)
                .header("content-type", content_type)
                .header("set-cookie", "a=1; Path=/; HttpOnly")
                .header("set-cookie", "b=2; Path=/; Secure")
                .body(axum::body::Bytes::from_static(b"payload"))
                .expect("build http response")
        };

        let relay: reqwest::Response = build(200, "application/octet-stream").into();
        let result = sanitize_upstream_response(relay).await.expect("relay path");
        assert_eq!(
            result.headers().get_all("set-cookie").iter().count(),
            2,
            "relay path must preserve every set-cookie value"
        );

        let buffered: reqwest::Response = build(200, "application/json").into();
        let result = sanitize_upstream_response(buffered)
            .await
            .expect("buffered JSON path");
        assert_eq!(
            result.headers().get_all("set-cookie").iter().count(),
            2,
            "buffered JSON path must preserve every set-cookie value"
        );

        let error: reqwest::Response = build(502, "application/json").into();
        let result = sanitize_upstream_response(error).await.expect("error path");
        assert_eq!(
            result.headers().get_all("set-cookie").iter().count(),
            2,
            "error path must preserve every set-cookie value"
        );
    }

    // ---- C05 LOW: explicitly empty allowed_ips denies all -------------------

    #[tokio::test]
    async fn allowed_ips_guard_empty_list_denies_all() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        // `allowed_ips: []` must deny every request (the reference IP check
        // compares against the list directly), never silently allow all.
        let config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                allowed_ips: Some(vec![]),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let shared = crate::app::build_shared_state(config).expect("state");

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                allowed_ips_guard,
            ));

        for peer in ["10.0.0.5:8080", "8.8.8.8:1234"] {
            let req = Request::builder()
                .method("GET")
                .uri("/healthz")
                .extension(axum::extract::ConnectInfo(
                    peer.parse::<std::net::SocketAddr>().unwrap(),
                ))
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                axum::http::StatusCode::FORBIDDEN,
                "an explicitly empty allowlist must deny {peer}"
            );
        }
    }

    // ---- C05 LOW: distributed-brake outage log volume -----------------------

    #[test]
    fn distributed_brake_outage_logs_at_most_one_warn() {
        // During a sustained backing-store outage every block-check and every
        // failure record would log one WARN (up to 3 per request under
        // brute-force traffic). The one-shot must cap the WARN volume at one
        // per process; subsequent outage hits drop to debug.
        DISTRIBUTED_CACHE_WARNED.store(false, std::sync::atomic::Ordering::Relaxed);
        let warns = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let subscriber = WarnCounter(warns.clone());
        tracing::subscriber::with_default(subscriber, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let coord =
                    crate::coordination::Coordinator::new(std::sync::Arc::new(FailingCache))
                        .with_distributed(true);
                let brake = AuthBrake::new(10, 60, 300).with_coordinator(Some(coord));
                for i in 0..25u32 {
                    let ip = format!("198.18.{i}.{i}");
                    let _ = brake.is_blocked(&ip).await;
                    brake.record_failure(&ip).await;
                }
            });
        });
        assert!(
            warns.load(std::sync::atomic::Ordering::Relaxed) <= 1,
            "a sustained backing-store outage must log at most one WARN, got {}",
            warns.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    // ---- C05 LOW: brake-skip visibility -------------------------------------

    #[test]
    fn auth_brake_guard_unattributable_warns_once() {
        // Production stacks short-circuit in `auth_brake_guard` (the backstop
        // one-shot in auth/mod.rs is never reached), so the guard itself must
        // surface the "enforcement silently inactive" condition — once per
        // process, not per request.
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        BRAKE_SKIP_WARNED.store(false, std::sync::atomic::Ordering::Relaxed);
        let config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-brake-warn".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let shared = crate::app::build_shared_state(config).expect("state");
        assert!(
            shared.load_full().auth_brake.is_some(),
            "master key must arm the auth brake"
        );

        let warns = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let subscriber = WarnCounter(warns.clone());
        tracing::subscriber::with_default(subscriber, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let app = Router::new()
                    .route("/healthz", get(|| async { "ok" }))
                    .layer(axum::middleware::from_fn_with_state(
                        shared.clone(),
                        auth_brake_guard,
                    ));
                for _ in 0..5 {
                    // No ConnectInfo extension: unattributable → skip path.
                    let req = Request::builder()
                        .method("GET")
                        .uri("/healthz")
                        .body(Body::empty())
                        .unwrap();
                    let resp = app.clone().oneshot(req).await.unwrap();
                    assert_eq!(resp.status(), axum::http::StatusCode::OK);
                }
            });
        });
        assert_eq!(
            warns.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the first unattributable request must WARN exactly once, not per request"
        );
    }
}
