//! A disk-backed `Cache` implementation that stores each entry as a file.
//!
//! This is intended for **local development and single-instance deployments**
//! where Redis is unavailable but persistence across restarts is wanted. It
//! implements the same `Cache` trait as [`InMemoryCache`](crate::cache::InMemoryCache)
//! and `oxllm_cache_redis::RedisCache`, so it can be swapped in wherever a
//! `Cache` is accepted.
//!
//! ## Storage layout
//!
//! Each key maps to a file named `<sha256(key)>` under [`DiskCache::base_dir`].
//! The file begins with an 8-byte little-endian expiry header (epoch
//! milliseconds; `0` means "no TTL"), followed by the UTF-8 value bytes.
//!
//! ## Concurrency
//!
//! `incr` / `incr_if_below` override the trait defaults with a
//! `parking_lot::Mutex`-guarded read-check-write, giving true **in-process**
//! atomicity (mirroring [`InMemoryCache`](crate::cache::InMemoryCache)). This
//! is the correct model for a single-instance local cache; cross-process
//! atomicity is not provided.
//!
//! Not wired in by default — construct explicitly with [`DiskCache::new`].
//! This legacy backend does not supply atomic budget-window receipts: budget
//! admission fails closed through the Cache default rather than refunding an
//! unproven file incarnation. Response caching and legacy counters are unchanged.

use std::fmt::Write as _;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use oxllm_core::{Cache as CacheTrait, Error, IncrResult};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};

/// Number of bytes in the on-disk expiry header.
const HEADER_LEN: usize = 8;
/// Sentinel stored in the header to mean "no TTL".
const NO_EXPIRY: u64 = 0;

/// A disk-backed `Cache` that persists entries as files under a base dir.
///
/// See the [module docs](self) for the storage layout and concurrency model.
pub struct DiskCache {
    base_dir: PathBuf,
    /// Serializes `incr` / `incr_if_below` read-check-write sections so they
    /// are truly atomic within this process. `Arc` so it can be cheaply cloned
    /// into each `spawn_blocking` task.
    write_lock: Arc<Mutex<()>>,
}

impl DiskCache {
    /// Create a `DiskCache` rooted at `base_dir`, creating the directory (and
    /// any missing parents) if it does not already exist.
    ///
    /// Also performs two best-effort GC passes over `base_dir`:
    /// - removes orphaned `*.tmp` files left behind by a previous process that
    ///   crashed between `write` and `rename` in `write_entry_at`. A `.tmp`
    ///   file is only alive for the duration of a single write, so any present
    ///   at construction time is orphaned.
    /// - removes expired entry files. The read path (`read_entry_at`) lazily
    ///   deletes an expired entry on hit, but an entry that expires and is
    ///   never re-read stays on disk indefinitely — a slow unbounded disk-space
    ///   leak for a long-running single-instance deployment. Reclaiming it at
    ///   construction bounds disk usage across restarts.
    pub fn new<P: Into<PathBuf>>(base_dir: P) -> Result<Self, Error> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)
            .map_err(|e| Error::Config(format!("disk cache dir: {e}")))?;
        cleanup_orphaned_tmp(&base_dir);
        cleanup_expired_entries(&base_dir);
        Ok(Self {
            base_dir,
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    /// The directory this cache reads from and writes to.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Resolve the on-disk path for `key` (a SHA-256 hex digest).
    fn path_for(&self, key: &str) -> PathBuf {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            // Writes directly into `hex` with no per-byte allocation.
            let _ = write!(hex, "{byte:02x}");
        }
        self.base_dir.join(hex)
    }

    /// Decode an entry file's bytes into `(value, expiry_millis)` where
    /// `expiry_millis == None` means "no TTL".
    fn decode(bytes: &[u8]) -> Option<(String, Option<u64>)> {
        if bytes.len() < HEADER_LEN {
            // An empty / truncated file is treated as a miss.
            return None;
        }
        let mut header = [0u8; HEADER_LEN];
        header.copy_from_slice(&bytes[..HEADER_LEN]);
        let expiry_raw = u64::from_le_bytes(header);
        let expiry = (expiry_raw != NO_EXPIRY).then_some(expiry_raw);
        let value = std::str::from_utf8(&bytes[HEADER_LEN..]).ok()?.to_string();
        Some((value, expiry))
    }

    /// Whether an expiry timestamp (epoch millis) has passed.
    fn is_expired(expiry_millis: Option<u64>) -> bool {
        match expiry_millis {
            Some(exp) => now_millis() >= exp,
            None => false,
        }
    }
}

/// Outcome of reading an entry file, distinguishing a genuine miss from a
/// present-but-unreadable entry. The distinction matters for counters: a
/// corrupted spend counter must not silently read as zero (which would reset
/// the caller's budget ceiling to "unlimited").
enum ReadOutcome {
    /// No file on disk, or an expired entry that was swept on read.
    Absent,
    /// A valid, live entry.
    Present(String, Option<u64>),
    /// A file exists but is unreadable / truncated / non-UTF-8.
    Corrupt,
}

/// Read the raw `(value, expiry)` for `path`, honoring expiry (expired entries
/// are deleted and read as [`ReadOutcome::Absent`]).
fn read_entry_at(path: &Path) -> ReadOutcome {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == ErrorKind::NotFound => return ReadOutcome::Absent,
        Err(_) => return ReadOutcome::Corrupt,
    };
    let Some((value, expiry)) = DiskCache::decode(&bytes) else {
        return ReadOutcome::Corrupt;
    };
    if DiskCache::is_expired(expiry) {
        // Best-effort cleanup of an expired entry.
        let _ = std::fs::remove_file(path);
        return ReadOutcome::Absent;
    }
    ReadOutcome::Present(value, expiry)
}

/// Read the current counter value at `path`, failing closed (`Err`) when a
/// value is present but corrupt or unreadable. A genuinely absent key reads as
/// `0.0`. The prior expiry is returned so callers can preserve the fixed
/// window (see [`next_expiry`]).
fn read_counter(path: &Path, key_for_log: &str) -> Result<(f64, Option<u64>), Error> {
    match read_entry_at(path) {
        ReadOutcome::Present(v, exp) => match v.parse::<f64>() {
            Ok(n) => Ok((n, exp)),
            Err(_) => {
                tracing::warn!(key = %key_for_log, "disk counter corrupt; failing closed");
                Err(Error::Config("disk counter corrupt".into()))
            }
        },
        ReadOutcome::Corrupt => {
            tracing::warn!(key = %key_for_log, "disk counter unreadable; failing closed");
            Err(Error::Config("disk counter unreadable".into()))
        }
        ReadOutcome::Absent => Ok((0.0, None)),
    }
}

/// Resolve the absolute expiry timestamp to stamp, preserving a fixed window:
/// an existing expiry is reused unchanged (mirroring `InMemoryCache::incr`,
/// which sets the TTL only when the counter is new), and a counter without a
/// prior expiry picks up `ttl`.
fn next_expiry(existing: Option<u64>, ttl: Option<Duration>) -> u64 {
    match existing {
        Some(exp) => exp,
        None => match ttl {
            Some(d) => now_millis().saturating_add(clamp_ttl_millis(d)),
            None => NO_EXPIRY,
        },
    }
}

/// Clamp a TTL to `MAX_TTL_SECS` and return it as milliseconds. Mirrors the
/// in-memory cache's `clamp_ttl` so the disk cache never wraps `u64` on an
/// absurd caller TTL (e.g. a huge JWT `exp` claim): without the clamp,
/// `d.as_millis() as u64` truncates `u128 → u64` and a ~5.8e8-year TTL wraps
/// to ~384 ms, expiring the entry almost immediately.
fn clamp_ttl_millis(d: Duration) -> u64 {
    super::clamp_ttl(d).as_millis() as u64
}

/// Encode and atomically persist `value` with the given absolute expiry
/// timestamp. Atomicity is achieved by writing a temp file then renaming it
/// into place. I/O failures are surfaced so callers that settle spend/budget
/// debits on the returned value do not silently drop them.
fn write_entry_at(path: &Path, value: &str, expiry: u64) -> Result<(), Error> {
    let mut out = Vec::with_capacity(HEADER_LEN + value.len());
    out.extend_from_slice(&expiry.to_le_bytes());
    out.extend_from_slice(value.as_bytes());

    // Atomic replace via temp file + rename.
    let tmp = tmp_path_for(path);
    if let Err(e) = std::fs::write(&tmp, &out) {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::Network(format!("disk cache write: {e}")));
    }
    // rename will overwrite the destination on unix; on Windows a best-effort
    // remove-then-rename is used.
    #[cfg(unix)]
    let res = std::fs::rename(&tmp, path);
    #[cfg(not(unix))]
    let res = {
        let _ = std::fs::remove_file(path);
        std::fs::rename(&tmp, path)
    };
    if let Err(e) = res {
        // The write succeeded but the rename did not: clean up the temp file
        // so repeated failures do not accumulate .tmp files in base_dir.
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::Network(format!("disk cache rename: {e}")));
    }
    Ok(())
}

/// Best-effort removal of an entry file (missing file is a no-op).
fn delete_entry_at(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Helper to derive a unique temp-file path next to a target path. Including
/// the PID and current time guards against collisions between concurrent
/// writers and across processes.
fn tmp_path_for(path: &Path) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut s = path.as_os_str().to_owned();
    s.push(format!(".{pid}.{nanos}.tmp"));
    PathBuf::from(s)
}

/// Best-effort removal of orphaned `*.tmp` files under `base_dir`. A `.tmp`
/// file is only alive for the duration of a single [`write_entry_at`] call;
/// any present when a new `DiskCache` is constructed was left behind by a
/// previous process that died between `write` and `rename`. Errors are logged
/// at `warn` (not surfaced) so a single unreadable entry does not prevent the
/// cache from starting.
fn cleanup_orphaned_tmp(base_dir: &Path) {
    let entries = match std::fs::read_dir(base_dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, dir = ?base_dir, "disk cache tmp GC: read_dir failed");
            return;
        }
    };
    let mut removed = 0u32;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "tmp") {
            if let Err(e) = std::fs::remove_file(&path) {
                if e.kind() != ErrorKind::NotFound {
                    tracing::warn!(error = %e, path = ?path, "disk cache tmp GC: remove failed");
                }
            } else {
                removed += 1;
            }
        }
    }
    if removed > 0 {
        tracing::info!(removed, dir = ?base_dir, "disk cache: cleaned orphaned tmp files");
    }
}

/// Best-effort removal of expired entry files under `base_dir`. The read path
/// ([`read_entry_at`]) deletes an expired entry on hit, but an entry that
/// expires and is never re-read stays on disk indefinitely — a slow unbounded
/// disk-space leak for a long-running single-instance deployment. Running this
/// sweep at construction reclaims that space across restarts. Errors reading
/// individual entries are skipped (not surfaced) so a single unreadable file
/// does not prevent the rest of the sweep.
fn cleanup_expired_entries(base_dir: &Path) {
    let entries = match std::fs::read_dir(base_dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, dir = ?base_dir, "disk cache expired GC: read_dir failed");
            return;
        }
    };
    let mut removed = 0u32;
    for entry in entries.flatten() {
        let path = entry.path();
        // Skip orphaned .tmp files; those are handled by `cleanup_orphaned_tmp`.
        if path.extension().is_some_and(|ext| ext == "tmp") {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Some((_, expiry)) = DiskCache::decode(&bytes) else {
            continue;
        };
        if DiskCache::is_expired(expiry) && std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    if removed > 0 {
        tracing::info!(removed, dir = ?base_dir, "disk cache: cleaned expired entries");
    }
}

/// Current time as milliseconds since the Unix epoch.
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[async_trait]
impl CacheTrait for DiskCache {
    async fn get(&self, key: &str) -> Option<String> {
        let path = self.path_for(key);
        match tokio::task::spawn_blocking(move || read_entry_at(&path)).await {
            Ok(ReadOutcome::Present(v, _)) => Some(v),
            _ => None,
        }
    }

    async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
        if ttl.is_some_and(|ttl| ttl.is_zero()) {
            self.delete(key).await;
            return;
        }
        // The trait `set` is infallible, so a write failure cannot be surfaced
        // to the caller — log it loudly so the silent drop is at least
        // observable. (For the counter path, `incr`/`incr_if_below` surface
        // the same failure as `Err`.)
        let path = self.path_for(key);
        let expiry = match ttl {
            Some(d) => now_millis().saturating_add(clamp_ttl_millis(d)),
            None => NO_EXPIRY,
        };
        let key_str = key.to_string();
        match tokio::task::spawn_blocking(move || write_entry_at(&path, &value, expiry)).await {
            Ok(Err(e)) => tracing::error!(error = %e, key = %key_str, "disk cache set failed"),
            Err(e) => tracing::error!(error = %e, key = %key_str, "disk cache set task failed"),
            Ok(Ok(())) => {}
        }
    }

    async fn delete(&self, key: &str) {
        let path = self.path_for(key);
        let _ = tokio::task::spawn_blocking(move || delete_entry_at(&path)).await;
    }

    async fn incr(&self, key: &str, by: f64, ttl: Option<Duration>) -> Result<f64, Error> {
        if !by.is_finite() {
            return Err(Error::Config("incr rejected non-finite delta".into()));
        }
        let path = self.path_for(key);
        let lock = self.write_lock.clone();
        let key_str = key.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<f64, Error> {
            let _guard = lock.lock();
            let (current, existing_expiry) = read_counter(&path, &key_str)?;
            let new_val = current + by;
            write_entry_at(
                &path,
                &new_val.to_string(),
                next_expiry(existing_expiry, ttl),
            )?;
            Ok(new_val)
        })
        .await;
        result.map_err(|e| {
            tracing::error!(error = %e, key = %key, "disk cache incr task failed");
            Error::Internal
        })?
    }

    /// Truly atomic in-process compare-and-increment: the `write_lock` is held
    /// across the read-check-write inside a single `spawn_blocking` task, so
    /// concurrent callers within this instance can never exceed `ceiling`.
    async fn incr_if_below(
        &self,
        key: &str,
        delta: f64,
        ceiling: f64,
        ttl: Option<Duration>,
    ) -> Result<IncrResult, Error> {
        if !delta.is_finite() || !ceiling.is_finite() {
            return Err(Error::Config(
                "incr_if_below rejected non-finite delta/ceiling".into(),
            ));
        }
        let path = self.path_for(key);
        let lock = self.write_lock.clone();
        let key_str = key.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<IncrResult, Error> {
            let _guard = lock.lock();
            let (current, existing_expiry) = read_counter(&path, &key_str)?;
            let new_val = current + delta;
            if new_val <= ceiling {
                write_entry_at(
                    &path,
                    &new_val.to_string(),
                    next_expiry(existing_expiry, ttl),
                )?;
                Ok(IncrResult::Allowed(new_val))
            } else {
                Ok(IncrResult::Rejected(current))
            }
        })
        .await;
        result.map_err(|e| {
            tracing::error!(error = %e, key = %key, "disk cache incr_if_below task failed");
            Error::Internal
        })?
    }

    // `counter_value` inherits the trait default (delegate to
    // `try_counter_value`), which is correct once `try_counter_value` is
    // overridden to surface corruption observably — see below.

    /// Override the trait default so corruption is observable on the read
    /// path, not just the write path. The trait default goes through `get` →
    /// parse, which only emits a `warn` when the file body is valid UTF-8 but
    /// non-numeric (a truncated or non-UTF-8 body makes `decode` return
    /// `None`, so `get` returns `None` silently). Going through
    /// `read_entry_at` directly lets the `Corrupt` branch fail closed with
    /// `Err` (after a `warn`), matching `read_counter`/`incr`, so a pre-call
    /// budget check on a corrupt spend counter errors instead of silently
    /// reading as zero.
    async fn try_counter_value(&self, key: &str) -> Result<Option<f64>, Error> {
        let path = self.path_for(key);
        let key_str = key.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Option<f64>, Error> {
            match read_entry_at(&path) {
                ReadOutcome::Absent => Ok(None),
                ReadOutcome::Present(v, _) => match v.parse::<f64>() {
                    Ok(n) => Ok(Some(n)),
                    Err(_) => {
                        tracing::warn!(
                            key = %key_str,
                            "disk counter corrupt (non-numeric body); failing closed"
                        );
                        Err(Error::Config("disk counter corrupt".into()))
                    }
                },
                ReadOutcome::Corrupt => {
                    tracing::warn!(
                        key = %key_str,
                        "disk counter corrupt (truncated or non-utf8 body); failing closed"
                    );
                    Err(Error::Config("disk counter corrupt".into()))
                }
            }
        })
        .await;
        match result {
            Ok(inner) => inner,
            Err(e) => {
                tracing::error!(
                    error = %e, key = %key,
                    "disk cache try_counter_value task failed"
                );
                Err(Error::Internal)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn fresh_cache() -> DiskCache {
        let dir = tempfile::tempdir().expect("tempdir");
        // Leak the tempdir so its files survive for the test's lifetime.
        let path = dir.path().to_path_buf();
        std::mem::forget(dir);
        DiskCache::new(path).expect("disk cache")
    }

    #[tokio::test]
    async fn get_set_roundtrip() {
        let cache = fresh_cache();
        cache.set("k", "v".to_string(), None).await;
        assert_eq!(cache.get("k").await.as_deref(), Some("v"));
    }

    #[tokio::test]
    async fn miss_returns_none() {
        let cache = fresh_cache();
        assert!(cache.get("missing").await.is_none());
    }

    #[tokio::test]
    async fn delete_removes_entry() {
        let cache = fresh_cache();
        cache.set("k", "v".to_string(), None).await;
        cache.delete("k").await;
        assert!(cache.get("k").await.is_none());
    }

    #[tokio::test]
    async fn delete_missing_is_noop() {
        let cache = fresh_cache();
        // Should not panic.
        cache.delete("never-set").await;
        assert!(cache.get("never-set").await.is_none());
    }

    #[tokio::test]
    async fn set_overwrites_existing() {
        let cache = fresh_cache();
        cache.set("k", "v1".to_string(), None).await;
        cache.set("k", "v2".to_string(), None).await;
        assert_eq!(cache.get("k").await.as_deref(), Some("v2"));
    }

    #[tokio::test]
    async fn ttl_expires_entry() {
        let cache = fresh_cache();
        cache
            .set("k", "v".to_string(), Some(Duration::from_millis(50)))
            .await;
        assert_eq!(cache.get("k").await.as_deref(), Some("v"));
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(
            cache.get("k").await.is_none(),
            "entry should be gone after TTL"
        );
    }

    #[tokio::test]
    async fn ttl_expired_entry_is_deleted_on_read() {
        let cache = fresh_cache();
        cache
            .set("k", "v".to_string(), Some(Duration::from_millis(50)))
            .await;
        tokio::time::sleep(Duration::from_millis(120)).await;
        let _ = cache.get("k").await; // triggers cleanup
        assert!(
            !path_exists(&cache, "k"),
            "expired entry file should be removed on read"
        );
    }

    #[tokio::test]
    async fn no_ttl_persists() {
        let cache = fresh_cache();
        cache.set("k", "v".to_string(), None).await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(cache.get("k").await.as_deref(), Some("v"));
    }

    #[tokio::test]
    async fn incr_returns_new_value() {
        let cache = fresh_cache();
        assert!((cache.incr("c", 1.0, None).await.unwrap() - 1.0).abs() < 1e-9);
        assert!((cache.incr("c", 2.0, None).await.unwrap() - 3.0).abs() < 1e-9);
        assert!((cache.incr("c", 0.5, None).await.unwrap() - 3.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn incr_starts_from_zero_on_miss() {
        let cache = fresh_cache();
        assert!(
            (cache.incr("fresh", 5.0, None).await.unwrap() - 5.0).abs() < 1e-9,
            "incrementing a missing key starts at 0"
        );
    }

    #[tokio::test]
    async fn counter_value_reads_back() {
        let cache = fresh_cache();
        let _ = cache.incr("c", 4.0, None).await;
        assert_eq!(cache.counter_value("c").await, Some(4.0));
    }

    #[tokio::test]
    async fn counter_value_missing_is_none() {
        let cache = fresh_cache();
        assert!(cache.counter_value("missing").await.is_none());
    }

    #[tokio::test]
    async fn incr_if_below_allowed_when_under_ceiling() {
        let cache = fresh_cache();
        let res = cache
            .incr_if_below("rpm:k", 1.0, 3.0, Some(Duration::from_secs(60)))
            .await
            .expect("ok");
        assert_eq!(res, IncrResult::Allowed(1.0));
        assert_eq!(cache.counter_value("rpm:k").await, Some(1.0));
    }

    #[tokio::test]
    async fn incr_if_below_rejected_when_exceeds_ceiling() {
        let cache = fresh_cache();
        let _ = cache.incr("rpm:k", 3.0, None).await;
        let res = cache
            .incr_if_below("rpm:k", 1.0, 3.0, Some(Duration::from_secs(60)))
            .await
            .expect("ok");
        assert_eq!(res, IncrResult::Rejected(3.0));
        assert_eq!(
            cache.counter_value("rpm:k").await,
            Some(3.0),
            "rejected reservation must not mutate the counter"
        );
    }

    #[tokio::test]
    async fn incr_if_below_at_boundary_allows_exact_ceiling() {
        let cache = fresh_cache();
        let _ = cache.incr("rpm:k", 2.0, None).await;
        // 2 + 1 == 3 == ceiling -> allowed
        let res = cache
            .incr_if_below("rpm:k", 1.0, 3.0, None)
            .await
            .expect("ok");
        assert_eq!(res, IncrResult::Allowed(3.0));
    }

    #[tokio::test]
    async fn incr_if_below_keys_are_independent() {
        let cache = fresh_cache();
        let a = cache
            .incr_if_below("rpm:a", 1.0, 1.0, None)
            .await
            .expect("ok");
        let b = cache
            .incr_if_below("rpm:b", 1.0, 1.0, None)
            .await
            .expect("ok");
        assert_eq!(a, IncrResult::Allowed(1.0));
        assert_eq!(b, IncrResult::Allowed(1.0));
    }

    #[tokio::test]
    async fn incr_if_below_concurrent_never_overcommits() {
        let cache = Arc::new(fresh_cache());
        let ceiling = 3.0f64;
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            handles.push(tokio::spawn(async move {
                cache
                    .incr_if_below("rpm:race", 1.0, ceiling, Some(Duration::from_secs(60)))
                    .await
                    .expect("ok")
            }));
        }
        let mut allowed = 0u32;
        for h in handles {
            if matches!(h.await.expect("join"), IncrResult::Allowed(_)) {
                allowed += 1;
            }
        }
        assert_eq!(
            allowed, 3,
            "exactly `ceiling` callers admitted; got {allowed}"
        );
        assert_eq!(cache.counter_value("rpm:race").await, Some(3.0));
    }

    #[tokio::test]
    async fn persists_across_instances() {
        // The core promise of a disk cache: data written by one instance is
        // readable by a new instance sharing the same base dir.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_path_buf();
        {
            let cache = DiskCache::new(path.clone()).expect("cache");
            cache.set("k", "persisted".to_string(), None).await;
        }
        let cache2 = DiskCache::new(path).expect("cache2");
        assert_eq!(cache2.get("k").await.as_deref(), Some("persisted"));
    }

    #[tokio::test]
    async fn creates_base_dir_if_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let nested = dir.path().join("a/b/c");
        let cache = DiskCache::new(&nested).expect("nested cache");
        assert!(nested.exists(), "nested base dir should be created");
        cache.set("k", "v".to_string(), None).await;
        assert_eq!(cache.get("k").await.as_deref(), Some("v"));
    }

    #[tokio::test]
    async fn key_hashes_to_stable_filename() {
        let cache = fresh_cache();
        let p1 = cache.path_for("some/weird:key with spaces");
        let p2 = cache.path_for("some/weird:key with spaces");
        assert_eq!(p1, p2, "same key -> same file");
        // The filename must be a 64-char hex digest (no path-unsafe chars).
        let name = p1.file_name().unwrap().to_str().unwrap();
        assert_eq!(name.len(), 64);
        assert!(name.chars().all(|c| c.is_ascii_hexdigit()));
    }

    /// Concurrency: 100 callers each incrementing a shared counter by 1. The
    /// in-process write-lock serializes them, so the final counter must be
    /// exactly 100 and every returned value must be unique (no lost updates).
    #[tokio::test]
    async fn incr_concurrent_is_consistent() {
        let cache = Arc::new(fresh_cache());
        let mut handles = Vec::new();
        for _ in 0..100 {
            let cache = cache.clone();
            handles.push(tokio::spawn(async move {
                cache.incr("shared", 1.0, None).await.expect("ok")
            }));
        }
        let mut values = Vec::new();
        for h in handles {
            values.push(h.await.expect("join"));
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
        assert_eq!(
            values.len(),
            100,
            "all 100 increments must produce distinct values (no lost updates)"
        );
        assert_eq!(
            cache.counter_value("shared").await,
            Some(100.0),
            "final counter must be exactly 100"
        );
    }

    fn path_exists(cache: &DiskCache, key: &str) -> bool {
        cache.path_for(key).exists()
    }

    /// Read the raw expiry timestamp (epoch millis) stamped on a counter's
    /// file header. Used to assert fixed-window TTL behavior.
    fn read_stored_expiry(cache: &DiskCache, key: &str) -> u64 {
        let bytes = std::fs::read(cache.path_for(key)).expect("read entry file");
        assert!(bytes.len() >= HEADER_LEN, "entry file too short");
        let mut header = [0u8; HEADER_LEN];
        header.copy_from_slice(&bytes[..HEADER_LEN]);
        u64::from_le_bytes(header)
    }

    // ---- HIGH 1: write failures must surface as Err (fail closed) ----------

    /// When the base dir becomes unwritable, `incr` must report the failure
    /// rather than returning the would-be new value while dropping the debit.
    #[cfg(unix)]
    #[tokio::test]
    async fn incr_returns_err_when_disk_write_fails() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = DiskCache::new(dir.path()).expect("cache");
        let mut perms = std::fs::metadata(dir.path()).unwrap().permissions();
        perms.set_mode(0o444);
        std::fs::set_permissions(dir.path(), perms).expect("chmod read-only");
        // Root ignores directory mode bits, so the failure this test needs
        // cannot be produced. Nothing to assert in that case.
        if std::fs::write(dir.path().join(".probe"), b"x").is_ok() {
            let mut perms = std::fs::metadata(dir.path()).unwrap().permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(dir.path(), perms);
            return;
        }
        let result = cache.incr("k", 1.0, None).await;
        // Restore perms so the tempdir can be cleaned up.
        let mut perms = std::fs::metadata(dir.path()).unwrap().permissions();
        perms.set_mode(0o755);
        let _ = std::fs::set_permissions(dir.path(), perms);
        assert!(
            result.is_err(),
            "incr must surface a write failure as Err, got {result:?}"
        );
    }

    /// Same contract for `incr_if_below` on the allowed (write) path.
    #[cfg(unix)]
    #[tokio::test]
    async fn incr_if_below_returns_err_when_disk_write_fails() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = DiskCache::new(dir.path()).expect("cache");
        let mut perms = std::fs::metadata(dir.path()).unwrap().permissions();
        perms.set_mode(0o444);
        std::fs::set_permissions(dir.path(), perms).expect("chmod read-only");
        if std::fs::write(dir.path().join(".probe"), b"x").is_ok() {
            let mut perms = std::fs::metadata(dir.path()).unwrap().permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(dir.path(), perms);
            return;
        }
        let result = cache
            .incr_if_below("k", 1.0, 3.0, Some(Duration::from_secs(60)))
            .await;
        let mut perms = std::fs::metadata(dir.path()).unwrap().permissions();
        perms.set_mode(0o755);
        let _ = std::fs::set_permissions(dir.path(), perms);
        assert!(
            result.is_err(),
            "incr_if_below must surface a write failure as Err, got {result:?}"
        );
    }

    // ---- HIGH 2: corrupt counters must not reset to zero ------------------

    /// A counter file whose body is non-numeric must not silently parse as
    /// zero (which would re-open an exhausted budget). `incr` fails closed.
    #[tokio::test]
    async fn incr_fails_on_corrupted_counter() {
        let cache = fresh_cache();
        let path = cache.path_for("spend:k");
        let mut data = NO_EXPIRY.to_le_bytes().to_vec();
        data.extend_from_slice(b"not-a-number");
        std::fs::write(&path, &data).expect("write corrupt counter");
        let result = cache.incr("spend:k", 1.0, None).await;
        assert!(
            result.is_err(),
            "incr must not silently reset a corrupt counter to the delta"
        );
    }

    /// A truncated header (shorter than 8 bytes) is also treated as corrupt
    /// for counter operations.
    #[tokio::test]
    async fn incr_fails_on_truncated_counter() {
        let cache = fresh_cache();
        let path = cache.path_for("spend:k");
        std::fs::write(&path, b"\x01\x02\x03").expect("write truncated counter");
        let result = cache.incr("spend:k", 1.0, None).await;
        assert!(result.is_err(), "incr must fail closed on a truncated file");
    }

    /// A non-numeric present value fails closed: `try_counter_value` returns
    /// `Err` (and the trait-default `counter_value` maps that to `None`)
    /// rather than silently reading zero.
    #[tokio::test]
    async fn counter_value_err_on_non_numeric() {
        let cache = fresh_cache();
        let path = cache.path_for("spend:k");
        let mut data = NO_EXPIRY.to_le_bytes().to_vec();
        data.extend_from_slice(b"not-a-number");
        std::fs::write(&path, &data).expect("write corrupt counter");
        assert!(
            cache.try_counter_value("spend:k").await.is_err(),
            "non-numeric counter must fail closed, not read as absent"
        );
        assert_eq!(cache.counter_value("spend:k").await, None);
    }

    /// A truncated header (fewer than 8 bytes) must fail closed: the
    /// `try_counter_value` override goes through `read_entry_at`, so the
    /// `Corrupt` branch returns `Err` instead of being silently swallowed by
    /// `get`'s `decode → None` path.
    #[tokio::test]
    async fn counter_value_err_on_truncated_file() {
        let cache = fresh_cache();
        let path = cache.path_for("spend:k");
        std::fs::write(&path, b"\x01\x02\x03").expect("write truncated counter");
        assert!(
            cache.try_counter_value("spend:k").await.is_err(),
            "truncated counter file must fail closed, not read as absent"
        );
        // `incr` and `try_counter_value` must agree on the corrupt signal:
        // both treat the file as unusable rather than one silently resetting it.
        assert!(
            cache.incr("spend:k", 1.0, None).await.is_err(),
            "incr must fail closed on the same truncated file"
        );
    }

    /// A non-UTF-8 body (invalid continuation bytes) must also fail closed
    /// via the overridden `try_counter_value`. This pins the
    /// previously-unasserted `read_entry_at` Corrupt branch.
    #[tokio::test]
    async fn counter_value_err_on_non_utf8_body() {
        let cache = fresh_cache();
        let path = cache.path_for("spend:k");
        let mut data = NO_EXPIRY.to_le_bytes().to_vec();
        data.extend_from_slice(&[0xff, 0xfe, 0xfd, 0x80]);
        std::fs::write(&path, &data).expect("write non-utf8 counter");
        assert!(
            cache.try_counter_value("spend:k").await.is_err(),
            "non-utf8 counter body must fail closed, not read as absent"
        );
        assert!(
            cache.incr("spend:k", 1.0, None).await.is_err(),
            "incr must fail closed on the same non-utf8 file"
        );
    }

    // ---- HIGH 3: counter TTL is a fixed window, not sliding ---------------

    /// `incr` with a TTL must stamp the window only on the first write;
    /// subsequent increments reuse the original expiry (matching
    /// `InMemoryCache`). Otherwise an RPM counter under sustained load never
    /// expires because each request bumps the expiry forward.
    #[tokio::test]
    async fn incr_preserves_fixed_window_expiry() {
        let cache = fresh_cache();
        let ttl = Duration::from_secs(60);

        let t1a = now_millis();
        cache
            .incr("rpm:k", 1.0, Some(ttl))
            .await
            .expect("first incr");
        let t1b = now_millis();

        // Sleep long enough that a sliding window would observably move the
        // expiry forward, but short enough to stay well inside the 60s window.
        tokio::time::sleep(Duration::from_secs(2)).await;

        let t2a = now_millis();
        cache
            .incr("rpm:k", 1.0, Some(ttl))
            .await
            .expect("second incr");
        let t2b = now_millis();

        let stored = read_stored_expiry(&cache, "rpm:k");
        // The stored expiry should be anchored to the FIRST write's window...
        let d_from_first = (stored as i64 - (t1a + 60_000) as i64).abs();
        // ...and NOT the second write's window (which a sliding window would).
        let d_from_second = (stored as i64 - (t2a + 60_000) as i64).abs();
        assert!(
            d_from_first < d_from_second,
            "expiry should match the first write's window, not the second \
             (d_first={d_from_first}ms, d_second={d_from_second}ms)"
        );
        // Sanity: the stored expiry sits within the first window's bounds.
        assert!(
            stored >= t1a.saturating_add(60_000) && stored <= t2b.saturating_add(60_000),
            "stored expiry {stored} outside plausible fixed-window range"
        );
        let _ = t1b;
    }

    /// `incr_if_below`'s allowed path must also preserve the fixed window.
    #[tokio::test]
    async fn incr_if_below_preserves_fixed_window_expiry() {
        let cache = fresh_cache();
        let ttl = Duration::from_secs(60);
        let t1a = now_millis();
        cache
            .incr_if_below("rpm:k", 1.0, 3.0, Some(ttl))
            .await
            .expect("first incr");
        tokio::time::sleep(Duration::from_secs(2)).await;
        let t2a = now_millis();
        cache
            .incr_if_below("rpm:k", 1.0, 3.0, Some(ttl))
            .await
            .expect("second incr");
        let stored = read_stored_expiry(&cache, "rpm:k");
        let d_from_first = (stored as i64 - (t1a + 60_000) as i64).abs();
        let d_from_second = (stored as i64 - (t2a + 60_000) as i64).abs();
        assert!(
            d_from_first < d_from_second,
            "incr_if_below should preserve the first window \
             (d_first={d_from_first}ms, d_second={d_from_second}ms)"
        );
    }

    /// A counter that already has an expiry keeps it even when a later `incr`
    /// passes `None` TTL (the None must not erase the existing window).
    #[tokio::test]
    async fn incr_with_none_ttl_keeps_existing_window() {
        let cache = fresh_cache();
        let ttl = Duration::from_secs(60);
        let t1 = now_millis();
        cache
            .incr("rpm:k", 1.0, Some(ttl))
            .await
            .expect("first incr");
        let stored_after_first = read_stored_expiry(&cache, "rpm:k");
        cache.incr("rpm:k", 1.0, None).await.expect("second incr");
        let stored_after_second = read_stored_expiry(&cache, "rpm:k");
        assert_eq!(
            stored_after_first, stored_after_second,
            "a None TTL on a later incr must not change the existing window"
        );
        let _ = t1;
    }

    // ---- MEDIUM: huge TTL must not wrap `as u64` to a near-zero expiry -------

    /// Read just the stamped expiry header (epoch millis) for `key`, or `None`
    /// if the file is absent.
    fn read_stored_expiry_opt(cache: &DiskCache, key: &str) -> Option<u64> {
        let path = cache.path_for(key);
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < HEADER_LEN {
            return None;
        }
        let mut header = [0u8; HEADER_LEN];
        header.copy_from_slice(&bytes[..HEADER_LEN]);
        Some(u64::from_le_bytes(header))
    }

    /// `set` with an absurd caller TTL must not wrap the stamped expiry to a
    /// near-zero timestamp. `Duration::from_secs(u64::MAX)` has
    /// `as_millis() == u128::MAX`, and an unclamped `as u64` cast would wrap
    /// the low 64 bits — yielding an expiry within seconds of `now` instead of
    /// the intended "very long". The clamp pins the stamped expiry to a
    /// far-future timestamp.
    #[tokio::test]
    async fn set_huge_ttl_does_not_wrap_to_near_zero() {
        let cache = fresh_cache();
        cache
            .set("k", "v".to_string(), Some(Duration::from_secs(u64::MAX)))
            .await;
        let expiry = read_stored_expiry_opt(&cache, "k").expect("entry written");
        let now = now_millis();
        assert!(
            expiry == NO_EXPIRY || expiry.saturating_sub(now) > 60_000,
            "huge TTL must not wrap to a near-zero expiry; \
             stamped expiry={expiry}, now={now}"
        );
    }

    /// Same contract for `incr`: the stamped counter expiry must be far in the
    /// future, not wrapped to ~384 ms (the concrete wrap value for
    /// `Duration::from_secs(u64::MAX)`).
    #[tokio::test]
    async fn incr_huge_ttl_does_not_wrap_to_near_zero() {
        let cache = fresh_cache();
        cache
            .incr("c", 1.0, Some(Duration::from_secs(u64::MAX)))
            .await
            .expect("ok");
        let expiry = read_stored_expiry_opt(&cache, "c").expect("counter written");
        let now = now_millis();
        assert!(
            expiry == NO_EXPIRY || expiry.saturating_sub(now) > 60_000,
            "huge TTL must not wrap to a near-zero expiry; \
             stamped expiry={expiry}, now={now}"
        );
    }

    /// Same contract for `incr_if_below`'s allowed path.
    #[tokio::test]
    async fn incr_if_below_huge_ttl_does_not_wrap_to_near_zero() {
        let cache = fresh_cache();
        cache
            .incr_if_below("c", 1.0, 3.0, Some(Duration::from_secs(u64::MAX)))
            .await
            .expect("ok");
        let expiry = read_stored_expiry_opt(&cache, "c").expect("counter written");
        let now = now_millis();
        assert!(
            expiry == NO_EXPIRY || expiry.saturating_sub(now) > 60_000,
            "huge TTL must not wrap to a near-zero expiry; \
             stamped expiry={expiry}, now={now}"
        );
    }

    // ---- LOW: constructor GC of orphaned .tmp files -----------------------

    /// A `.tmp` file left behind by a previous crashed process must be removed
    /// when a new `DiskCache` is constructed, so repeated crashes do not
    /// accumulate stale temp files under `base_dir`.
    #[tokio::test]
    async fn constructor_gc_removes_orphaned_tmp_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_path_buf();
        // Simulate a crashed writer: a live entry plus two orphaned .tmp files.
        {
            let cache = DiskCache::new(path.clone()).expect("cache");
            cache.set("live", "v".to_string(), None).await;
        }
        let live_path = DiskCache::new(path.clone())
            .expect("cache")
            .path_for("live");
        let tmp1 = tmp_path_for(&live_path);
        let tmp2 = {
            let mut s = live_path.clone().into_os_string();
            s.push(".999999.123456789.tmp");
            PathBuf::from(s)
        };
        std::fs::write(&tmp1, b"stale").expect("write tmp1");
        std::fs::write(&tmp2, b"stale").expect("write tmp2");
        assert!(tmp1.exists() && tmp2.exists());

        // Constructing a new cache over the same dir must GC the .tmp files.
        let cache = DiskCache::new(path.clone()).expect("cache2");
        assert!(!tmp1.exists(), "orphaned tmp1 must be removed on construct");
        assert!(!tmp2.exists(), "orphaned tmp2 must be removed on construct");
        // ... and must leave the live entry intact.
        assert_eq!(cache.get("live").await.as_deref(), Some("v"));
    }

    // ---- MEDIUM: DiskCache::incr rejects non-finite deltas -----------------

    /// `incr` with a non-finite delta (NaN, ±∞) must reject with `Err` and
    /// leave the counter unchanged — mirroring `InMemoryCache::incr`. Without
    /// the guard, `current + by = NaN` is serialized via `to_string()` →
    /// `"NaN"`, which `f64::from_str` happily parses back, poisoning the
    /// counter so every subsequent `incr_if_below` evaluates
    /// `NaN <= ceiling == false` → `Rejected` (bricking the scope).
    #[tokio::test]
    async fn incr_rejects_nan_delta_disk() {
        let cache = fresh_cache();
        let _ = cache.incr("spend:k", 1.0, None).await.unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let res = cache.incr("spend:k", bad, None).await;
            assert!(
                res.is_err(),
                "disk incr must reject non-finite delta ({bad:?}), got {res:?}"
            );
        }
        // The counter must be unchanged (not poisoned) after the rejected
        // writes — `counter_value` reads back the original value, and a
        // legitimate reservation on the same key must still succeed.
        assert_eq!(
            cache.counter_value("spend:k").await,
            Some(1.0),
            "rejected non-finite delta must not mutate the disk counter"
        );
        let res = cache.incr_if_below("spend:k", 1.0, 10.0, None).await;
        assert!(
            matches!(res, Ok(IncrResult::Allowed(2.0))),
            "disk scope must remain usable after a rejected non-finite incr, got {res:?}"
        );
    }

    /// `incr_if_below` with a non-finite delta or ceiling must return `Err`
    /// (matching the trait default), not `Ok(Rejected(...))`. A NaN delta
    /// would compute `NaN <= ceiling == false` → a silent `Rejected`; a NaN
    /// ceiling would reject all traffic. Callers that check `is_err()` to
    /// detect invalid input must see the error.
    #[tokio::test]
    async fn incr_if_below_rejects_non_finite_delta_disk_returns_err() {
        let cache = fresh_cache();
        let _ = cache.incr("rpm:k", 1.0, None).await.unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let res = cache.incr_if_below("rpm:k", bad, 3.0, None).await;
            assert!(
                res.is_err(),
                "disk incr_if_below must reject non-finite delta ({bad:?}), got {res:?}"
            );
        }
        // Non-finite ceiling must also be rejected (would silently reject all
        // traffic instead of surfacing a config error).
        let res = cache.incr_if_below("rpm:k", 1.0, f64::NAN, None).await;
        assert!(
            res.is_err(),
            "disk incr_if_below must reject non-finite ceiling, got {res:?}"
        );
        // The counter must be unchanged after the rejected writes.
        assert_eq!(cache.counter_value("rpm:k").await, Some(1.0));
    }

    // ---- LOW: constructor GC reclaims expired entries ----------------------

    /// Entries that expire and are never re-read stay on disk indefinitely
    /// (the read path only lazily deletes on hit). Constructing a new
    /// `DiskCache` over the same dir must reclaim those expired entry files so
    /// a long-running single-instance deployment does not leak disk space
    /// across restarts. A live (non-expired) entry must be left intact.
    #[tokio::test]
    async fn expired_entries_reclaimed_on_restart() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_path_buf();
        let expired_path = {
            let cache = DiskCache::new(path.clone()).expect("cache");
            cache
                .set("expired", "v".to_string(), Some(Duration::from_millis(50)))
                .await;
            cache.path_for("expired")
        };
        let live_path = {
            let cache = DiskCache::new(path.clone()).expect("cache");
            cache
                .set("live", "v".to_string(), Some(Duration::from_secs(3600)))
                .await;
            cache.path_for("live")
        };
        assert!(expired_path.exists() && live_path.exists());

        // Let the short TTL elapse so "expired" is past its window, then
        // construct a fresh cache — the expired entry file must be gone while
        // the live one survives.
        tokio::time::sleep(Duration::from_millis(120)).await;
        let cache = DiskCache::new(path.clone()).expect("cache2");
        assert!(
            !expired_path.exists(),
            "expired entry file must be reclaimed on construct"
        );
        assert!(
            live_path.exists(),
            "live entry file must survive the expired-entry sweep"
        );
        assert_eq!(cache.get("live").await.as_deref(), Some("v"));
        assert!(cache.get("expired").await.is_none());
    }
}
