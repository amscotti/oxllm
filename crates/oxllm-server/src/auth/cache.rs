//! DB-authoritative versions for cached authentication snapshots.
//!
//! A delete/pub-sub message cannot fence a DB read that started before a
//! mutation and finishes its cache write afterwards. Each snapshot therefore
//! carries a content version of the complete DB record (including absence).
//! Reads validate that version against the shared DB, never a locally cached
//! generation. Fills also validate after the awaited cache write and retry if
//! the record changed. Lost invalidations and late DualCache backfills can
//! leave old bytes in a cache, but cannot make those bytes authoritative again.
//!
//! This deliberately costs one DB read on a warm lookup, two on a cold fill.
//! It needs no eviction-sensitive Redis counters, distributed leases, schema
//! changes, or post-commit writes to a second authority. Cache outages fall
//! back to the DB; inability to read the authority fails closed as a 503.

use std::future::Future;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

use super::{VK_DB_ERROR, VK_NEGATIVE_TTL, VK_POSITIVE_TTL, hash_key_hex};

const MAX_FILL_ATTEMPTS: usize = 3;

#[derive(Serialize, Deserialize)]
struct Snapshot<T> {
    /// Hash of the complete serialized Option<DB record>, not `updated_at`:
    /// writers may reuse a timestamp, and deleting a record must change it.
    version: String,
    value: Option<T>,
}

fn record_version<R: Serialize>(record: &Option<R>) -> Result<String, ()> {
    serde_json::to_vec(record)
        .map(|bytes| hash_key_hex(&bytes))
        .map_err(|error| {
            tracing::warn!(error = %error, "unable to version auth DB record; failing closed");
        })
}

async fn read_record<R, Read, ReadFuture>(cache_key: &str, read: &Read) -> Result<Option<R>, ()>
where
    Read: Fn() -> ReadFuture,
    ReadFuture: Future<Output = anyhow::Result<Option<R>>>,
{
    read().await.map_err(|error| {
        tracing::warn!(error = %error, cache_key, "auth DB version read failed; failing closed");
    })
}

/// Resolve one independently reloadable auth scope. Both positive snapshots
/// and missing/invalid-record tombstones are versioned. Legacy unversioned
/// entries are misses. `project` may reject an inactive/expired key with None.
pub(super) async fn load<R, T, Read, ReadFuture, Project>(
    state: &AppState,
    cache_key: &str,
    read: Read,
    project: Project,
) -> Result<Option<T>, ()>
where
    R: Serialize,
    T: Serialize + DeserializeOwned,
    Read: Fn() -> ReadFuture,
    ReadFuture: Future<Output = anyhow::Result<Option<R>>>,
    Project: Fn(R) -> Option<T>,
{
    let cached = state.cache.get(cache_key).await;
    if cached.as_deref() == Some(VK_DB_ERROR) {
        return Err(());
    }
    let mut record = match read_record(cache_key, &read).await {
        Ok(record) => record,
        Err(()) => {
            cache_db_error(state, cache_key).await;
            return Err(());
        }
    };

    let mut version = record_version(&record)?;
    if let Some(snapshot) = cached
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Snapshot<T>>(raw).ok())
    {
        if snapshot.version == version {
            return Ok(snapshot.value);
        }
    }

    for _ in 0..MAX_FILL_ATTEMPTS {
        let value = record.and_then(&project);
        let snapshot = Snapshot { version, value };
        let ttl = if snapshot.value.is_some() {
            VK_POSITIVE_TTL
        } else {
            VK_NEGATIVE_TTL
        };
        if let Ok(json) = serde_json::to_string(&snapshot) {
            state.cache.set(cache_key, json, Some(ttl)).await;
        }

        // A paused set may land AFTER a committed mutation's invalidation.
        // Do not return that old projection (or keep using its generation).
        // The next fill uses the newly observed record/version. Even if this
        // read fails, future hits cannot admit the old snapshot without a
        // successful fresh DB version check.
        record = match read_record(cache_key, &read).await {
            Ok(record) => record,
            Err(()) => {
                cache_db_error(state, cache_key).await;
                return Err(());
            }
        };
        version = record_version(&record)?;
        if version == snapshot.version {
            return Ok(snapshot.value);
        }
    }

    tracing::warn!(
        cache_key,
        "auth record changed repeatedly during cache fill; failing closed"
    );
    Err(())
}

async fn cache_db_error(state: &AppState, cache_key: &str) {
    state
        .cache
        .set(
            cache_key,
            VK_DB_ERROR.to_string(),
            Some(std::time::Duration::from_secs(1)),
        )
        .await;
}
