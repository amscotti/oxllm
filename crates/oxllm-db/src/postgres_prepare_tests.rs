//! Live PostgreSQL regressions for coordinated online spend preparation.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use sqlx::Row;

use crate::migrate::OXLLM_MIGRATE_LOCK_KEY;
use crate::postgres_prepare::PREPARATION_APPLICATION;
use crate::release_tests::{PostgresFixture, fixture_migration_lock, load_v1};
use crate::{Database, PostgresPreparationTimedOut, SCHEMA_VERSION, SpendEventIndexNotReady};

async fn fixture() -> Result<Option<PostgresFixture>> {
    let Ok(url) = std::env::var("OXLLM_TEST_POSTGRES_URL") else {
        eprintln!("skipping: OXLLM_TEST_POSTGRES_URL not set");
        return Ok(None);
    };
    if !url.starts_with("postgres://") && !url.starts_with("postgresql://") {
        return Err(anyhow!(
            "PostgreSQL preparation tests require the disposable PostgreSQL fixture"
        ));
    }
    Ok(Some(PostgresFixture::new(&url, 8).await))
}

async fn stage_columns(db: &Database) -> Result<()> {
    let mut conn = db.pool().acquire().await?;
    conn.close_on_drop();
    db.prepare_spend_columns(
        &mut conn,
        Duration::from_secs(1),
        Instant::now(),
        Duration::from_secs(5),
    )
    .await?;
    conn.close().await?;
    Ok(())
}

async fn builder_pid(db: &Database) -> Result<i64> {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let row: Option<(i64,)> = sqlx::query_as(
                "SELECT CAST(pid AS BIGINT) FROM pg_stat_activity \
                 WHERE application_name = $1 || ':' || md5(current_database() || ':' || current_schema()) AND state = 'active' \
                   AND position('CREATE UNIQUE INDEX CONCURRENTLY idx_spend_logs_event_id' IN query) = 1 \
                 ORDER BY backend_start DESC LIMIT 1",
            ).bind(PREPARATION_APPLICATION).fetch_optional(db.pool()).await?;
            if let Some((pid,)) = row {
                return Ok::<_, anyhow::Error>(pid);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }).await.map_err(|_| anyhow!("online index builder did not enter its blocked build"))?
}

async fn backend_closed(db: &Database, pid: i64) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(6), async {
        loop {
            let count: (i64,) =
                sqlx::query_as("SELECT COUNT(*) FROM pg_stat_activity WHERE pid = $1")
                    .bind(pid)
                    .fetch_one(db.pool())
                    .await?;
            if count.0 == 0 {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .map_err(|_| anyhow!("canceled preparation backend/session lock was not closed"))?
}

async fn settings(db: &Database) -> Result<(String, String, String, String)> {
    Ok(sqlx::query_as(
        "SELECT current_setting('lock_timeout'), current_setting('statement_timeout'), \
         current_setting('client_connection_check_interval'), current_setting('application_name')",
    )
    .fetch_one(db.pool())
    .await?)
}

#[tokio::test]
async fn postgres_online_columns_commit_without_publishing_schema_readiness() -> Result<()> {
    let Some(fixture) = fixture().await? else {
        return Ok(());
    };
    let db = &fixture.db;
    load_v1(db).await;
    let mut conn = db.pool().acquire().await?;
    conn.close_on_drop();
    let pid: (i64,) = sqlx::query_as("SELECT CAST(pg_backend_pid() AS BIGINT)")
        .fetch_one(&mut *conn)
        .await?;
    db.prepare_spend_columns(
        &mut conn,
        Duration::from_secs(1),
        Instant::now(),
        Duration::from_secs(5),
    )
    .await?;
    assert_eq!(db.get_schema_version().await?, 1);
    let shape: (String, String) = sqlx::query_as(
        "SELECT CAST(data_type AS TEXT), CAST(is_nullable AS TEXT) FROM information_schema.columns \
         WHERE table_schema = current_schema() AND table_name = 'spend_logs' AND column_name = 'event_id'",
    ).fetch_one(db.pool()).await?;
    assert_eq!(shape, ("text".into(), "YES".into()));
    let locks: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_locks WHERE pid = $1 AND relation = 'spend_logs'::regclass \
         AND mode = 'AccessExclusiveLock' AND granted",
    )
    .bind(pid.0)
    .fetch_one(db.pool())
    .await?;
    assert_eq!(locks.0, 0, "the ALTER commit must precede historical scans");
    sqlx::query("INSERT INTO spend_logs (timestamp, model, provider, cost) VALUES ('2026-07-31T12:00:00Z', 'old', 'old', 0.125)")
        .execute(db.pool()).await?;
    conn.close().await?;
    db.migrate().await?;
    assert_eq!(db.get_schema_version().await?, SCHEMA_VERSION);
    assert_eq!(db.count_spend(None, None).await?, 4);
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn postgres_online_peers_wait_for_preparation_while_old_writers_progress() -> Result<()> {
    let Some(fixture) = fixture().await? else {
        return Ok(());
    };
    let db = &fixture.db;
    load_v1(db).await;
    stage_columns(db).await?;
    let mut blocker = db.pool().begin().await?;
    // This lock blocks concurrent index creation but remains compatible with
    // ordinary old-replica reads and writes. Columns already committed above.
    sqlx::query("LOCK TABLE spend_logs IN SHARE UPDATE EXCLUSIVE MODE")
        .execute(&mut *blocker)
        .await?;
    let owner_db = Arc::clone(db);
    let owner = tokio::spawn(async move {
        owner_db
            .migrate_with_timeouts(Duration::from_millis(40), Duration::from_secs(5))
            .await
    });
    let pid = builder_pid(db).await?;
    let exclusive: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_locks WHERE pid = $1 AND relation = 'spend_logs'::regclass \
         AND mode = 'AccessExclusiveLock' AND granted",
    )
    .bind(pid)
    .fetch_one(db.pool())
    .await?;
    assert_eq!(exclusive.0, 0);
    assert_eq!(
        db.get_schema_version().await?,
        1,
        "readiness must wait for a valid index"
    );
    let peer_db = Arc::clone(db);
    let peer = tokio::spawn(async move {
        peer_db
            .migrate_with_timeouts(Duration::from_millis(40), Duration::from_secs(5))
            .await
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(!owner.is_finished());
    assert!(
        !peer.is_finished(),
        "peer must use preparation budget, not the 40ms DDL bound"
    );
    let waiting: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid \
         WHERE l.locktype = 'advisory' AND l.classid::bigint = $1 AND l.objid::bigint = $2 \
           AND NOT l.granted AND a.application_name = $3 || ':' || md5(current_database() || ':' || current_schema())",
    )
    .bind(OXLLM_MIGRATE_LOCK_KEY >> 32)
    .bind(OXLLM_MIGRATE_LOCK_KEY & 0xffff_ffff)
    .bind(PREPARATION_APPLICATION)
    .fetch_one(db.pool())
    .await?;
    assert_eq!(
        waiting.0, 0,
        "standby must not retain a blocked SQL snapshot/advisory waiter"
    );
    sqlx::query(
        "INSERT INTO spend_logs (timestamp, request_id, model, provider, cost, tags) \
         VALUES ('2026-07-31T23:59:59.999999999Z', 'online-old-writer', 'old', 'old', 0.125, '[\"live\",\"live\"]')",
    ).execute(db.pool()).await?;
    let readers: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_logs")
        .fetch_one(db.pool())
        .await?;
    assert_eq!(readers.0, 4);
    blocker.rollback().await?;
    tokio::time::timeout(Duration::from_secs(6), owner).await???;
    tokio::time::timeout(Duration::from_secs(6), peer).await???;
    assert_eq!(db.get_schema_version().await?, SCHEMA_VERSION);
    assert_eq!(
        db.count_spend_filtered(&crate::SpendFilter {
            tag_name: Some("live".into()),
            ..Default::default()
        })
        .await?,
        1
    );
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn postgres_online_cancel_closes_owner_without_pool_setting_or_lock_leaks() -> Result<()> {
    let Some(fixture) = fixture().await? else {
        return Ok(());
    };
    let db = &fixture.db;
    load_v1(db).await;
    stage_columns(db).await?;
    let baseline = settings(db).await?;
    let mut blocker = db.pool().begin().await?;
    sqlx::query("LOCK TABLE spend_logs IN SHARE UPDATE EXCLUSIVE MODE")
        .execute(&mut *blocker)
        .await?;
    let owner_db = Arc::clone(db);
    let owner = tokio::spawn(async move { owner_db.migrate().await });
    let pid = builder_pid(db).await?;
    owner.abort();
    assert!(owner.await.is_err_and(|error| error.is_cancelled()));
    backend_closed(db, pid).await?;
    let locks: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM pg_locks WHERE pid = $1 AND locktype = 'advisory'")
            .bind(pid)
            .fetch_one(db.pool())
            .await?;
    assert_eq!(locks.0, 0);
    assert_eq!(settings(db).await?, baseline);
    assert_eq!(db.get_schema_version().await?, 1);
    let retained: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_logs")
        .fetch_one(db.pool())
        .await?;
    assert_eq!(retained.0, 3);
    blocker.rollback().await?;
    // Cancellation while waiting for the table lock creates no index artifact.
    db.migrate().await?;
    assert_eq!(db.get_schema_version().await?, SCHEMA_VERSION);
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn postgres_online_explicit_lock_bounds_are_not_extended() -> Result<()> {
    let Some(fixture) = fixture().await? else {
        return Ok(());
    };
    let db = &fixture.db;
    load_v1(db).await;
    let baseline = settings(db).await?;
    let blocker = fixture_migration_lock(db).await?;
    let started = Instant::now();
    let error = db
        .migrate_with_lock_timeout(Duration::from_millis(40))
        .await
        .err()
        .ok_or_else(|| anyhow!("explicit preparation bound unexpectedly succeeded"))?;
    assert!(started.elapsed() < Duration::from_secs(1));
    assert!(
        error
            .downcast_ref::<PostgresPreparationTimedOut>()
            .is_some()
            || error.to_string().contains("migration advisory lock")
    );
    tokio::time::sleep(Duration::from_millis(150)).await;
    let waiting: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid \
         WHERE l.locktype = 'advisory' AND l.classid::bigint = $1 AND l.objid::bigint = $2 \
           AND NOT l.granted AND a.application_name = $3 || ':' || md5(current_database() || ':' || current_schema())",
    )
    .bind(OXLLM_MIGRATE_LOCK_KEY >> 32)
    .bind(OXLLM_MIGRATE_LOCK_KEY & 0xffff_ffff)
    .bind(PREPARATION_APPLICATION)
    .fetch_one(db.pool())
    .await?;
    assert_eq!(
        waiting.0, 0,
        "timed-out coordinator must not leave a later lock owner behind"
    );
    assert_eq!(settings(db).await?, baseline);
    blocker.close().await?;
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn postgres_online_corrupt_indexes_and_event_ids_are_preserved() -> Result<()> {
    let Some(fixture) = fixture().await? else {
        return Ok(());
    };
    let db = &fixture.db;
    load_v1(db).await;
    stage_columns(db).await?;
    sqlx::query("CREATE INDEX idx_spend_logs_event_id ON spend_logs(event_id)")
        .execute(db.pool())
        .await?;
    let error = db
        .migrate()
        .await
        .err()
        .ok_or_else(|| anyhow!("wrong index was accepted"))?;
    assert!(error.downcast_ref::<SpendEventIndexNotReady>().is_some());
    let shape: (i64,) = sqlx::query_as(
        "SELECT CAST(CASE WHEN indisunique THEN 1 ELSE 0 END AS BIGINT) FROM pg_index \
         WHERE indexrelid = 'idx_spend_logs_event_id'::regclass",
    )
    .fetch_one(db.pool())
    .await?;
    assert_eq!(shape.0, 0);
    // Remove only this test's deliberately wrong index, then verify corruption
    // in event identities is rejected without deleting or rewriting any row.
    sqlx::query("DROP INDEX idx_spend_logs_event_id")
        .execute(db.pool())
        .await?;
    sqlx::query("UPDATE spend_logs SET event_id = 'duplicate-event'")
        .execute(db.pool())
        .await?;
    let error = db
        .migrate()
        .await
        .err()
        .ok_or_else(|| anyhow!("duplicate identities were accepted"))?;
    assert!(error.downcast_ref::<SpendEventIndexNotReady>().is_some());
    assert_eq!(db.get_schema_version().await?, 1);
    let retained: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM spend_logs WHERE event_id = 'duplicate-event'")
            .fetch_one(db.pool())
            .await?;
    assert_eq!(retained.0, 3);
    let invalid = sqlx::query(
        "SELECT CAST(CASE WHEN indisvalid THEN 1 ELSE 0 END AS BIGINT) AS ready FROM pg_index \
         WHERE indexrelid = 'idx_spend_logs_event_id'::regclass",
    )
    .fetch_one(db.pool())
    .await?;
    assert_eq!(invalid.try_get::<i64, _>("ready")?, 0);
    let retry = db
        .migrate()
        .await
        .err()
        .ok_or_else(|| anyhow!("invalid index was silently replaced"))?;
    assert!(retry.downcast_ref::<SpendEventIndexNotReady>().is_some());
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn postgres_online_fresh_and_already_ready_paths_remain_cheap() -> Result<()> {
    let Some(fixture) = fixture().await? else {
        return Ok(());
    };
    let db = &fixture.db;
    db.migrate().await?;
    assert_eq!(db.get_schema_version().await?, SCHEMA_VERSION);
    let coordinator = fixture_migration_lock(db).await?;
    let mut blocker = db.pool().begin().await?;
    sqlx::query("LOCK TABLE spend_logs IN SHARE UPDATE EXCLUSIVE MODE")
        .execute(&mut *blocker)
        .await?;
    let ready_db = Arc::clone(db);
    let ready = tokio::spawn(async move {
        ready_db
            .migrate_with_timeouts(Duration::from_millis(40), Duration::from_secs(5))
            .await
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !ready.is_finished(),
        "ready migrator must coordinate before any transaction"
    );
    coordinator.close().await?;
    tokio::time::timeout(Duration::from_secs(6), ready).await???;
    assert_eq!(db.get_schema_version().await?, SCHEMA_VERSION);
    blocker.rollback().await?;
    fixture.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn postgres_online_fresh_ready_and_legacy_schemas_progress_concurrently() -> Result<()> {
    let Some(legacy) = fixture().await? else {
        return Ok(());
    };
    let Some(fresh) = fixture().await? else {
        return Ok(());
    };
    let Some(ready) = fixture().await? else {
        return Ok(());
    };
    ready.db.migrate().await?;
    load_v1(&legacy.db).await;
    stage_columns(&legacy.db).await?;
    let mut blocker = legacy.db.pool().begin().await?;
    sqlx::query("LOCK TABLE spend_logs IN SHARE UPDATE EXCLUSIVE MODE")
        .execute(&mut *blocker)
        .await?;
    let owner_db = Arc::clone(&legacy.db);
    let owner = tokio::spawn(async move {
        owner_db
            .migrate_with_timeouts(Duration::from_millis(40), Duration::from_secs(10))
            .await
    });
    let owner_pid = builder_pid(&legacy.db).await?;
    let fresh_db = Arc::clone(&fresh.db);
    let fresh_task = tokio::spawn(async move {
        fresh_db
            .migrate_with_timeouts(Duration::from_millis(40), Duration::from_secs(10))
            .await
    });
    let ready_db = Arc::clone(&ready.db);
    let ready_task = tokio::spawn(async move {
        ready_db
            .migrate_with_timeouts(Duration::from_millis(40), Duration::from_secs(10))
            .await
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(!fresh_task.is_finished());
    assert!(!ready_task.is_finished());
    for db in [&fresh.db, &ready.db] {
        let standby = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let standby: (i64, i64) = sqlx::query_as(
                    "SELECT COUNT(*), COUNT(a.backend_xmin) FROM pg_stat_activity a \
                     WHERE a.application_name = $1 || ':' || md5(current_database() || ':' || current_schema()) \
                       AND a.state = 'idle'",
                ).bind(PREPARATION_APPLICATION).fetch_one(db.pool()).await?;
                if standby.0 != 0 { return Ok::<_, anyhow::Error>(standby); }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }).await??;
        assert_eq!(standby.0, 1, "this schema must own its standby session");
        assert_eq!(
            standby.1, 0,
            "standby must release snapshots between nonblocking attempts"
        );
    }
    let exclusive: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_locks WHERE pid = $1 AND mode = 'AccessExclusiveLock' AND granted",
    )
    .bind(owner_pid)
    .fetch_one(legacy.db.pool())
    .await?;
    assert_eq!(exclusive.0, 0);
    let reader: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_logs")
        .fetch_one(legacy.db.pool())
        .await?;
    assert_eq!(reader.0, 3);
    blocker.rollback().await?;
    tokio::time::timeout(Duration::from_secs(12), owner).await???;
    tokio::time::timeout(Duration::from_secs(12), fresh_task).await???;
    tokio::time::timeout(Duration::from_secs(12), ready_task).await???;
    for db in [&legacy.db, &fresh.db, &ready.db] {
        assert_eq!(db.get_schema_version().await?, SCHEMA_VERSION);
    }
    legacy.cleanup().await;
    fresh.cleanup().await;
    ready.cleanup().await;
    Ok(())
}
