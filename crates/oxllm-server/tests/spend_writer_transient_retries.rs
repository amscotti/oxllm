//! Optional PostgreSQL regressions for spend-batch rollback/retry boundaries.
//! Each case owns a fresh schema on the isolated fixture selected by
//! OXLLM_TEST_POSTGRES_URL. Schema cleanup also runs on assertion panics.

#![cfg(feature = "db")]

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use futures_util::FutureExt;
use oxllm_db::{CustomerRecord, Database, TagRecord};
use oxllm_server::spend_writer::{SpendEntry, spawn_with_shutdown};
use serde_json::json;

async fn with_postgres_fixture<F, Fut>(case: F) -> Result<()>
where
    F: FnOnce(Arc<Database>) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let Ok(raw_url) = std::env::var("OXLLM_TEST_POSTGRES_URL") else {
        eprintln!("skipping PostgreSQL spend retry regression: OXLLM_TEST_POSTGRES_URL is unset");
        return Ok(());
    };
    let mut scoped_url = url::Url::parse(&raw_url)?;
    ensure!(
        matches!(scoped_url.scheme(), "postgres" | "postgresql"),
        "spend retry regressions require the isolated PostgreSQL fixture"
    );
    let admin = Database::connect(&raw_url).await?;
    let schema = format!("spend_retry_{}", uuid::Uuid::new_v4().simple());
    sqlx::query(&format!("CREATE SCHEMA {schema}"))
        .execute(admin.pool())
        .await?;

    let mut database = None;
    let outcome = std::panic::AssertUnwindSafe(async {
        let mut options = String::new();
        let mut parameters = Vec::new();
        for (key, value) in scoped_url.query_pairs() {
            if key == "options" {
                options.push_str(&value);
                options.push(' ');
            } else {
                parameters.push((key.into_owned(), value.into_owned()));
            }
        }
        // No public-schema fallback: migrations, sequences and triggers must
        // remain inside this test's schema. Preserve other fixture options.
        options.push_str(&format!(
            "-c search_path={schema} -c statement_timeout=2000 -c lock_timeout=100"
        ));
        scoped_url
            .query_pairs_mut()
            .clear()
            .extend_pairs(parameters)
            .append_pair("options", &options);
        let db = Database::connect(scoped_url.as_str()).await?;
        database = Some(Arc::clone(&db));
        let current: (String,) = sqlx::query_as("SELECT CAST(current_schema() AS TEXT)")
            .fetch_one(db.pool())
            .await?;
        ensure!(
            current.0 == schema,
            "test search_path did not select its schema"
        );
        db.migrate().await?;
        case(db).await
    })
    .catch_unwind()
    .await;

    if let Some(db) = database {
        db.pool().close().await;
    }
    let cleanup = sqlx::query(&format!("DROP SCHEMA {schema} CASCADE"))
        .execute(admin.pool())
        .await;
    admin.pool().close().await;
    cleanup.context("cleaning up isolated spend retry schema")?;
    match outcome {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

fn dropped_count() -> u64 {
    let _ = oxllm_server::metrics::install();
    oxllm_server::metrics::render()
        .lines()
        .find(|line| {
            line.starts_with("oxllm_spend_entries_dropped_total{")
                && line.contains("reason=\"batch_error\"")
        })
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|count| count.parse().ok())
        .unwrap_or(0)
}

async fn create_counter_scopes(db: &Database) -> Result<()> {
    let now = oxllm_db::now_iso();
    db.create_customer(&CustomerRecord {
        end_user_id: "retry-customer".into(),
        alias: None,
        email: None,
        max_budget: None,
        spend: 0.0,
        budget_duration: None,
        blocked: false,
        model_blocklist: None,
        created_at: now.clone(),
        updated_at: now.clone(),
    })
    .await?;
    db.create_tag(&TagRecord {
        name: "retry-tag".into(),
        description: None,
        models: None,
        model_info: None,
        spend: Some(0.0),
        max_budget: None,
        created_at: now.clone(),
        updated_at: now,
    })
    .await
}

async fn install_rollback_trigger(db: &Database, code: &str, failures: u32) -> Result<()> {
    ensure!(
        matches!(code, "40001" | "40P01" | "55P03" | "23514"),
        "test must use a fixed SQLSTATE"
    );
    sqlx::raw_sql(&format!(
        "CREATE SEQUENCE retry_attempts;
         CREATE FUNCTION reject_counter_update() RETURNS trigger LANGUAGE plpgsql AS $body$
         BEGIN
             IF nextval('retry_attempts') <= {failures} THEN
                 RAISE EXCEPTION USING ERRCODE = '{code}', MESSAGE = 'scripted rollback';
             END IF;
             RETURN NULL;
         END;
         $body$;
         CREATE TRIGGER reject_counter_update AFTER UPDATE ON tags
             FOR EACH STATEMENT EXECUTE FUNCTION reject_counter_update();"
    ))
    .execute(db.pool())
    .await?;
    Ok(())
}

async fn flush_two_events_on_shutdown(db: Arc<Database>) -> Result<()> {
    let entry: SpendEntry = serde_json::from_value(json!({
        "timestamp": "2026-01-01T00:00:00Z",
        "request_id": "same-request-id",
        "model": "test-model",
        "provider": "test-provider",
        "cost": 0.25,
        "end_user_id": "retry-customer",
        "request_tags": "[\"retry-tag\"]",
    }))?;
    let (writer, handle) = spawn_with_shutdown(db, 3600, 8);
    // Both writes are buffered before yielding to the current-thread runtime,
    // so the shutdown drain must persist this as one two-event batch.
    writer.queue(entry.clone()).await;
    writer.queue(entry).await;
    tokio::time::timeout(Duration::from_secs(10), handle.shutdown())
        .await
        .context("spend retry shutdown must remain bounded")?;
    drop(writer);
    Ok(())
}

async fn assert_accounting(db: &Database, events: i64, cost: f64) -> Result<()> {
    let rows: (i64, i64) =
        sqlx::query_as("SELECT COUNT(*), COUNT(DISTINCT event_id) FROM spend_logs")
            .fetch_one(db.pool())
            .await?;
    ensure!(
        rows == (events, events),
        "expected unique metering events, got {rows:?}"
    );
    let memberships: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_log_tags")
        .fetch_one(db.pool())
        .await?;
    ensure!(
        memberships.0 == events,
        "tag memberships must be applied once per event"
    );
    let customer = db
        .get_customer("retry-customer")
        .await?
        .context("customer")?;
    let tag = db.get_tag("retry-tag").await?.context("tag")?;
    ensure!(customer.spend == cost, "customer was charged incorrectly");
    ensure!(tag.spend == Some(cost), "tag was charged incorrectly");
    Ok(())
}

#[tokio::test]
#[serial_test::serial]
async fn postgres_transient_spend_rollbacks_recover_without_double_accounting() -> Result<()> {
    for code in ["40001", "40P01", "55P03"] {
        with_postgres_fixture(|db| async move {
            create_counter_scopes(&db).await?;
            // Fail after inserting logs/memberships and updating the customer:
            // retrying must start a fresh transaction, not retain partial spend.
            install_rollback_trigger(&db, code, 1).await?;
            let drops = dropped_count();
            flush_two_events_on_shutdown(Arc::clone(&db)).await?;
            let attempts: (i64,) = sqlx::query_as("SELECT last_value FROM retry_attempts")
                .fetch_one(db.pool())
                .await?;
            ensure!(
                attempts.0 == 2,
                "{code} must retry exactly once, got {attempts:?}"
            );
            assert_accounting(&db, 2, 0.5).await?;
            ensure!(
                dropped_count() == drops,
                "successful recovery must not count drops"
            );
            Ok(())
        })
        .await?;
    }
    Ok(())
}

#[tokio::test]
#[serial_test::serial]
async fn postgres_transient_spend_retry_exhaustion_drops_only_once() -> Result<()> {
    with_postgres_fixture(|db| async move {
        create_counter_scopes(&db).await?;
        install_rollback_trigger(&db, "55P03", 100).await?;
        let drops = dropped_count();
        flush_two_events_on_shutdown(Arc::clone(&db)).await?;
        let attempts: (i64,) = sqlx::query_as("SELECT last_value FROM retry_attempts")
            .fetch_one(db.pool())
            .await?;
        ensure!(attempts.0 == 4, "retry budget must end after four attempts");
        assert_accounting(&db, 0, 0.0).await?;
        ensure!(
            dropped_count() - drops == 2,
            "exhaustion must count each event once"
        );
        Ok(())
    })
    .await
}

#[tokio::test]
#[serial_test::serial]
async fn postgres_permanent_spend_error_skips_batch_and_per_row_retries() -> Result<()> {
    with_postgres_fixture(|db| async move {
        create_counter_scopes(&db).await?;
        install_rollback_trigger(&db, "23514", 100).await?;
        let drops = dropped_count();
        flush_two_events_on_shutdown(Arc::clone(&db)).await?;
        let attempts: (i64,) = sqlx::query_as("SELECT last_value FROM retry_attempts")
            .fetch_one(db.pool())
            .await?;
        ensure!(
            attempts.0 == 1,
            "constraint errors must be terminal on the first attempt"
        );
        assert_accounting(&db, 0, 0.0).await?;
        ensure!(
            dropped_count() - drops == 2,
            "permanent errors must count each drop once"
        );
        Ok(())
    })
    .await
}

#[tokio::test]
#[serial_test::serial]
async fn postgres_commit_failure_never_retries_a_transient_looking_source() -> Result<()> {
    with_postgres_fixture(|db| async move {
        create_counter_scopes(&db).await?;
        sqlx::raw_sql(
            "CREATE SEQUENCE retry_attempts;
             CREATE FUNCTION reject_commit() RETURNS trigger LANGUAGE plpgsql AS $body$
             BEGIN
                 PERFORM nextval('retry_attempts');
                 RAISE EXCEPTION USING ERRCODE = '40001', MESSAGE = 'scripted commit failure';
             END;
             $body$;
             CREATE CONSTRAINT TRIGGER reject_commit AFTER INSERT ON spend_logs
                 DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_commit();",
        )
        .execute(db.pool())
        .await?;
        let drops = dropped_count();
        flush_two_events_on_shutdown(Arc::clone(&db)).await?;
        let attempts: (i64,) = sqlx::query_as("SELECT last_value FROM retry_attempts")
            .fetch_one(db.pool())
            .await?;
        ensure!(
            attempts.0 == 1,
            "a commit wrapper must bypass both retry paths"
        );
        assert_accounting(&db, 0, 0.0).await?;
        ensure!(
            dropped_count() - drops == 2,
            "commit failures must count each drop once"
        );
        Ok(())
    })
    .await
}
