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
async fn postgres_oversized_session_identity_cannot_poison_unrelated_spend_rows() -> Result<()> {
    with_postgres_fixture(|db| async move {
        create_counter_scopes(&db).await?;
        let long = (0..512u32).map(|value| format!("{:08x}", value.wrapping_mul(0x9e37_79b9))).collect::<String>();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-litellm-session-id", long.parse()?);
        let session = oxllm_server::pipeline::extract_session_id(&headers).context("session header")?;
        let entry: SpendEntry = serde_json::from_value(json!({
            "timestamp":"2026-10-01T00:00:00Z", "request_id":"bounded-session-request",
            "model":"test-model", "provider":"test-provider", "key_id":"long-session-owner",
            "session_id":long, "cost":0.25, "end_user_id":"retry-customer", "request_tags":"[\"retry-tag\"]"
        }))?;
        let mut other = entry.clone();
        other.request_id = Some("unrelated-session-request".into());
        other.key_id = Some("other-owner".into());
        other.session_id = Some("ordinary-session".into());
        let drops = dropped_count();
        let (writer, handle) = spawn_with_shutdown(db.clone(), 3600, 8);
        writer.queue(entry).await;
        writer.queue(other).await;
        tokio::time::timeout(Duration::from_secs(10), handle.shutdown()).await?;
        drop(writer);
        assert_accounting(&db, 2, 0.5).await?;
        let row = db.query_spend_by_request_id("bounded-session-request").await?.context("bounded row")?;
        ensure!(row.session_id.as_deref() == Some(&long[..256]), "stored session differs from routing identity");
        ensure!(row.session_id.as_deref() == Some(session.as_str()), "the writer must apply the same normalization as extraction");
        let other = db.query_spend_by_request_id("unrelated-session-request").await?.context("unrelated row")?;
        ensure!(other.session_id.as_deref() == Some("ordinary-session"), "unrelated row was changed");
        ensure!(dropped_count() == drops, "a normalized session must not drop its batch");
        Ok(())
    }).await
}

#[tokio::test]
#[serial_test::serial]
async fn postgres_oversized_end_user_identity_cannot_poison_or_misattribute_a_batch() -> Result<()>
{
    with_postgres_fixture(|db| async move {
        create_counter_scopes(&db).await?;
        let long = (0..512u32).map(|value| format!("{:08x}", value.wrapping_mul(0x9e37_79b9))).collect::<String>();
        let oversized: SpendEntry = serde_json::from_value(json!({
            "timestamp":"2026-10-01T00:00:00Z", "request_id":"oversized-end-user", "model":"test-model",
            "provider":"test-provider", "key_id":"first-owner", "end_user_id":long, "cost":0.25,
            "request_tags":"[\"retry-tag\"]"
        }))?;
        let mut ordinary = oversized.clone();
        ordinary.request_id = Some("ordinary-end-user".into());
        ordinary.key_id = Some("second-owner".into());
        ordinary.end_user_id = Some("retry-customer".into());
        let drops = dropped_count();
        let (writer, handle) = spawn_with_shutdown(db.clone(), 3600, 8);
        writer.queue(oversized).await;
        writer.queue(ordinary).await;
        tokio::time::timeout(Duration::from_secs(10), handle.shutdown()).await?;
        drop(writer);
        ensure!(db.count_spend(None, None).await? == 2, "invalid attribution must not discard either metering event");
        ensure!(db.total_spend(None, None).await? == 0.5, "billed costs must remain intact");
        let first = db.query_spend_by_request_id("oversized-end-user").await?.context("first row")?;
        ensure!(first.end_user_id.is_none(), "invalid identity must not be truncated into a different customer");
        let second = db.query_spend_by_request_id("ordinary-end-user").await?.context("unrelated row")?;
        ensure!(second.end_user_id.as_deref() == Some("retry-customer"));
        ensure!(db.get_customer("retry-customer").await?.context("customer")?.spend == 0.25, "only the ordinary customer's event is attributed");
        ensure!(db.get_tag("retry-tag").await?.context("tag")?.spend == Some(0.5), "tag counters must retain both events");
        ensure!(dropped_count() == drops);
        Ok(())
    }).await
}

#[tokio::test]
#[serial_test::serial]
async fn postgres_oversized_model_attribution_cannot_poison_unrelated_spend_rows() -> Result<()> {
    with_postgres_fixture(|db| async move {
        create_counter_scopes(&db).await?;
        let long = (0..512u32).map(|value| format!("{:08x}", value.wrapping_mul(0x9e37_79b9))).collect::<String>();
        let entry: SpendEntry = serde_json::from_value(json!({
            "timestamp":"2026-10-01T00:00:00Z", "request_id":"oversized-model", "model":long,
            "provider":"test-provider", "key_id":"first-owner", "end_user_id":"retry-customer", "cost":0.25,
            "request_tags":"[\"retry-tag\"]"
        }))?;
        let mut different = entry.clone();
        different.request_id = Some("different-model".into());
        different.model.push('x');
        let mut ordinary = entry.clone();
        ordinary.request_id = Some("ordinary-model".into());
        ordinary.model = "ordinary-model".into();
        let drops = dropped_count();
        let (writer, handle) = spawn_with_shutdown(db.clone(), 3600, 8);
        for entry in [entry, different, ordinary] { writer.queue(entry).await; }
        tokio::time::timeout(Duration::from_secs(10), handle.shutdown()).await?;
        drop(writer);
        assert_accounting(&db, 3, 0.75).await?;
        let first = db.query_spend_by_request_id("oversized-model").await?.context("first row")?;
        let different = db.query_spend_by_request_id("different-model").await?.context("different row")?;
        ensure!(first.model.len() <= 512 && first.model.contains("...#sha256:"), "oversized model attribution must be bounded and marked");
        ensure!(first.model != different.model, "distinct oversized models must not collapse to the same truncated prefix");
        ensure!(db.query_spend_by_request_id("ordinary-model").await?.context("ordinary row")?.model == "ordinary-model");
        ensure!(dropped_count() == drops);
        Ok(())
    }).await
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
