//! Behavioral release regressions shared by SQLite and the opt-in Postgres
//! CI suite. All data is synthetic; Postgres tests own isolated UUID schemas.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use sqlx::Row;

use super::*;

pub(crate) const POSTGRES_FIXTURE_APPLICATION: &str = "oxllm database fixture";

async fn sqlite() -> Arc<Database> {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    db.migrate().await.unwrap();
    db
}

fn log(
    ts: &str,
    request_id: &str,
    cost: Option<f64>,
    tags: Option<&str>,
    primary: Option<&str>,
) -> SpendLog {
    SpendLog {
        id: None,
        timestamp: ts.into(),
        request_id: Some(request_id.into()),
        model: "gpt-4o".into(),
        provider: "openai".into(),
        prompt_tokens: Some(10),
        completion_tokens: Some(5),
        total_tokens: Some(15),
        cost,
        key_id: None,
        team_id: None,
        user_id: None,
        end_user_id: None,
        tag_name: primary.map(str::to_string),
        tags: tags.map(str::to_string),
        status: Some("success".into()),
        latency_ms: Some(42),
    }
}

async fn customer(db: &Database, id: &str, spend: f64) {
    db.create_customer(&CustomerRecord {
        end_user_id: id.into(),
        alias: None,
        email: None,
        max_budget: Some(100.0),
        spend,
        budget_duration: None,
        blocked: false,
        model_blocklist: None,
        created_at: now_iso(),
        updated_at: now_iso(),
    })
    .await
    .unwrap();
}

async fn tag(db: &Database, name: &str, spend: Option<f64>) {
    db.create_tag(&TagRecord {
        name: name.into(),
        description: None,
        models: None,
        model_info: None,
        spend,
        max_budget: Some(100.0),
        created_at: now_iso(),
        updated_at: now_iso(),
    })
    .await
    .unwrap();
}

async fn metering_contract(db: &Database) {
    customer(db, "meter-customer", 10.0).await;
    customer(db, "unused-customer", 4.0).await;
    tag(db, "app", Some(2.0)).await;
    tag(db, "prod", None).await;
    tag(db, "label-only", Some(8.0)).await;
    let mut events = Vec::new();
    for (cost, tags, primary, owner) in [
        (
            Some(1.25),
            Some(r#"["app","prod","app"]"#),
            Some("label-only"),
            "meter-customer",
        ),
        (
            Some(0.5),
            Some(r#"["prod"]"#),
            Some("prod"),
            "meter-customer",
        ),
        (Some(-0.25), None, Some("app"), "meter-customer"),
        (
            Some(0.75),
            Some(r#"["app","unknown"]"#),
            Some("app"),
            "missing-customer",
        ),
        (None, None, None, "meter-customer"),
    ] {
        let mut row = log(
            "2026-01-01T00:00:00Z",
            "shared-request-id",
            cost,
            tags,
            primary,
        );
        row.end_user_id = Some(owner.into());
        events.push(SpendLogEvent::new(row));
    }
    let mut with_duplicate = events.clone();
    with_duplicate.push(events[0].clone());
    assert_eq!(db.insert_spend_events(&with_duplicate).await.unwrap(), 5);
    // Simulate retry after a lost commit acknowledgment, split into the
    // writer's per-row fallback shape. These must all be committed no-ops.
    for event in events.iter().rev() {
        assert_eq!(
            db.insert_spend_events(std::slice::from_ref(event))
                .await
                .unwrap(),
            0
        );
    }
    assert_eq!(db.insert_spend_events(&events).await.unwrap(), 0);
    assert_eq!(db.count_spend(None, None).await.unwrap(), 5);
    assert_eq!(db.total_spend(None, None).await.unwrap(), 2.25);
    assert_eq!(
        db.get_customer("meter-customer")
            .await
            .unwrap()
            .unwrap()
            .spend,
        11.5
    );
    assert_eq!(
        db.get_customer("unused-customer")
            .await
            .unwrap()
            .unwrap()
            .spend,
        4.0
    );
    assert!(db.get_customer("missing-customer").await.unwrap().is_none());
    assert_eq!(db.get_tag("app").await.unwrap().unwrap().spend, Some(3.75));
    assert_eq!(db.get_tag("prod").await.unwrap().unwrap().spend, Some(1.75));
    assert_eq!(
        db.get_tag("label-only").await.unwrap().unwrap().spend,
        Some(8.0)
    );
    assert!(db.get_tag("unknown").await.unwrap().is_none());
    let groups: BTreeMap<_, _> = db
        .query_spend_by_tag(None, None)
        .await
        .unwrap()
        .into_iter()
        .map(|r| (r.tag_name, (r.total_spend, r.total_requests)))
        .collect();
    assert_eq!(groups.get(&Some("app".into())), Some(&(1.75, 3)));
    assert_eq!(groups.get(&Some("prod".into())), Some(&(1.75, 2)));
    assert_eq!(groups.get(&Some("unknown".into())), Some(&(0.75, 1)));
    assert_eq!(groups.get(&None), Some(&(0.0, 1)));
    assert!(!groups.contains_key(&Some("label-only".into())));
}

#[tokio::test]
async fn spend_events_are_atomic_idempotent_and_count_every_distinct_tag() {
    metering_contract(&*sqlite().await).await;
}

async fn counter_rollback_contract(db: &Database) {
    customer(db, "rollback-customer", 3.0).await;
    tag(db, "rollback-tag", Some(7.0)).await;
    let trigger = match db.placeholder_style() {
        PlaceholderStyle::Question => {
            "CREATE TRIGGER reject_counter BEFORE UPDATE ON tags \
             WHEN NEW.name = 'rollback-tag' BEGIN SELECT RAISE(ABORT, 'counter rejected'); END"
        }
        PlaceholderStyle::Dollar => {
            sqlx::query(
                "CREATE FUNCTION reject_counter_fn() RETURNS trigger LANGUAGE plpgsql AS $$ \
                         BEGIN RAISE EXCEPTION 'counter rejected'; END; $$",
            )
            .execute(db.pool())
            .await
            .unwrap();
            "CREATE TRIGGER reject_counter BEFORE UPDATE ON tags FOR EACH ROW \
             WHEN (NEW.name = 'rollback-tag') EXECUTE FUNCTION reject_counter_fn()"
        }
    };
    sqlx::query(trigger).execute(db.pool()).await.unwrap();
    // More than one INSERT chunk on both backends, then fail in the final
    // tag update AFTER all logs, memberships and customer deltas were written.
    let n = match db.placeholder_style() {
        PlaceholderStyle::Question => 90,
        PlaceholderStyle::Dollar => 3_900,
    };
    let events: Vec<_> = (0..n)
        .map(|i| {
            let mut row = log(
                "2026-02-01T00:00:00Z",
                &format!("rollback-{i}"),
                Some(0.125),
                Some(r#"["rollback-tag","secondary","rollback-tag"]"#),
                Some("rollback-tag"),
            );
            row.end_user_id = Some("rollback-customer".into());
            SpendLogEvent::new(row)
        })
        .collect();
    assert!(db.insert_spend_events(&events).await.is_err());
    assert_eq!(db.count_spend(None, None).await.unwrap(), 0);
    let memberships: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_log_tags")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(memberships.0, 0);
    assert_eq!(
        db.get_customer("rollback-customer")
            .await
            .unwrap()
            .unwrap()
            .spend,
        3.0
    );
    assert_eq!(
        db.get_tag("rollback-tag").await.unwrap().unwrap().spend,
        Some(7.0)
    );
    let drop_trigger = match db.placeholder_style() {
        PlaceholderStyle::Question => "DROP TRIGGER reject_counter",
        PlaceholderStyle::Dollar => "DROP TRIGGER reject_counter ON tags",
    };
    sqlx::query(drop_trigger).execute(db.pool()).await.unwrap();
    assert_eq!(db.insert_spend_events(&events).await.unwrap(), n);
    assert_eq!(db.insert_spend_events(&events).await.unwrap(), 0);
    assert_eq!(db.count_spend(None, None).await.unwrap(), n as i64);
    assert_eq!(
        db.get_customer("rollback-customer")
            .await
            .unwrap()
            .unwrap()
            .spend,
        3.0 + n as f64 * 0.125
    );
    assert_eq!(
        db.get_tag("rollback-tag").await.unwrap().unwrap().spend,
        Some(7.0 + n as f64 * 0.125)
    );
}

#[tokio::test]
async fn counter_failure_rolls_back_all_log_chunks_memberships_and_counters() {
    counter_rollback_contract(&*sqlite().await).await;
}

#[tokio::test]
async fn batched_counters_update_each_scope_once_per_batch() {
    let db = sqlite().await;
    customer(&db, "meter", 0.0).await;
    tag(&db, "app", None).await;
    tag(&db, "prod", None).await;
    sqlx::query("CREATE TABLE update_calls (scope TEXT NOT NULL)")
        .execute(db.pool())
        .await
        .unwrap();
    sqlx::query(
        "CREATE TRIGGER count_customer_updates AFTER UPDATE ON customers \
                 BEGIN INSERT INTO update_calls VALUES ('customer'); END",
    )
    .execute(db.pool())
    .await
    .unwrap();
    sqlx::query(
        "CREATE TRIGGER count_tag_updates AFTER UPDATE ON tags \
                 BEGIN INSERT INTO update_calls VALUES (NEW.name); END",
    )
    .execute(db.pool())
    .await
    .unwrap();
    let rows: Vec<_> = (0..90)
        .map(|i| {
            let mut row = log(
                "2026-01-01T00:00:00Z",
                &format!("batch-{i}"),
                Some(0.125),
                Some(r#"["app","prod"]"#),
                Some("app"),
            );
            row.end_user_id = Some("meter".into());
            row
        })
        .collect();
    db.insert_spend_logs_batch(&rows).await.unwrap();
    let calls: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM update_calls")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(
        calls.0, 3,
        "90 requests require one customer and two tag updates"
    );
    assert_eq!(
        db.get_customer("meter").await.unwrap().unwrap().spend,
        11.25
    );
    assert_eq!(
        db.get_tag("prod").await.unwrap().unwrap().spend,
        Some(11.25)
    );
}

async fn literal_tags_contract(db: &Database) {
    let names = [
        "app%",
        "app_",
        "appX",
        "Prod",
        "prod",
        "app",
        "app-ab",
        "a\"b",
        r"a\b",
        "line\nbreak",
        "雪",
        "quote', OR 1=1 --",
    ];
    for (i, name) in names.iter().enumerate() {
        let json = serde_json::to_string(&vec!["primary", *name, *name]).unwrap();
        db.insert_spend_log(&log(
            "2026-03-01T00:00:00Z",
            &format!("literal-{i}"),
            Some(i as f64 + 1.0),
            Some(&json),
            Some("primary"),
        ))
        .await
        .unwrap();
    }
    // Noncanonical JSON escape spelling must match the decoded tag, too.
    db.insert_spend_log(&log(
        "2026-03-01T00:00:00Z",
        "escaped",
        Some(0.25),
        Some(r#"["\u0050rod"]"#),
        None,
    ))
    .await
    .unwrap();
    db.insert_spend_log(&log(
        "2026-03-01T00:00:00Z",
        "legacy",
        Some(0.5),
        Some("malformed JSON"),
        Some("app%"),
    ))
    .await
    .unwrap();
    for (i, name) in names.iter().enumerate() {
        let filter = SpendFilter {
            tag_name: Some((*name).into()),
            ..Default::default()
        };
        let count = if *name == "Prod" || *name == "app%" {
            2
        } else {
            1
        };
        let extra = if *name == "Prod" {
            0.25
        } else if *name == "app%" {
            0.5
        } else {
            0.0
        };
        let expected = i as f64 + 1.0 + extra;
        assert_eq!(db.query_spend(&filter).await.unwrap().len(), count);
        assert_eq!(
            db.count_spend_filtered(&filter).await.unwrap(),
            count as i64
        );
        assert_eq!(db.total_spend_filtered(&filter).await.unwrap(), expected);
        assert_eq!(
            db.query_spend_paginated(None, 1, &filter)
                .await
                .unwrap()
                .len(),
            1
        );
    }
    assert_eq!(
        db.count_spend_filtered(&SpendFilter {
            tag_name: Some("PROD".into()),
            ..Default::default()
        })
        .await
        .unwrap(),
        0
    );
}

#[tokio::test]
async fn tag_filters_are_literal_case_sensitive_json_memberships() {
    literal_tags_contract(&*sqlite().await).await;
}

async fn tag_pagination_contract(db: &Database) {
    // Unique combinations grow with requests, but the shared tag's total
    // must include every request even on a one-group response page.
    let rows: Vec<_> = (0..1_105)
        .map(|i| {
            let tags =
                serde_json::to_string(&["shared".to_string(), format!("tag-{i:04}")]).unwrap();
            log(
                "2026-04-01T23:59:59.999999999Z",
                &format!("page-{i}"),
                Some(0.125),
                Some(&tags),
                Some("shared"),
            )
        })
        .collect();
    db.insert_spend_logs_batch(&rows).await.unwrap();
    let first = db
        .query_spend_by_tag_paginated(Some("2026-04-01"), Some("2026-04-01"), 1, 0)
        .await
        .unwrap();
    assert_eq!(first.data.len(), 1);
    assert_eq!(first.data[0].tag_name.as_deref(), Some("shared"));
    assert_eq!(first.data[0].total_requests, 1_105);
    assert_eq!(first.data[0].total_spend, 138.125);
    assert_eq!(first.next_offset, Some(1));
    let mut seen = BTreeSet::from([Some("shared".to_string())]);
    let mut offset = first.next_offset;
    while let Some(next) = offset {
        let page = db
            .query_spend_by_tag_paginated(Some("2026-04-01"), Some("2026-04-01"), 127, next)
            .await
            .unwrap();
        assert!(page.data.len() <= 127);
        for row in &page.data {
            assert!(
                seen.insert(row.tag_name.clone()),
                "repeated group on later page"
            );
            assert_eq!(row.total_spend, 0.125);
            assert_eq!(row.total_requests, 1);
        }
        offset = page.next_offset;
    }
    assert_eq!(seen.len(), 1_106);
    assert!(
        db.query_spend_by_tag(None, None).await.is_err(),
        "unpaged helper must not silently return an incomplete billing rollup"
    );
    let clamped = db
        .query_spend_by_tag_paginated(None, None, -1, -5)
        .await
        .unwrap();
    assert_eq!(
        (clamped.limit, clamped.offset, clamped.data.len()),
        (1, 0, 1)
    );
    assert_eq!(
        db.query_spend_by_tag_paginated(None, None, i64::MAX, 0)
            .await
            .unwrap()
            .data
            .len(),
        1_000
    );
}

#[tokio::test]
async fn tag_pages_bound_groups_without_truncating_their_billing_totals() {
    tag_pagination_contract(&*sqlite().await).await;
}

async fn retention_contract(db: &Database) {
    customer(db, "retained-total", 4.0).await;
    tag(db, "retained-total", Some(3.0)).await;
    let mut rows: Vec<_> = (0..1_007)
        .map(|i| {
            let mut row = log(
                "2026-05-01T00:00:00Z",
                &format!("old-{i}"),
                Some(0.125),
                Some(r#"["retained-total","other"]"#),
                Some("retained-total"),
            );
            row.end_user_id = Some("retained-total".into());
            row
        })
        .collect();
    rows.push(log(
        "2026-06-01T00:00:00Z",
        "boundary",
        Some(1.0),
        None,
        None,
    ));
    db.insert_spend_logs_batch(&rows).await.unwrap();
    assert_eq!(
        db.delete_spend_logs_before("2026-06-01T00:00:00Z")
            .await
            .unwrap(),
        1_000
    );
    assert_eq!(db.count_spend(None, None).await.unwrap(), 8);
    // A writer can make progress between prune calls, without waiting for
    // the full old-data backlog to disappear.
    db.insert_spend_log(&log(
        "2026-06-01T00:00:00.000000001Z",
        "new-writer",
        Some(2.0),
        None,
        None,
    ))
    .await
    .unwrap();
    assert_eq!(
        db.delete_spend_logs_before_batch("2026-06-01T00:00:00Z", -1)
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        db.delete_spend_logs_before("2026-06-01T00:00:00Z")
            .await
            .unwrap(),
        6
    );
    assert_eq!(
        db.delete_spend_logs_before("2026-06-01T00:00:00Z")
            .await
            .unwrap(),
        0
    );
    assert_eq!(db.total_spend(None, None).await.unwrap(), 3.0);
    let memberships: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_log_tags")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(
        memberships.0, 0,
        "pruning must clean up indexed memberships"
    );
    assert_eq!(
        db.get_customer("retained-total")
            .await
            .unwrap()
            .unwrap()
            .spend,
        129.875
    );
    assert_eq!(
        db.get_tag("retained-total").await.unwrap().unwrap().spend,
        Some(128.875)
    );
}

#[tokio::test]
async fn retention_is_bounded_and_preserves_durable_lifetime_totals() {
    retention_contract(&*sqlite().await).await;
}

pub(crate) async fn load_v1(db: &Database) {
    let mut fixture = include_str!("fixtures/v1.sql").to_string();
    if matches!(db.placeholder_style(), PlaceholderStyle::Dollar) {
        fixture = fixture
            .replace(
                "INTEGER PRIMARY KEY AUTOINCREMENT",
                "INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY",
            )
            .replace(" REAL", " DOUBLE PRECISION");
    }
    for statement in fixture.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        sqlx::query(statement).execute(db.pool()).await.unwrap();
    }
    assert_eq!(db.get_schema_version().await.unwrap(), 1);
}

async fn assert_v1_preserved(db: &Database) {
    db.migrate().await.expect("real v1 schema upgrades");
    db.migrate().await.expect("rerun is idempotent");
    assert_eq!(db.get_schema_version().await.unwrap(), SCHEMA_VERSION);
    let key = db.get_key_by_hash("fixture-hash").await.unwrap().unwrap();
    assert!(!key.is_active);
    assert_eq!(key.max_budget, Some(12.5));
    assert_eq!(key.rpm, Some(5));
    assert_eq!(key.tpm, Some(100));
    assert_eq!(key.key_alias.as_deref(), Some("Original key"));
    assert_eq!(key.org_id.as_deref(), Some("v1-org"));
    assert_eq!(key.project_id.as_deref(), Some("v1-project"));
    assert_eq!(key.models.as_deref(), Some(r#"["gpt-4o"]"#));
    assert!(key.allowed_cache_controls.is_none());
    assert_eq!(key.expires_at.as_deref(), Some("2030-01-01T00:00:00Z"));
    let team = db.get_team("v1-team").await.unwrap().unwrap();
    assert!(!team.blocked);
    assert_eq!(team.max_budget, Some(50.25));
    assert_eq!(
        db.get_user("v1-user").await.unwrap().unwrap().max_budget,
        Some(25.5)
    );
    let org = db.get_organization("v1-org").await.unwrap().unwrap();
    assert_eq!(org.org_name, "Original org");
    assert_eq!(org.max_budget, Some(100.5));
    assert!(org.models.is_none() && org.metadata.is_none());
    assert_eq!(
        db.get_project("v1-project")
            .await
            .unwrap()
            .unwrap()
            .team_id
            .as_deref(),
        Some("v1-team")
    );
    let model = db.get_model("v1-model").await.unwrap().unwrap();
    assert_eq!(
        model.oxllm_params,
        serde_json::json!({"model": "openai/gpt-4o"})
    );
    assert_eq!(model.model_info, Some(serde_json::json!({"fixture": true})));
    assert_eq!(db.count_spend(None, None).await.unwrap(), 3);
    assert_eq!(
        db.total_spend(Some("2026-07-31"), Some("2026-07-31"))
            .await
            .unwrap(),
        4.0
    );
    let whole = db
        .query_spend_by_request_id("v1-whole")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(whole.id, Some(1));
    assert_eq!(whole.timestamp, "2026-07-31T23:59:59.000000000Z");
    assert_eq!(
        (
            whole.prompt_tokens,
            whole.completion_tokens,
            whole.total_tokens,
            whole.latency_ms
        ),
        (Some(10), Some(5), Some(15), Some(42))
    );
    assert!(whole.tags.is_none() && whole.end_user_id.is_none());
    let exact = SpendFilter {
        start_date: Some("2026-07-31T23:59:59.5000Z".into()),
        end_date: Some("2026-07-31T23:59:59.5Z".into()),
        ..Default::default()
    };
    assert_eq!(db.total_spend_filtered(&exact).await.unwrap(), 2.5);
    assert_eq!(db.query_spend(&exact).await.unwrap().len(), 1);
    // New fields and writes must also work after the upgrade, not just reads.
    tag(db, "after-upgrade", None).await;
    db.insert_spend_log(&log(
        "2026-08-01T00:00:00Z",
        "after-upgrade",
        Some(0.125),
        Some(r#"["after-upgrade"]"#),
        Some("after-upgrade"),
    ))
    .await
    .unwrap();
    assert_eq!(
        db.get_tag("after-upgrade").await.unwrap().unwrap().spend,
        Some(0.125)
    );
    assert_eq!(
        db.delete_spend_logs_before("2026-07-31T23:59:59.5Z")
            .await
            .unwrap(),
        1
    );
    assert_eq!(db.count_spend(None, None).await.unwrap(), 3);
}

#[tokio::test]
async fn authentic_v1_upgrade_preserves_data_and_normalizes_legacy_timestamps() {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    load_v1(&db).await;
    assert_v1_preserved(&db).await;
}

async fn migration_repair_failure_contract(db: &Database) {
    load_v1(db).await;
    // A later repair failure must roll back only its bounded batch. The DDL
    // and earlier repair commits remain ready, and pending rows read correctly.
    for i in 0..300 {
        let request_id = if i == 299 {
            "fail-legacy".to_string()
        } else {
            format!("legacy-{i}")
        };
        let q = format!(
            "INSERT INTO spend_logs (timestamp, request_id, model, provider, cost) \
             VALUES ('2026-07-31T23:59:59.5Z', {}, 'm', 'p', 0.125)",
            db.ph(1)
        );
        sqlx::query(&q)
            .bind(request_id)
            .execute(db.pool())
            .await
            .unwrap();
    }
    let trigger = match db.placeholder_style() {
        PlaceholderStyle::Question => {
            "CREATE TRIGGER reject_legacy_timestamp BEFORE UPDATE OF timestamp ON spend_logs \
             WHEN NEW.request_id = 'fail-legacy' \
             BEGIN SELECT RAISE(ABORT, 'legacy timestamp rejected'); END"
        }
        PlaceholderStyle::Dollar => {
            sqlx::query(
                "CREATE FUNCTION reject_legacy_timestamp_fn() RETURNS trigger LANGUAGE plpgsql AS $$ \
                 BEGIN RAISE EXCEPTION 'legacy timestamp rejected'; END; $$",
            )
            .execute(db.pool())
            .await
            .unwrap();
            "CREATE TRIGGER reject_legacy_timestamp BEFORE UPDATE OF timestamp ON spend_logs \
             FOR EACH ROW WHEN (NEW.request_id = 'fail-legacy') \
             EXECUTE FUNCTION reject_legacy_timestamp_fn()"
        }
    };
    sqlx::query(trigger).execute(db.pool()).await.unwrap();
    db.migrate()
        .await
        .expect("DDL readiness is independent of repair failure");
    assert_eq!(db.get_schema_version().await.unwrap(), SCHEMA_VERSION);
    let original: (String,) =
        sqlx::query_as("SELECT timestamp FROM spend_logs WHERE request_id = 'v1-whole'")
            .fetch_one(db.pool())
            .await
            .unwrap();
    assert_eq!(original.0, "2026-07-31T23:59:59.000000000Z");
    let pending: (String, i64) = sqlx::query_as(
        "SELECT timestamp, CAST(metadata_version AS BIGINT) FROM spend_logs WHERE request_id = 'legacy-253'",
    )
    .fetch_one(db.pool())
    .await
    .unwrap();
    assert_eq!(pending, ("2026-07-31T23:59:59.5Z".into(), 0));
    assert!(db.repair_spend_metadata_batch(256).await.is_err());
    let read = db
        .query_spend_by_request_id("fail-legacy")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(read.timestamp, "2026-07-31T23:59:59.500000000Z");
    assert_eq!(
        db.total_spend(Some("2026-07-31"), Some("2026-07-31"))
            .await
            .unwrap(),
        41.5
    );
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_logs")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count.0, 303);
    let drop_trigger = match db.placeholder_style() {
        PlaceholderStyle::Question => "DROP TRIGGER reject_legacy_timestamp",
        PlaceholderStyle::Dollar => "DROP TRIGGER reject_legacy_timestamp ON spend_logs",
    };
    sqlx::query(drop_trigger).execute(db.pool()).await.unwrap();
    db.migrate().await.unwrap();
    assert_eq!(db.get_schema_version().await.unwrap(), SCHEMA_VERSION);
    assert_eq!(db.count_spend(None, None).await.unwrap(), 303);
    assert_eq!(
        db.query_spend_by_request_id("fail-legacy")
            .await
            .unwrap()
            .unwrap()
            .timestamp,
        "2026-07-31T23:59:59.500000000Z"
    );
    assert_eq!(
        db.total_spend(Some("2026-07-31"), Some("2026-07-31"))
            .await
            .unwrap(),
        41.5
    );
}

#[tokio::test]
async fn failed_legacy_repair_keeps_ready_schema_prior_commits_and_exact_reads() {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    migration_repair_failure_contract(&db).await;
}

#[tokio::test]
async fn malformed_create_is_not_swallowed_as_an_idempotent_column_backfill() {
    let db = sqlite().await;
    let mut conn = db.pool().acquire().await.unwrap();
    let result = db
        .run_migration_batch(
            vec!["CREATE TABLE malformed (duplicate TEXT, duplicate TEXT)".into()],
            &mut conn,
        )
        .await;
    assert!(
        result.is_err(),
        "only repeated ALTER ADD COLUMN is idempotent"
    );
}

#[tokio::test]
async fn legacy_tag_backfill_preserves_counters_and_indexes_decoded_memberships() {
    let db = sqlite().await;
    tag(&db, "Prod%", Some(12.0)).await;
    // Emulate retained rows written by v10, bypassing today's write path.
    sqlx::query(
        "INSERT INTO spend_logs (timestamp, request_id, model, provider, cost, tags, tag_name) \
        VALUES ('2026-07-31T23:59:59.5Z', 'legacy-tags', 'm', 'p', 0.5, \
        '[\"\\u0050rod%\",\"Prod%\",\"secondary\"]', 'Prod%')",
    )
    .execute(db.pool())
    .await
    .unwrap();
    sqlx::query("DELETE FROM _schema_version WHERE version >= 11")
        .execute(db.pool())
        .await
        .unwrap();
    db.migrate().await.unwrap();
    db.migrate().await.unwrap();
    assert_eq!(
        db.get_tag("Prod%").await.unwrap().unwrap().spend,
        Some(12.0),
        "backfill must not replay already-accounted durable spend"
    );
    for name in ["Prod%", "secondary"] {
        let filter = SpendFilter {
            tag_name: Some(name.into()),
            ..Default::default()
        };
        assert_eq!(db.count_spend_filtered(&filter).await.unwrap(), 1);
        assert_eq!(db.total_spend_filtered(&filter).await.unwrap(), 0.5);
    }
}

async fn scope_cleanup_contract(db: &Database) {
    let now = now_iso();
    db.create_organization(&OrganizationRecord {
        org_id: "scope-org".into(),
        org_name: "scope org".into(),
        max_budget: Some(100.0),
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now.clone(),
        updated_at: now.clone(),
        models: None,
        metadata: None,
        soft_budget: None,
        max_parallel_requests: None,
        model_max_budget: None,
    })
    .await
    .unwrap();
    db.create_team(&TeamRecord {
        team_id: "scope-team".into(),
        team_name: "scope team".into(),
        max_budget: Some(50.0),
        budget_duration: None,
        rpm: Some(10),
        tpm: Some(100),
        created_at: now.clone(),
        updated_at: now.clone(),
        models: None,
        blocked: true,
    })
    .await
    .unwrap();
    let team = db.get_team("scope-team").await.unwrap().unwrap();
    assert!(team.blocked);
    assert_eq!((team.rpm, team.tpm), (Some(10), Some(100)));
    db.create_user(&UserRecord {
        user_id: "scope-user".into(),
        user_name: "scope user".into(),
        max_budget: Some(25.0),
        budget_duration: None,
        created_at: now.clone(),
        updated_at: now.clone(),
        models: None,
    })
    .await
    .unwrap();
    db.create_project(&ProjectRecord {
        project_id: "scope-project".into(),
        project_name: "scope project".into(),
        org_id: Some("scope-org".into()),
        team_id: Some("scope-team".into()),
        max_budget: Some(20.0),
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now.clone(),
        updated_at: now.clone(),
    })
    .await
    .unwrap();
    for (id, active) in [("active", true), ("blocked", false)] {
        db.create_key(&KeyRecord {
            key_id: id.into(),
            key_hash: format!("scope-hash-{id}"),
            key_alias: None,
            org_id: Some("scope-org".into()),
            project_id: Some("scope-project".into()),
            team_id: Some("scope-team".into()),
            user_id: Some("scope-user".into()),
            max_budget: Some(13.0),
            budget_duration: None,
            rpm: Some(5),
            tpm: Some(50),
            created_at: now.clone(),
            updated_at: now.clone(),
            expires_at: None,
            is_active: active,
            models: None,
            allowed_cache_controls: None,
        })
        .await
        .unwrap();
    }
    let trigger = match db.placeholder_style() {
        PlaceholderStyle::Question => {
            "CREATE TRIGGER reject_detach BEFORE UPDATE ON keys \
            WHEN NEW.org_id IS NULL BEGIN SELECT RAISE(ABORT, 'detach rejected'); END"
        }
        PlaceholderStyle::Dollar => {
            sqlx::query(
                "CREATE FUNCTION reject_detach_fn() RETURNS trigger LANGUAGE plpgsql AS $$ \
                BEGIN RAISE EXCEPTION 'detach rejected'; END; $$",
            )
            .execute(db.pool())
            .await
            .unwrap();
            "CREATE TRIGGER reject_detach BEFORE UPDATE ON keys FOR EACH ROW \
             WHEN (NEW.org_id IS NULL) EXECUTE FUNCTION reject_detach_fn()"
        }
    };
    sqlx::query(trigger).execute(db.pool()).await.unwrap();
    assert!(db.delete_organization("scope-org").await.is_err());
    assert!(db.get_organization("scope-org").await.unwrap().is_some());
    assert_eq!(
        db.get_key("active")
            .await
            .unwrap()
            .unwrap()
            .org_id
            .as_deref(),
        Some("scope-org")
    );
    assert_eq!(
        db.get_project("scope-project")
            .await
            .unwrap()
            .unwrap()
            .org_id
            .as_deref(),
        Some("scope-org")
    );
    let drop_trigger = match db.placeholder_style() {
        PlaceholderStyle::Question => "DROP TRIGGER reject_detach",
        PlaceholderStyle::Dollar => "DROP TRIGGER reject_detach ON keys",
    };
    sqlx::query(drop_trigger).execute(db.pool()).await.unwrap();
    let deleted = db
        .delete_organization_and_detach("scope-org")
        .await
        .unwrap();
    assert!(deleted.deleted);
    assert_eq!(deleted.key_hashes.len(), 2);
    assert_eq!(deleted.project_ids, vec!["scope-project"]);
    let project = db.get_project("scope-project").await.unwrap().unwrap();
    assert!(project.org_id.is_none());
    assert_eq!(project.team_id.as_deref(), Some("scope-team"));
    for (id, active) in [("active", true), ("blocked", false)] {
        let key = db.get_key(id).await.unwrap().unwrap();
        assert!(key.org_id.is_none());
        assert_eq!(key.project_id.as_deref(), Some("scope-project"));
        assert_eq!(key.is_active, active);
        assert_eq!(
            (key.max_budget, key.rpm, key.tpm),
            (Some(13.0), Some(5), Some(50))
        );
    }
    assert!(
        !db.delete_organization_and_detach("scope-org")
            .await
            .unwrap()
            .deleted
    );
    assert_eq!(
        db.delete_team_and_detach("scope-team")
            .await
            .unwrap()
            .project_ids,
        vec!["scope-project"]
    );
    assert!(
        db.get_project("scope-project")
            .await
            .unwrap()
            .unwrap()
            .team_id
            .is_none()
    );
    assert_eq!(
        db.delete_user_and_detach("scope-user")
            .await
            .unwrap()
            .key_hashes
            .len(),
        2
    );
    assert_eq!(
        db.delete_project_and_detach("scope-project")
            .await
            .unwrap()
            .key_hashes
            .len(),
        2
    );
    let key = db.get_key("blocked").await.unwrap().unwrap();
    assert!(key.team_id.is_none() && key.user_id.is_none() && key.project_id.is_none());
    assert!(!key.is_active);
}

#[tokio::test]
async fn scope_deletion_detaches_atomically_and_keeps_other_limits_and_key_status() {
    scope_cleanup_contract(&*sqlite().await).await;
}

#[tokio::test]
async fn scope_lookups_have_usable_indices() {
    let db = sqlite().await;
    for column in ["team_id", "user_id", "org_id", "project_id"] {
        let rows = sqlx::query(&format!(
            "EXPLAIN QUERY PLAN SELECT key_hash FROM keys WHERE {column} = ?"
        ))
        .bind("scope")
        .fetch_all(db.pool())
        .await
        .unwrap();
        let plans: Vec<String> = rows.iter().map(|r| r.try_get("detail").unwrap()).collect();
        assert!(
            plans
                .iter()
                .any(|p| p.contains(&format!("idx_keys_{column}"))),
            "{plans:?}"
        );
    }
}

pub(crate) struct PostgresFixture {
    pub(crate) db: Arc<Database>,
    admin: sqlx::Pool<sqlx::Any>,
    schema: String,
}

impl PostgresFixture {
    pub(crate) async fn new(url: &str, max_connections: u32) -> Self {
        sqlx::any::install_default_drivers();
        let schema = format!("oxllm_test_{}", uuid::Uuid::new_v4().simple());
        let admin = sqlx::pool::PoolOptions::<sqlx::Any>::new()
            .max_connections(1)
            .connect(url)
            .await
            .expect("connect dedicated CI Postgres");
        sqlx::query(&format!("CREATE SCHEMA {schema}"))
            .execute(&admin)
            .await
            .unwrap();
        let pool_schema = schema.clone();
        let pool = sqlx::pool::PoolOptions::<sqlx::Any>::new()
            .max_connections(max_connections)
            .after_connect(move |conn, _| {
                let q = format!("SET search_path TO {pool_schema}");
                Box::pin(async move {
                    sqlx::query(&q).execute(&mut *conn).await?;
                    sqlx::query(
                        "SELECT set_config('application_name', $1 || ':' || md5(current_database() || ':' || current_schema()), false)",
                    )
                    .bind(POSTGRES_FIXTURE_APPLICATION)
                    .execute(&mut *conn)
                    .await?;
                    Ok(())
                })
            })
            .connect(url)
            .await
            .unwrap();
        Self {
            db: Arc::new(Database::new(pool, PlaceholderStyle::Dollar)),
            admin,
            schema,
        }
    }

    pub(crate) async fn cleanup(self) {
        self.db.pool().close().await;
        sqlx::query(&format!("DROP SCHEMA {} CASCADE", self.schema))
            .execute(&self.admin)
            .await
            .unwrap();
        self.admin.close().await;
    }
}

/// Test lock holders follow the same nonblocking session discipline as real
/// migrators. A blocking advisory SELECT in an open transaction can itself
/// deadlock a concurrent index build in another test schema.
pub(crate) async fn fixture_migration_lock(
    db: &Database,
) -> anyhow::Result<sqlx::pool::PoolConnection<sqlx::Any>> {
    let mut conn = db.pool().acquire().await?;
    conn.close_on_drop();
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let acquired: (i64,) = sqlx::query_as(
                "SELECT CAST(CASE WHEN pg_try_advisory_lock($1) THEN 1 ELSE 0 END AS BIGINT)",
            )
            .bind(crate::migrate::OXLLM_MIGRATE_LOCK_KEY)
            .fetch_one(&mut *conn)
            .await?;
            if acquired.0 == 1 {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await??;
    Ok(conn)
}

async fn postgres_lock_timeout_contract(db: &Database) {
    let key = crate::migrate::OXLLM_MIGRATE_LOCK_KEY;
    let mut holder = fixture_migration_lock(db).await.unwrap();
    let err = tokio::time::timeout(
        Duration::from_secs(5),
        db.migrate_with_lock_timeout(Duration::from_millis(40)),
    )
    .await
    .expect("server must terminate the blocked advisory statement")
    .unwrap_err();
    assert!(
        err.to_string().contains("migration advisory lock")
            || err.downcast_ref::<PostgresPreparationTimedOut>().is_some()
    );
    // Inspect the server while the holder is STILL locked. A canceled client
    // future alone would leave a waiter behind until this holder releases.
    let waiting: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid \
         WHERE l.locktype = 'advisory' AND l.classid::bigint = $1 AND l.objid::bigint = $2 \
           AND NOT l.granted AND a.application_name = $3 || ':' || md5(current_database() || ':' || current_schema())",
    )
    .bind(key >> 32)
    .bind(key & 0xffff_ffff)
    .bind(crate::postgres_prepare::PREPARATION_APPLICATION)
    .fetch_one(&mut *holder)
    .await
    .unwrap();
    assert_eq!(
        waiting.0, 0,
        "timed-out migration must leave no server-side waiter"
    );
    holder.close().await.unwrap();
    db.migrate().await.unwrap();
    db.migrate().await.unwrap();
    assert_eq!(db.get_schema_version().await.unwrap(), SCHEMA_VERSION);
}

/// Pause a historical repair AFTER the DDL commit. Other replicas, readers and
/// old-style writers must progress without the migration advisory/table locks.
async fn postgres_repair_releases_ddl_locks_contract(db: &Arc<Database>) {
    load_v1(db).await;
    let repair_lock = crate::migrate::OXLLM_MIGRATE_LOCK_KEY + 1;
    sqlx::query(&format!(
        "CREATE FUNCTION pause_spend_repair_fn() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN \
         PERFORM set_config('lock_timeout', '5s', true); \
         PERFORM pg_advisory_xact_lock({repair_lock}); RETURN NEW; END; $$"
    ))
    .execute(db.pool())
    .await
    .unwrap();
    sqlx::query(
        "CREATE TRIGGER pause_spend_repair BEFORE UPDATE OF timestamp ON spend_logs \
         FOR EACH ROW WHEN (OLD.request_id = 'v1-whole') EXECUTE FUNCTION pause_spend_repair_fn()",
    )
    .execute(db.pool())
    .await
    .unwrap();
    let mut holder = db.pool().begin().await.unwrap();
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(repair_lock)
        .execute(&mut *holder)
        .await
        .unwrap();
    let migrating_db = Arc::clone(db);
    let migration = tokio::spawn(async move { migrating_db.migrate().await });
    let repair_pid = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let waiting: Option<(i64,)> = sqlx::query_as(
                "SELECT CAST(l.pid AS BIGINT) FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid \
                 WHERE l.locktype = 'advisory' AND l.classid::bigint = $1 AND l.objid::bigint = $2 \
                   AND NOT l.granted AND a.application_name = $3 || ':' || md5(current_database() || ':' || current_schema())",
            )
            .bind(repair_lock >> 32)
            .bind(repair_lock & 0xffff_ffff)
            .bind(POSTGRES_FIXTURE_APPLICATION)
            .fetch_optional(db.pool())
            .await
            .unwrap();
            if let Some((pid,)) = waiting {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("repair must begin after DDL readiness");
    assert_eq!(db.get_schema_version().await.unwrap(), SCHEMA_VERSION);
    let exclusive: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_locks WHERE relation = 'spend_logs'::regclass \
         AND mode = 'AccessExclusiveLock' AND granted",
    )
    .fetch_one(db.pool())
    .await
    .unwrap();
    assert_eq!(
        exclusive.0, 0,
        "historical repair must not retain the DDL table lock"
    );
    let retained_lock: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_locks WHERE pid = $1 AND locktype = 'advisory' \
         AND classid::bigint = $2 AND objid::bigint = $3 AND granted",
    )
    .bind(repair_pid)
    .bind(crate::migrate::OXLLM_MIGRATE_LOCK_KEY >> 32)
    .bind(crate::migrate::OXLLM_MIGRATE_LOCK_KEY & 0xffff_ffff)
    .fetch_one(db.pool())
    .await
    .unwrap();
    assert_eq!(
        retained_lock.0, 0,
        "repair must not retain the migration advisory lock"
    );
    sqlx::query(
        "INSERT INTO spend_logs (timestamp, request_id, model, provider, cost, tags) \
         VALUES ('2026-07-31T23:59:59.999999999Z', 'live-old-writer', 'm', 'p', 0.125, '[\"live\",\"live\"]')"
    ).execute(db.pool()).await.unwrap();
    assert_eq!(
        db.count_spend_filtered(&SpendFilter {
            tag_name: Some("live".into()),
            ..Default::default()
        })
        .await
        .unwrap(),
        1
    );
    let standby_repair =
        tokio::time::timeout(Duration::from_secs(1), db.repair_spend_metadata_batch(256))
            .await
            .expect("another ready worker skips contending maintenance")
            .unwrap();
    assert_eq!(standby_repair.scanned_rows, 0);
    holder.rollback().await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), migration)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        db.total_spend(Some("2026-07-31"), Some("2026-07-31"))
            .await
            .unwrap(),
        4.125
    );
    // Even the current-schema fast path must bound waits on version-table DDL.
    let mut holder = db.pool().begin().await.unwrap();
    sqlx::query("LOCK TABLE _schema_version IN ACCESS EXCLUSIVE MODE")
        .execute(&mut *holder)
        .await
        .unwrap();
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        db.migrate_with_lock_timeout(Duration::from_millis(40)),
    )
    .await
    .expect("readiness check must use the server-side lock timeout")
    .unwrap_err();
    assert!(
        error
            .downcast_ref::<PostgresPreparationTimedOut>()
            .is_some()
            || error
                .downcast_ref::<sqlx::Error>()
                .and_then(|error| error.as_database_error())
                .and_then(|error| error.code())
                .is_some_and(|code| matches!(code.as_ref(), "55P03" | "57014"))
    );
    holder.rollback().await.unwrap();
}

async fn postgres_incomplete_event_index_contract(db: &Database) {
    load_v1(db).await;
    sqlx::query("ALTER TABLE spend_logs ADD COLUMN event_id TEXT")
        .execute(db.pool())
        .await
        .unwrap();
    // A same-named non-unique fixture index exercises IF NOT EXISTS skipping
    // an unusable definition. Production migration must reject it intact.
    sqlx::query("CREATE INDEX idx_spend_logs_event_id ON spend_logs(event_id)")
        .execute(db.pool())
        .await
        .unwrap();
    let error = db.migrate().await.unwrap_err();
    assert!(error.downcast_ref::<SpendEventIndexNotReady>().is_some());
    assert_eq!(db.get_schema_version().await.unwrap(), 1);
    let retained: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_logs")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(retained.0, 3);
    let unchanged: (i64,) = sqlx::query_as(
        "SELECT CAST(CASE WHEN indisunique THEN 1 ELSE 0 END AS BIGINT) \
         FROM pg_index WHERE indexrelid = 'idx_spend_logs_event_id'::regclass",
    )
    .fetch_one(db.pool())
    .await
    .unwrap();
    assert_eq!(
        unchanged.0, 0,
        "migration must not replace an operator index"
    );
    // Replace only our deliberately broken fixture index, then prepare the
    // valid one concurrently outside the application DDL/advisory transaction.
    sqlx::query("DROP INDEX idx_spend_logs_event_id")
        .execute(db.pool())
        .await
        .unwrap();
    sqlx::query("CREATE UNIQUE INDEX CONCURRENTLY idx_spend_logs_event_id ON spend_logs(event_id)")
        .execute(db.pool())
        .await
        .unwrap();
    db.migrate().await.unwrap();
    assert_eq!(db.get_schema_version().await.unwrap(), SCHEMA_VERSION);
    assert_eq!(db.count_spend(None, None).await.unwrap(), 3);
}

/// Called by the existing postgres_roundtrip_ci entry point, so CI's test
/// filter now exercises actual migrations, queries, transactions and retries.
pub(crate) async fn postgres_release_validation() {
    let Ok(url) = std::env::var("OXLLM_TEST_POSTGRES_URL") else {
        eprintln!("skipping: OXLLM_TEST_POSTGRES_URL not set");
        return;
    };
    assert!(
        url.starts_with("postgres://") || url.starts_with("postgresql://"),
        "OXLLM_TEST_POSTGRES_URL must select the disposable CI Postgres database"
    );
    for scenario in 0..19 {
        let max_connections = if scenario == 17 { 1 } else { 4 };
        let fixture = PostgresFixture::new(&url, max_connections).await;
        let db = &fixture.db;
        if scenario == 6 {
            load_v1(db).await;
            // A DDL lock failure must roll back the whole upgrade/version
            // chain and leave it safely rerunnable after the holder exits.
            let mut holder = db.pool().begin().await.unwrap();
            sqlx::query("LOCK TABLE keys IN ACCESS EXCLUSIVE MODE")
                .execute(&mut *holder)
                .await
                .unwrap();
            let failed = tokio::time::timeout(
                Duration::from_secs(5),
                db.migrate_with_lock_timeout(Duration::from_millis(40)),
            )
            .await
            .unwrap();
            assert!(failed.is_err());
            assert_eq!(db.get_schema_version().await.unwrap(), 1);
            holder.rollback().await.unwrap();
            assert_v1_preserved(db).await;
        } else if scenario == 7 {
            migration_repair_failure_contract(db).await;
        } else if scenario == 15 {
            crate::db_regression_tests::legacy_model_duplicates_contract(db)
                .await
                .expect("legacy aliases preserved");
        } else if scenario == 16 {
            postgres_repair_releases_ddl_locks_contract(db).await;
        } else if scenario == 17 {
            db.migrate().await.unwrap();
            crate::db_regression_tests::nullable_float_bind_contract(db)
                .await
                .expect("cached NULL/value monetary and integer parameter types");
        } else if scenario == 18 {
            postgres_incomplete_event_index_contract(db).await;
        } else {
            postgres_lock_timeout_contract(db).await;
            match scenario {
                0 => metering_contract(db).await,
                1 => counter_rollback_contract(db).await,
                2 => literal_tags_contract(db).await,
                3 => tag_pagination_contract(db).await,
                4 => retention_contract(db).await,
                5 => scope_cleanup_contract(db).await,
                8 => crate::db_regression_tests::rolling_tag_writes_contract(db)
                    .await
                    .expect("live exact tags"),
                9 => crate::db_regression_tests::rolling_timestamp_contract(db)
                    .await
                    .expect("live UTC nanoseconds"),
                10 => crate::db_regression_tests::bounded_metadata_contract(db)
                    .await
                    .expect("bounded metadata repair"),
                11 => crate::db_regression_tests::rollup_pages_contract(db)
                    .await
                    .expect("bounded rollup pages"),
                12 => crate::db_regression_tests::concurrent_model_alias_contract(db, db)
                    .await
                    .expect("atomic model aliases"),
                13 => crate::db_regression_tests::key_cache_controls_contract(db)
                    .await
                    .expect("cache-control key CRUD"),
                14 => crate::db_regression_tests::budget_chunks_contract(db)
                    .await
                    .expect("set-based budget deletes"),
                _ => unreachable!(),
            }
        }
        fixture.cleanup().await;
    }
}
