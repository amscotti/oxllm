//! DB regressions for rolling writes, bounded rollups, alias claims and CRUD.
//! The same contracts are exercised by the opt-in isolated Postgres fixtures.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use sqlx::Row;

use super::*;
use crate::binds::bind_optional_f64;

async fn sqlite() -> Result<Arc<Database>> {
    let db = Database::connect("sqlite::memory:").await?;
    db.migrate().await?;
    Ok(db)
}

fn key(id: &str) -> KeyRecord {
    KeyRecord {
        key_id: id.into(),
        key_hash: format!("hash-{id}"),
        key_alias: None,
        team_id: None,
        user_id: None,
        org_id: None,
        project_id: None,
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now_iso(),
        updated_at: now_iso(),
        expires_at: None,
        is_active: true,
        models: None,
        allowed_cache_controls: None,
    }
}

fn log(timestamp: &str, request_id: &str) -> SpendLog {
    SpendLog {
        deployment_id: None,
        requested_model: None,
        model_group: None,
        id: None,
        timestamp: timestamp.into(),
        request_id: Some(request_id.into()),
        model: "model".into(),
        provider: "provider".into(),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        cost: Some(0.125),
        key_id: None,
        team_id: None,
        user_id: None,
        end_user_id: None,
        tag_name: None,
        tags: None,
        status: None,
        latency_ms: None,
    }
}

/// Emulate old writers/imports: intentionally omit metadata_version and all
/// membership writes, even when an importer supplies a durable event identity.
async fn raw_logs(db: &Database, logs: &[SpendLog]) -> Result<()> {
    const COLS: usize = 11;
    for chunk in logs.chunks(db.param_limit() / COLS) {
        let values = (0..chunk.len())
            .map(|i| {
                let fields = (0..COLS)
                    .map(|j| db.ph(i * COLS + j + 1))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({fields})")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let q = format!(
            "INSERT INTO spend_logs (timestamp, request_id, model, provider, cost, key_id, \
             team_id, user_id, tags, tag_name, event_id) VALUES {values}"
        );
        let mut query = sqlx::query(&q);
        for row in chunk {
            query = query
                .bind(&row.timestamp)
                .bind(row.request_id.as_deref())
                .bind(&row.model)
                .bind(&row.provider);
            query = bind_optional_f64(query, row.cost)
                .bind(row.key_id.as_deref())
                .bind(row.team_id.as_deref())
                .bind(row.user_id.as_deref())
                .bind(row.tags.as_deref())
                .bind(row.tag_name.as_deref())
                .bind(uuid::Uuid::new_v4().to_string());
        }
        query.execute(db.pool()).await?;
    }
    Ok(())
}

pub(crate) async fn rolling_tag_writes_contract(db: &Database) -> Result<()> {
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
    ];
    let mut rows = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let mut row = log("2026-07-31T23:59:59.5Z", &format!("literal-{i}"));
        row.tags = Some(serde_json::to_string(&["primary", *name, *name])?);
        row.tag_name = Some("outside-primary".into());
        rows.push(row);
    }
    for (id, tags, primary) in [
        ("escaped", Some(r#"["\u0050rod"]"#), None),
        ("malformed", Some("not JSON"), Some("app%")),
        ("mixed", Some(r#"["ignored",42]"#), Some("mixed-fallback")),
        (
            "object",
            Some(r#"{"ignored":"object"}"#),
            Some("object-fallback"),
        ),
        ("empty", Some(r#"[""]"#), Some("empty-fallback")),
        ("untagged", Some("null"), None),
    ] {
        let mut row = log("2026-07-31T23:59:59Z", id);
        row.tags = tags.map(str::to_string);
        row.tag_name = primary.map(str::to_string);
        rows.push(row);
    }
    raw_logs(db, &rows).await?;
    let indexed: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_log_tags")
        .fetch_one(db.pool())
        .await?;
    assert_eq!(
        indexed.0, 0,
        "the live fallback must work before any startup repair"
    );
    for name in names
        .into_iter()
        .chain(["mixed-fallback", "object-fallback", "empty-fallback"])
    {
        let count = if name == "Prod" || name == "app%" {
            2
        } else {
            1
        };
        let filter = SpendFilter {
            start_date: Some("2026-07-31".into()),
            end_date: Some("2026-07-31".into()),
            tag_name: Some(name.into()),
            ..Default::default()
        };
        assert_eq!(db.query_spend(&filter).await?.len(), count as usize);
        assert_eq!(
            db.query_spend_paginated(None, 100, &filter).await?.len(),
            count as usize
        );
        assert_eq!(db.count_spend_filtered(&filter).await?, count);
        assert_eq!(
            db.total_spend_filtered(&filter).await?,
            count as f64 * 0.125
        );
    }
    for absent in ["", "APP%", "PROD", "outside-primary", "ignored"] {
        assert_eq!(
            db.count_spend_filtered(&SpendFilter {
                tag_name: Some(absent.into()),
                ..Default::default()
            })
            .await?,
            0
        );
    }
    let groups = db.query_spend_by_tag_paginated(None, None, 100, 0).await?;
    assert!(!groups.has_more);
    let totals: BTreeMap<_, _> = groups
        .data
        .into_iter()
        .map(|r| (r.tag_name, (r.total_spend, r.total_requests)))
        .collect();
    assert_eq!(
        totals.get(&Some("primary".into())),
        Some(&(11.0 * 0.125, 11))
    );
    assert_eq!(totals.get(&Some("Prod".into())), Some(&(0.25, 2)));
    assert_eq!(
        totals.get(&None),
        Some(&(0.125, 1)),
        "tagged old rows must not fall into the null group"
    );

    let repaired = db.repair_spend_metadata_batch(256).await?;
    assert_eq!(repaired.repaired_rows, rows.len());
    let update = format!(
        "UPDATE spend_logs SET tags = {}, tag_name = {}, timestamp = {} WHERE request_id = 'literal-0'",
        db.ph(1),
        db.ph(2),
        db.ph(3)
    );
    sqlx::query(&update)
        .bind(r#"["new","\u0050rod","new"]"#)
        .bind("outside-primary")
        .bind("2026-07-31T23:59:59.999999999-01:00")
        .execute(db.pool())
        .await?;
    let stale: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM spend_log_tags st JOIN spend_logs s ON s.id = st.spend_log_id \
         WHERE s.request_id = 'literal-0' AND st.tag_name = 'app%'",
    )
    .fetch_one(db.pool())
    .await?;
    assert_eq!(
        stale.0, 1,
        "exercise a stale index rather than only missing memberships"
    );
    assert_eq!(
        db.count_spend_filtered(&SpendFilter {
            tag_name: Some("app%".into()),
            ..Default::default()
        })
        .await?,
        1
    );
    assert_eq!(
        db.total_spend_filtered(&SpendFilter {
            tag_name: Some("new".into()),
            end_date: Some("2026-07-31".into()),
            ..Default::default()
        })
        .await?,
        0.0
    );
    assert_eq!(
        db.total_spend_filtered(&SpendFilter {
            tag_name: Some("new".into()),
            start_date: Some("2026-08-01".into()),
            end_date: Some("2026-08-01".into()),
            ..Default::default()
        })
        .await?,
        0.125
    );
    let changed = db
        .query_spend_by_request_id("literal-0")
        .await?
        .ok_or_else(|| anyhow!("changed row missing"))?;
    assert_eq!(changed.timestamp, "2026-08-01T00:59:59.999999999Z");
    assert_eq!(db.repair_spend_metadata_batch(256).await?.repaired_rows, 1);
    let stale: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM spend_log_tags st JOIN spend_logs s ON s.id = st.spend_log_id \
         WHERE s.request_id = 'literal-0' AND st.tag_name = 'app%'",
    )
    .fetch_one(db.pool())
    .await?;
    assert_eq!(stale.0, 0);
    Ok(())
}

#[tokio::test]
async fn live_old_and_imported_rows_match_exact_tags() -> Result<()> {
    let db = sqlite().await?;
    rolling_tag_writes_contract(&db).await
}

pub(crate) async fn rolling_timestamp_contract(db: &Database) -> Result<()> {
    let timestamps = [
        "2026-07-31T23:59:59Z",
        "2026-07-31T23:59:59.000000001Z",
        "2026-07-31T23:59:59.1Z",
        "2026-07-31T23:59:59.10Z",
        "2026-07-31T23:59:59.100000001Z",
        "2026-08-01T00:59:59.999999999+01:00",
        "2026-07-31T22:59:59.999999999-01:00",
        "2026-07-31T23:59:59.999999999-00:01",
        "2026-08-01T00:00:00Z",
    ];
    let rows: Vec<_> = timestamps
        .iter()
        .enumerate()
        .map(|(i, ts)| {
            let mut row = log(ts, &format!("clock-{i}"));
            row.key_id = Some("clock-key".into());
            row
        })
        .collect();
    raw_logs(db, &rows).await?;
    let mut expected = Vec::new();
    for row in &rows {
        let request_id = row
            .request_id
            .as_deref()
            .ok_or_else(|| anyhow!("fixture request id missing"))?;
        let stored = db
            .query_spend_by_request_id(request_id)
            .await?
            .ok_or_else(|| anyhow!("fixture row missing"))?;
        assert_eq!(stored.timestamp, normalize_iso(&row.timestamp));
        expected.push((
            normalize_iso(&row.timestamp),
            stored.id.ok_or_else(|| anyhow!("fixture row id missing"))?,
        ));
    }
    expected.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    let actual: Vec<_> = db
        .query_spend(&SpendFilter::default())
        .await?
        .into_iter()
        .map(|r| (r.timestamp, r.id))
        .collect();
    assert_eq!(
        actual,
        expected
            .iter()
            .map(|(ts, id)| (ts.clone(), Some(*id)))
            .collect::<Vec<_>>()
    );
    let filter = SpendFilter {
        start_date: Some("2026-07-31T23:59:59.1000Z".into()),
        end_date: Some("2026-07-31T23:59:59.1Z".into()),
        ..Default::default()
    };
    assert_eq!(db.count_spend_filtered(&filter).await?, 2);
    assert_eq!(db.total_spend_filtered(&filter).await?, 0.25);
    assert_eq!(db.query_spend(&filter).await?.len(), 2);
    let date_filter = SpendFilter {
        start_date: Some("2026-07-31".into()),
        end_date: Some("2026-07-31".into()),
        ..Default::default()
    };
    assert_eq!(db.count_spend_filtered(&date_filter).await?, 7);
    assert_eq!(
        db.total_spend(Some("2026-07-31"), Some("2026-07-31"))
            .await?,
        7.0 * 0.125
    );
    assert_eq!(
        db.query_spend_by_key_paginated(Some("2026-07-31"), Some("2026-07-31"), 1, 0)
            .await?
            .data[0]
            .last_used
            .as_deref(),
        Some("2026-07-31T23:59:59.999999999Z")
    );
    let asc = db
        .query_spend_paginated(
            None,
            100,
            &SpendFilter {
                sort_by: Some("timestamp".into()),
                sort_order: Some("asc".into()),
                ..Default::default()
            },
        )
        .await?;
    assert_eq!(
        asc.into_iter().map(|r| r.id).collect::<Vec<_>>(),
        expected
            .iter()
            .rev()
            .map(|(_, id)| Some(*id))
            .collect::<Vec<_>>()
    );
    let since = "2026-07-31T23:59:59.100000001Z";
    assert_eq!(db.count_spend(Some("clock-key"), Some(since)).await?, 5);
    assert_eq!(
        db.get_total_spend(Some("clock-key"), Some(since)).await?,
        5.0 * 0.125
    );
    assert_eq!(
        db.delete_spend_logs_before_batch("2026-07-31T23:59:59.1Z", 100)
            .await?,
        2
    );
    assert_eq!(db.count_spend(None, None).await?, 7);
    extended_timestamp_contract(db).await
}

async fn extended_timestamp_contract(db: &Database) -> Result<()> {
    let timestamps = [
        "2026-07-31T23:59:60.999999999Z",
        "2026-07-31T23:59:59,123456789Z",
        "2026-08-01T00:00:29.987654321+00:00:30",
        "2026-07-31T23:59:29.987654321-000030",
        "2026-08-01T00:59:59.000000001+01",
        "2026-08-01T00:59:59.1+0100",
        "20260731T235959.1Z",
        "2026-07-31T23:59Z",
        "2026-07-31T23Z",
        "2026-08-01T00:59+01[UTC]",
    ];
    let rows: Vec<_> = timestamps
        .iter()
        .enumerate()
        .map(|(i, ts)| {
            let mut row = log(ts, &format!("extended-clock-{i}"));
            row.key_id = Some("extended-clock".into());
            row
        })
        .collect();
    raw_logs(db, &rows).await?;
    let mut expected = Vec::new();
    for (i, timestamp) in timestamps.iter().enumerate() {
        let canonical = normalize_iso(timestamp);
        assert_ne!(
            canonical, *timestamp,
            "fixture must exercise a parseable noncanonical timestamp"
        );
        let row = db
            .query_spend_by_request_id(&format!("extended-clock-{i}"))
            .await?
            .ok_or_else(|| anyhow!("extended timestamp row missing"))?;
        assert_eq!(row.timestamp, canonical);
        expected.push((canonical, row.id));
    }
    expected.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    let filter = SpendFilter {
        key_id: Some("extended-clock".into()),
        start_date: Some("2026-07-31".into()),
        end_date: Some("2026-07-31".into()),
        ..Default::default()
    };
    assert_eq!(
        db.query_spend(&filter)
            .await?
            .into_iter()
            .map(|r| (r.timestamp, r.id))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        db.count_spend_filtered(&filter).await?,
        timestamps.len() as i64
    );
    assert_eq!(
        db.total_spend_filtered(&filter).await?,
        timestamps.len() as f64 * 0.125
    );
    let exact = SpendFilter {
        key_id: Some("extended-clock".into()),
        start_date: Some("2026-07-31T23:59:59.987654321Z".into()),
        end_date: Some("2026-07-31T23:59:59.987654321Z".into()),
        ..Default::default()
    };
    assert_eq!(db.count_spend_filtered(&exact).await?, 2);
    Ok(())
}

#[tokio::test]
async fn live_timestamp_reads_order_nanoseconds_with_inclusive_bounds() -> Result<()> {
    let db = sqlite().await?;
    rolling_timestamp_contract(&db).await
}

pub(crate) async fn bounded_metadata_contract(db: &Database) -> Result<()> {
    let rows: Vec<_> = (0..1_105)
        .map(|i| {
            let mut row = log("2026-07-31T23:59:59.5Z", &format!("repair-{i}"));
            row.tags = Some(r#"["Prod%","secondary","Prod%"]"#.into());
            row.tag_name = Some("Prod%".into());
            row
        })
        .collect();
    raw_logs(db, &rows).await?;
    db.migrate().await?;
    let pending: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM spend_logs WHERE metadata_version = 0")
            .fetch_one(db.pool())
            .await?;
    assert_eq!(
        pending.0, 81,
        "startup must stop after four committed 256-row batches"
    );
    let step = db.repair_spend_metadata_batch(-1).await?;
    assert_eq!(
        (step.scanned_rows, step.repaired_rows, step.has_more),
        (1, 1, true)
    );
    // A normal writer can commit between maintenance batches.
    db.insert_spend_log(&log("2026-08-01T00:00:00Z", "writer-between-batches"))
        .await?;
    let filter = SpendFilter {
        tag_name: Some("secondary".into()),
        ..Default::default()
    };
    assert_eq!(db.count_spend_filtered(&filter).await?, 1_105);
    assert_eq!(db.total_spend_filtered(&filter).await?, 138.125);
    db.migrate().await?;
    let pending: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM spend_logs WHERE metadata_version = 0")
            .fetch_one(db.pool())
            .await?;
    assert_eq!(
        pending.0, 0,
        "the persisted cursor resumes beyond the first startup's prefix"
    );
    let memberships: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_log_tags")
        .fetch_one(db.pool())
        .await?;
    assert_eq!(memberships.0, 2_210);
    let step = db.repair_spend_metadata_batch(i64::MAX).await?;
    assert_eq!((step.scanned_rows, step.repaired_rows), (256, 0));
    assert_eq!(db.count_spend_filtered(&filter).await?, 1_105);
    Ok(())
}

#[tokio::test]
async fn startup_repairs_are_bounded_committed_resumable_and_idempotent() -> Result<()> {
    let db = sqlite().await?;
    bounded_metadata_contract(&db).await
}

#[tokio::test]
async fn concurrent_sqlite_repairs_share_a_committed_cursor() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let url = format!(
        "sqlite:{}?mode=rwc",
        directory.path().join("repair.db").display()
    );
    let a = Database::connect(&url).await?;
    a.migrate().await?;
    let b = Database::connect(&url).await?;
    let rows: Vec<_> = (0..600)
        .map(|i| {
            let mut row = log("2026-07-31T23:59:59.5Z", &format!("replica-{i}"));
            row.tags = Some(r#"["one","two","one"]"#.into());
            row
        })
        .collect();
    raw_logs(&a, &rows).await?;
    let (first, second) = tokio::join!(
        a.repair_spend_metadata_batch(128),
        b.repair_spend_metadata_batch(128)
    );
    assert_eq!(first?.repaired_rows + second?.repaired_rows, 256);
    b.insert_spend_log(&log("2026-08-01T00:00:00Z", "concurrent-writer"))
        .await?;
    let (first, second) = tokio::join!(a.migrate(), b.migrate());
    first?;
    second?;
    let memberships: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_log_tags")
        .fetch_one(a.pool())
        .await?;
    assert_eq!(memberships.0, 1_200);
    assert_eq!(
        a.count_spend_filtered(&SpendFilter {
            tag_name: Some("two".into()),
            ..Default::default()
        })
        .await?,
        600
    );
    a.pool().close().await;
    b.pool().close().await;
    Ok(())
}

#[derive(Clone, Copy)]
enum Dimension {
    Key,
    Model,
    User,
    Team,
    Org,
}

type GroupRow = (Option<String>, Option<String>, f64, i64);

fn map_page<T>(page: SpendPage<T>, map: impl FnMut(T) -> GroupRow) -> SpendPage<GroupRow> {
    SpendPage {
        data: page.data.into_iter().map(map).collect(),
        limit: page.limit,
        offset: page.offset,
        has_more: page.has_more,
        next_offset: page.next_offset,
    }
}

async fn group_page(
    db: &Database,
    dimension: Dimension,
    limit: i64,
    offset: i64,
) -> Result<SpendPage<GroupRow>> {
    let start = Some("2026-07-31");
    let end = Some("2026-07-31");
    Ok(match dimension {
        Dimension::Key => map_page(
            db.query_spend_by_key_paginated(start, end, limit, offset)
                .await?,
            |r| (r.key_id, None, r.total_spend, r.total_requests),
        ),
        Dimension::Model => map_page(
            db.query_spend_by_model_paginated(start, end, limit, offset)
                .await?,
            |r| (r.model, r.provider, r.total_spend, r.total_requests),
        ),
        Dimension::User => map_page(
            db.query_spend_by_user_paginated(start, end, limit, offset)
                .await?,
            |r| (r.user_id, None, r.total_spend, r.total_requests),
        ),
        Dimension::Team => map_page(
            db.query_spend_by_team_paginated(start, end, limit, offset)
                .await?,
            |r| (r.team_id, None, r.total_spend, r.total_requests),
        ),
        Dimension::Org => map_page(
            db.query_spend_by_org_paginated(start, end, limit, offset)
                .await?,
            |r| (r.org_id, None, r.total_spend, r.total_requests),
        ),
    })
}

pub(crate) async fn rollup_pages_contract(db: &Database) -> Result<()> {
    const GROUPS: usize = 1_005;
    let ids: Vec<_> = (0..GROUPS).map(|i| format!("group-{i:04}")).collect();
    // Each key resolves to a distinct organization, without requiring unrelated
    // management fixtures for the other joined display-name tables.
    for chunk in ids.chunks(db.param_limit() / 5) {
        let values = (0..chunk.len())
            .map(|i| {
                let fields = (0..5)
                    .map(|j| db.ph(i * 5 + j + 1))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({fields})")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let q = format!(
            "INSERT INTO keys (key_id, key_hash, org_id, created_at, updated_at) VALUES {values}"
        );
        let mut query = sqlx::query(&q);
        for id in chunk {
            query = query
                .bind(id)
                .bind(format!("hash-{id}"))
                .bind(id)
                .bind("2026-07-31T00:00:00Z")
                .bind("2026-07-31T00:00:00Z");
        }
        query.execute(db.pool()).await?;
    }
    let mut rows = Vec::new();
    for id in &ids {
        for (suffix, timestamp) in [
            ("first", "2026-07-31T00:00:00Z"),
            ("last", "2026-07-31T23:59:59.999999999Z"),
        ] {
            let mut row = log(timestamp, &format!("{id}-{suffix}"));
            row.model = id.clone();
            row.key_id = Some(id.clone());
            row.user_id = Some(id.clone());
            row.team_id = Some(id.clone());
            rows.push(row);
        }
    }
    let mut null = log("2026-07-31T12:00:00Z", "null-group");
    null.model = "zz-null".into();
    null.cost = Some(0.25);
    rows.push(null);
    let mut provider_tie = log("2026-07-31T13:00:00Z", "provider-tie");
    provider_tie.model = ids[0].clone();
    provider_tie.provider = "alpha".into();
    provider_tie.cost = Some(0.25);
    rows.push(provider_tie);
    for timestamp in ["2026-07-30T23:59:59.999999999Z", "2026-08-01T00:00:00Z"] {
        let mut row = log(timestamp, timestamp);
        row.model = ids[0].clone();
        row.key_id = Some(ids[0].clone());
        row.user_id = Some(ids[0].clone());
        row.team_id = Some(ids[0].clone());
        row.cost = Some(100.0);
        rows.push(row);
    }
    db.insert_spend_logs_batch(&rows).await?;
    for dimension in [
        Dimension::Key,
        Dimension::Model,
        Dimension::User,
        Dimension::Team,
        Dimension::Org,
    ] {
        let mut data = Vec::new();
        let mut offset = 0;
        loop {
            let page = group_page(db, dimension, 127, offset).await?;
            assert_eq!((page.limit, page.offset), (127, offset));
            assert!(page.data.len() <= 127);
            if page.has_more {
                assert_eq!(page.next_offset, Some(offset + 127));
            } else {
                assert_eq!(page.next_offset, None);
            }
            data.extend(page.data);
            let Some(next) = page.next_offset else { break };
            offset = next;
        }
        let mut expected: Vec<GroupRow> = ids
            .iter()
            .map(|id| {
                let provider =
                    matches!(dimension, Dimension::Model).then(|| "provider".to_string());
                (Some(id.clone()), provider, 0.25, 2)
            })
            .collect();
        if matches!(dimension, Dimension::Model) {
            expected.insert(0, (Some(ids[0].clone()), Some("alpha".into()), 0.25, 1));
            expected.push((Some("zz-null".into()), Some("provider".into()), 0.25, 1));
        } else {
            expected.insert(0, (None, None, 0.5, 2));
        }
        assert_eq!(
            data, expected,
            "ties must not duplicate, skip or truncate complete groups"
        );
        let unique: BTreeSet<_> = data.iter().map(|r| (&r.0, &r.1)).collect();
        assert_eq!(unique.len(), data.len());
        let clamped = group_page(db, dimension, -1, -5).await?;
        assert_eq!(
            (clamped.limit, clamped.offset, clamped.data.len()),
            (1, 0, 1)
        );
        assert!(clamped.has_more);
        let maximum = group_page(db, dimension, i64::MAX, 0).await?;
        assert_eq!(
            (maximum.limit, maximum.data.len(), maximum.next_offset),
            (1_000, 1_000, Some(1_000))
        );
        let default = group_page(db, dimension, SPEND_ROLLUP_DEFAULT_LIMIT, 0).await?;
        assert_eq!(default.data.len(), 100);
        let empty = group_page(db, dimension, 100, i64::MAX).await?;
        assert!(empty.data.is_empty());
        assert!(!empty.has_more);
        assert_eq!(empty.next_offset, None);
        let unpaged = match dimension {
            Dimension::Key => db.query_spend_by_key(None, None).await.map(|_| ()),
            Dimension::Model => db.query_spend_by_model(None, None).await.map(|_| ()),
            Dimension::User => db.query_spend_by_user(None, None).await.map(|_| ()),
            Dimension::Team => db.query_spend_by_team(None, None).await.map(|_| ()),
            Dimension::Org => db.query_spend_by_org(None, None).await.map(|_| ()),
        };
        assert!(unpaged.is_err_and(|e| e.downcast_ref::<SpendRollupLimitExceeded>().is_some()));
    }
    let key_page = db
        .query_spend_by_key_paginated(Some("2026-07-31"), Some("2026-07-31"), 1, 1)
        .await?;
    assert_eq!(
        key_page.data[0].last_used.as_deref(),
        Some("2026-07-31T23:59:59.999999999Z")
    );
    assert_eq!(
        db.total_spend(Some("2026-07-31"), Some("2026-07-31"))
            .await?,
        GROUPS as f64 * 0.25 + 0.5
    );
    assert_eq!(
        db.count_spend_filtered(&SpendFilter {
            start_date: Some("2026-07-31".into()),
            end_date: Some("2026-07-31".into()),
            ..Default::default()
        })
        .await?,
        (GROUPS * 2 + 2) as i64
    );
    Ok(())
}

#[tokio::test]
async fn rollup_pages_are_bounded_complete_and_stably_ordered() -> Result<()> {
    let db = sqlite().await?;
    rollup_pages_contract(&db).await
}

fn model(id: &str, name: &str) -> ModelConfigRecord {
    ModelConfigRecord {
        model_id: id.into(),
        model_name: name.into(),
        oxllm_params: serde_json::json!({"model": "openai/gpt-4o", "fixture_id": id}),
        model_info: None,
        created_at: "2026-07-31T00:00:00.000000000Z".into(),
        updated_at: "2026-07-31T00:00:00.000000000Z".into(),
    }
}

pub(crate) async fn concurrent_model_alias_contract(a: &Database, b: &Database) -> Result<()> {
    let left = model("create-a", "same-alias");
    let right = model("create-z", "same-alias");
    let barrier = tokio::sync::Barrier::new(2);
    let (first, second) = tokio::join!(
        async {
            barrier.wait().await;
            a.create_model(&left).await
        },
        async {
            barrier.wait().await;
            b.create_model(&right).await
        },
    );
    assert_eq!(u8::from(first.is_ok()) + u8::from(second.is_ok()), 1);
    let conflict = first
        .err()
        .or_else(|| second.err())
        .ok_or_else(|| anyhow!("expected alias conflict"))?;
    assert_eq!(
        conflict
            .downcast_ref::<ModelNameConflict>()
            .map(|e| e.model_name.as_str()),
        Some("same-alias")
    );
    let rows = a.list_all_models_db().await?;
    assert_eq!(
        rows.len(),
        1,
        "losing concurrent create must leave no orphan row"
    );
    assert_eq!(
        rows[0].oxllm_params["fixture_id"].as_str(),
        Some(rows[0].model_id.as_str())
    );

    let old_a = model("rename-a", "old-a");
    let old_b = model("rename-z", "old-z");
    a.create_model(&old_a).await?;
    b.create_model(&old_b).await?;
    let mut new_a = old_a.clone();
    let mut new_b = old_b.clone();
    new_a.model_name = "renamed-alias".into();
    new_b.model_name = "renamed-alias".into();
    new_a.oxllm_params = serde_json::json!({"winner": "a"});
    new_b.oxllm_params = serde_json::json!({"winner": "z"});
    let barrier = tokio::sync::Barrier::new(2);
    let (first, second) = tokio::join!(
        async {
            barrier.wait().await;
            a.update_model(&old_a.model_id, &new_a).await
        },
        async {
            barrier.wait().await;
            b.update_model(&old_b.model_id, &new_b).await
        },
    );
    let first_won = first.is_ok();
    assert_eq!(u8::from(first_won) + u8::from(second.is_ok()), 1);
    let conflict = first
        .err()
        .or_else(|| second.err())
        .ok_or_else(|| anyhow!("expected rename conflict"))?;
    assert!(conflict.downcast_ref::<ModelNameConflict>().is_some());
    let loser = if first_won { &old_b } else { &old_a };
    let persisted = a
        .get_model(&loser.model_id)
        .await?
        .ok_or_else(|| anyhow!("loser row missing"))?;
    assert_eq!(persisted.model_name, loser.model_name);
    assert_eq!(
        persisted.oxllm_params, loser.oxllm_params,
        "conflict must roll back the whole mutation"
    );
    assert!(
        a.create_model(&model("blocked", &loser.model_name))
            .await
            .is_err_and(|e| e.downcast_ref::<ModelNameConflict>().is_some())
    );

    // A primary-key failure must roll back its newly reserved alias too.
    let duplicate_id = model(&loser.model_id, "failed-id-alias");
    assert!(a.create_model(&duplicate_id).await.is_err());
    a.create_model(&model("after-failed-id", "failed-id-alias"))
        .await?;
    a.create_model(&model("case-a", "Case-Alias")).await?;
    b.create_model(&model("case-z", "case-alias")).await?;
    let all = a.list_all_models_db().await?;
    let actual: Vec<_> = all.iter().map(|r| r.model_id.clone()).collect();
    let mut expected = actual.clone();
    expected.sort_by(|a, b| b.cmp(a));
    assert_eq!(
        actual, expected,
        "equal creation times must use model_id DESC on every replica"
    );
    assert_eq!(
        a.list_models_db(Some(2))
            .await?
            .into_iter()
            .map(|r| r.model_id)
            .collect::<Vec<_>>(),
        expected[..2].to_vec()
    );
    assert_eq!(a.delete_model("after-failed-id").await?, 1);
    b.create_model(&model("alias-reused", "failed-id-alias"))
        .await?;
    Ok(())
}

#[tokio::test]
async fn concurrent_sqlite_creates_and_renames_claim_model_aliases_atomically() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let url = format!(
        "sqlite:{}?mode=rwc",
        directory.path().join("models.db").display()
    );
    let a = Database::connect(&url).await?;
    a.migrate().await?;
    let b = Database::connect(&url).await?;
    concurrent_model_alias_contract(&a, &b).await?;
    a.pool().close().await;
    b.pool().close().await;
    Ok(())
}

/// Called on an unmigrated database, using the frozen historical schema.
pub(crate) async fn legacy_model_duplicates_contract(db: &Database) -> Result<()> {
    crate::release_tests::load_v1(db).await;
    let q = format!(
        "INSERT INTO models (model_id, model_name, litellm_params, model_info, created_at, updated_at) \
         VALUES ({}, 'legacy-alias', {}, NULL, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
        db.ph(1),
        db.ph(2)
    );
    for id in ["legacy-a", "legacy-z"] {
        sqlx::query(&q)
            .bind(id)
            .bind(serde_json::to_string(
                &serde_json::json!({"fixture_id": id}),
            )?)
            .execute(db.pool())
            .await?;
    }
    db.migrate().await?;
    db.migrate().await?;
    let duplicates: Vec<_> = db
        .list_all_models_db()
        .await?
        .into_iter()
        .filter(|r| r.model_name == "legacy-alias")
        .collect();
    assert_eq!(
        duplicates
            .iter()
            .map(|r| r.model_id.as_str())
            .collect::<Vec<_>>(),
        ["legacy-z", "legacy-a"]
    );
    for row in &duplicates {
        assert_eq!(
            row.oxllm_params["fixture_id"].as_str(),
            Some(row.model_id.as_str())
        );
    }
    let mut edit = duplicates[0].clone();
    edit.model_info = Some(serde_json::json!({"edited": true}));
    db.update_model(&edit.model_id, &edit).await?;
    assert!(
        db.create_model(&model("new-duplicate", "legacy-alias"))
            .await
            .is_err_and(|e| e.downcast_ref::<ModelNameConflict>().is_some())
    );
    // Concurrent deletion of the last legacy duplicates must release the name;
    // Postgres release decisions serialize on the claim row before rechecking.
    let (first, second) = tokio::join!(db.delete_model("legacy-a"), db.delete_model("legacy-z"));
    assert_eq!(first? + second?, 2);
    db.create_model(&model("legacy-reused", "legacy-alias"))
        .await?;
    let mut rename = model("legacy-reused", "free-renamed-alias");
    rename.model_info = Some(serde_json::json!({"rename": true}));
    db.update_model("legacy-reused", &rename).await?;
    db.create_model(&model("legacy-reused-again", "legacy-alias"))
        .await?;
    assert!(
        db.get_model("v1-model").await?.is_some(),
        "migration must preserve unrelated models"
    );
    Ok(())
}

#[tokio::test]
async fn legacy_duplicate_models_survive_and_release_their_last_alias() -> Result<()> {
    let db = Database::connect("sqlite::memory:").await?;
    legacy_model_duplicates_contract(&db).await
}

pub(crate) async fn key_cache_controls_contract(db: &Database) -> Result<()> {
    let mut record = key("cache-policy");
    record.models = Some(r#"["gpt-4o"]"#.into());
    record.allowed_cache_controls = Some(r#"["no-cache","no-store"]"#.into());
    let serialized = serde_json::to_value(&record)?;
    assert_eq!(
        serialized["allowed_cache_controls"],
        r#"["no-cache","no-store"]"#
    );
    let roundtrip: KeyRecord = serde_json::from_value(serialized.clone())?;
    assert_eq!(
        roundtrip.allowed_cache_controls,
        record.allowed_cache_controls
    );
    let mut old = serialized;
    old.as_object_mut()
        .ok_or_else(|| anyhow!("serialized key must be an object"))?
        .remove("allowed_cache_controls");
    let old: KeyRecord = serde_json::from_value(old)?;
    assert!(old.allowed_cache_controls.is_none());
    assert!(
        serde_json::to_value(&old)?
            .get("allowed_cache_controls")
            .is_none()
    );
    db.create_key(&record).await?;
    let by_id = db
        .get_key(&record.key_id)
        .await?
        .ok_or_else(|| anyhow!("key missing"))?;
    let by_hash = db
        .get_key_by_hash(&record.key_hash)
        .await?
        .ok_or_else(|| anyhow!("key hash missing"))?;
    let listed = db.list_keys(None).await?;
    assert_eq!(by_id.allowed_cache_controls, record.allowed_cache_controls);
    assert_eq!(
        by_hash.allowed_cache_controls,
        record.allowed_cache_controls
    );
    assert_eq!(
        listed[0].allowed_cache_controls,
        record.allowed_cache_controls
    );
    record.allowed_cache_controls = Some(r#"["private"]"#.into());
    assert!(db.update_key(&record).await?);
    db.rotate_key(&record.key_id, "rotated-cache-hash", &now_iso())
        .await?;
    let rotated = db
        .get_key_by_hash("rotated-cache-hash")
        .await?
        .ok_or_else(|| anyhow!("rotated key missing"))?;
    assert_eq!(
        rotated.allowed_cache_controls,
        record.allowed_cache_controls
    );
    record.allowed_cache_controls = None;
    assert!(db.update_key(&record).await?);
    let cleared = db
        .get_key(&record.key_id)
        .await?
        .ok_or_else(|| anyhow!("cleared key missing"))?;
    assert!(cleared.allowed_cache_controls.is_none());
    assert_eq!(cleared.models.as_deref(), Some(r#"["gpt-4o"]"#));
    let null: (Option<String>,) =
        sqlx::query_as("SELECT allowed_cache_controls FROM keys WHERE key_id = 'cache-policy'")
            .fetch_one(db.pool())
            .await?;
    assert_eq!(null.0, None);
    sqlx::query("INSERT INTO keys (key_id, key_hash, created_at, updated_at) VALUES ('imported-key', 'imported-hash', '2026-07-31T00:00:00Z', '2026-07-31T00:00:00Z')").execute(db.pool()).await?;
    assert!(
        db.get_key("imported-key")
            .await?
            .ok_or_else(|| anyhow!("imported key missing"))?
            .allowed_cache_controls
            .is_none()
    );
    Ok(())
}

#[tokio::test]
async fn cache_control_policy_roundtrips_serialization_and_crud() -> Result<()> {
    let db = sqlite().await?;
    key_cache_controls_contract(&db).await
}

fn budget(id: &str) -> BudgetRecord {
    BudgetRecord {
        budget_id: id.into(),
        max_budget: Some(10.0),
        soft_budget: None,
        max_parallel_requests: None,
        tpm_limit: None,
        rpm_limit: None,
        budget_duration: None,
        model_max_budget: None,
        created_at: now_iso(),
        updated_at: now_iso(),
    }
}

pub(crate) async fn budget_chunks_contract(db: &Database) -> Result<()> {
    for id in ["first-budget", "last-budget", "reject-budget"] {
        db.create_budget(&budget(id)).await?;
    }
    let limit = db.param_limit();
    let mut ids: Vec<_> = (0..limit + 2).map(|i| format!("missing-{i}")).collect();
    ids[0] = "first-budget".into();
    ids[limit - 1] = "first-budget".into();
    ids[limit] = "last-budget".into();
    ids[limit + 1] = "reject-budget".into();
    match db.placeholder_style() {
        PlaceholderStyle::Question => {
            sqlx::query("CREATE TRIGGER reject_budget_delete BEFORE DELETE ON budgets WHEN OLD.budget_id = 'reject-budget' BEGIN SELECT RAISE(ABORT, 'budget rejected'); END").execute(db.pool()).await?;
        }
        PlaceholderStyle::Dollar => {
            sqlx::query("CREATE FUNCTION reject_budget_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'budget rejected'; END; $$").execute(db.pool()).await?;
            sqlx::query("CREATE TRIGGER reject_budget_delete BEFORE DELETE ON budgets FOR EACH ROW WHEN (OLD.budget_id = 'reject-budget') EXECUTE FUNCTION reject_budget_delete_fn()").execute(db.pool()).await?;
            sqlx::query("CREATE TABLE budget_delete_statements (marker INTEGER NOT NULL)")
                .execute(db.pool())
                .await?;
            sqlx::query("CREATE FUNCTION count_budget_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN INSERT INTO budget_delete_statements VALUES (1); RETURN NULL; END; $$").execute(db.pool()).await?;
            sqlx::query("CREATE TRIGGER count_budget_delete AFTER DELETE ON budgets FOR EACH STATEMENT EXECUTE FUNCTION count_budget_delete_fn()").execute(db.pool()).await?;
        }
    }
    assert!(db.delete_budgets(&ids).await.is_err());
    for id in ["first-budget", "last-budget", "reject-budget"] {
        assert!(
            db.get_budget(id).await?.is_some(),
            "a later chunk failure must roll back earlier deletes"
        );
    }
    let drop_trigger = match db.placeholder_style() {
        PlaceholderStyle::Question => "DROP TRIGGER reject_budget_delete",
        PlaceholderStyle::Dollar => "DROP TRIGGER reject_budget_delete ON budgets",
    };
    sqlx::query(drop_trigger).execute(db.pool()).await?;
    assert_eq!(db.delete_budgets(&ids).await?, 3);
    if matches!(db.placeholder_style(), PlaceholderStyle::Dollar) {
        let statements: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM budget_delete_statements")
            .fetch_one(db.pool())
            .await?;
        assert_eq!(
            statements.0, 2,
            "set-based deletes must issue one statement per bind-limited chunk"
        );
    }
    assert_eq!(db.delete_budgets(&ids).await?, 0);
    assert_eq!(db.delete_budgets(&[]).await?, 0);
    db.create_budget(&budget("single-budget")).await?;
    db.delete_budget("single-budget").await?;
    db.delete_budget("single-budget").await?;
    assert!(db.get_budget("single-budget").await?.is_none());
    Ok(())
}

#[tokio::test]
async fn budget_delete_chunks_are_atomic_and_count_distinct_rows() -> Result<()> {
    let db = sqlite().await?;
    budget_chunks_contract(&db).await
}

/// Run on a one-connection pool so every NULL/value transition reuses the same
/// prepared statements. Precision deliberately exceeds f32, and NULL remains
/// NULL: substituting zero or narrowing real values cannot satisfy this test.
pub(crate) async fn nullable_float_bind_contract(db: &Database) -> Result<()> {
    let now = now_iso();
    let mut key = key("nullable-key");
    let mut team = TeamRecord {
        team_id: "nullable-team".into(),
        team_name: "Nullable team".into(),
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now.clone(),
        updated_at: now.clone(),
        models: None,
        blocked: false,
    };
    let mut user = UserRecord {
        user_id: "nullable-user".into(),
        user_name: "Nullable user".into(),
        max_budget: None,
        budget_duration: None,
        created_at: now.clone(),
        updated_at: now.clone(),
        models: None,
    };
    let mut org = OrganizationRecord {
        org_id: "nullable-org".into(),
        org_name: "Nullable org".into(),
        max_budget: None,
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
    };
    let mut project = ProjectRecord {
        project_id: "nullable-project".into(),
        project_name: "Nullable project".into(),
        org_id: None,
        team_id: None,
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now.clone(),
        updated_at: now.clone(),
    };
    let mut customer = CustomerRecord {
        end_user_id: "nullable-customer".into(),
        alias: None,
        email: None,
        max_budget: None,
        spend: 0.0,
        budget_duration: None,
        blocked: false,
        model_blocklist: None,
        created_at: now.clone(),
        updated_at: now.clone(),
    };
    let mut tag = TagRecord {
        name: "nullable-tag".into(),
        description: None,
        models: None,
        model_info: None,
        spend: None,
        max_budget: None,
        created_at: now.clone(),
        updated_at: now,
    };
    let mut budget = budget("nullable-budget");
    budget.max_budget = None;
    db.create_key(&key).await?;
    db.create_team(&team).await?;
    db.create_user(&user).await?;
    db.create_organization(&org).await?;
    db.create_project(&project).await?;
    db.create_customer(&customer).await?;
    db.create_tag(&tag).await?;
    db.create_budget(&budget).await?;

    let precise = 12_345.678_901_234_567_f64;
    for (index, value) in [None, Some(precise), None, Some(precise + 0.125)]
        .into_iter()
        .enumerate()
    {
        if matches!(db.placeholder_style(), PlaceholderStyle::Dollar) {
            let row = bind_optional_f64(
                sqlx::query("SELECT CAST(pg_typeof($1) AS TEXT) AS parameter_type, CAST($1 AS DOUBLE PRECISION) AS value"),
                value,
            ).fetch_one(db.pool()).await?;
            assert_eq!(
                row.try_get::<String, _>("parameter_type")?,
                "double precision"
            );
            assert_eq!(row.try_get::<Option<f64>, _>("value")?, value);
            let integer = value.map(|_| 42_i64);
            let row = sqlx::query(
                "SELECT CAST(pg_typeof($1) AS TEXT) AS parameter_type, CAST($1 AS BIGINT) AS value",
            )
            .bind(integer)
            .fetch_one(db.pool())
            .await?;
            assert_eq!(row.try_get::<String, _>("parameter_type")?, "bigint");
            assert_eq!(row.try_get::<Option<i64>, _>("value")?, integer);
        }
        let require = || anyhow!("nullable binding fixture row missing");
        let mut inserted_key = key.clone();
        inserted_key.key_id = format!("nullable-key-{index}");
        inserted_key.key_hash = format!("nullable-hash-{index}");
        inserted_key.max_budget = value;
        inserted_key.rpm = value.map(|_| 42_i64);
        db.create_key(&inserted_key).await?;
        let stored = db
            .get_key(&inserted_key.key_id)
            .await?
            .ok_or_else(require)?;
        assert_eq!(stored.max_budget, value);
        assert_eq!(stored.rpm, inserted_key.rpm);
        key.max_budget = value;
        key.rpm = inserted_key.rpm;
        assert!(db.update_key(&key).await?);
        assert_eq!(
            db.get_key(&key.key_id)
                .await?
                .ok_or_else(require)?
                .max_budget,
            value
        );

        let mut inserted_team = team.clone();
        inserted_team.team_id = format!("nullable-team-{index}");
        inserted_team.max_budget = value;
        db.create_team(&inserted_team).await?;
        assert_eq!(
            db.get_team(&inserted_team.team_id)
                .await?
                .ok_or_else(require)?
                .max_budget,
            value
        );
        team.max_budget = value;
        assert!(db.update_team(&team).await?);
        assert_eq!(
            db.get_team(&team.team_id)
                .await?
                .ok_or_else(require)?
                .max_budget,
            value
        );

        let mut inserted_user = user.clone();
        inserted_user.user_id = format!("nullable-user-{index}");
        inserted_user.max_budget = value;
        db.create_user(&inserted_user).await?;
        assert_eq!(
            db.get_user(&inserted_user.user_id)
                .await?
                .ok_or_else(require)?
                .max_budget,
            value
        );
        user.max_budget = value;
        assert!(db.update_user(&user).await?);
        assert_eq!(
            db.get_user(&user.user_id)
                .await?
                .ok_or_else(require)?
                .max_budget,
            value
        );

        let mut inserted_org = org.clone();
        inserted_org.org_id = format!("nullable-org-{index}");
        inserted_org.max_budget = value;
        inserted_org.soft_budget = value;
        db.create_organization(&inserted_org).await?;
        let stored = db
            .get_organization(&inserted_org.org_id)
            .await?
            .ok_or_else(require)?;
        assert_eq!((stored.max_budget, stored.soft_budget), (value, value));
        org.max_budget = value;
        org.soft_budget = value;
        assert!(db.update_organization(&org).await?);
        let stored = db
            .get_organization(&org.org_id)
            .await?
            .ok_or_else(require)?;
        assert_eq!((stored.max_budget, stored.soft_budget), (value, value));

        let mut inserted_project = project.clone();
        inserted_project.project_id = format!("nullable-project-{index}");
        inserted_project.max_budget = value;
        db.create_project(&inserted_project).await?;
        assert_eq!(
            db.get_project(&inserted_project.project_id)
                .await?
                .ok_or_else(require)?
                .max_budget,
            value
        );
        project.max_budget = value;
        assert!(db.update_project(&project).await?);
        assert_eq!(
            db.get_project(&project.project_id)
                .await?
                .ok_or_else(require)?
                .max_budget,
            value
        );

        let mut inserted_customer = customer.clone();
        inserted_customer.end_user_id = format!("nullable-customer-{index}");
        inserted_customer.max_budget = value;
        db.create_customer(&inserted_customer).await?;
        assert_eq!(
            db.get_customer(&inserted_customer.end_user_id)
                .await?
                .ok_or_else(require)?
                .max_budget,
            value
        );
        customer.max_budget = value;
        assert!(db.update_customer(&customer).await?);
        assert_eq!(
            db.get_customer(&customer.end_user_id)
                .await?
                .ok_or_else(require)?
                .max_budget,
            value
        );

        let mut inserted_tag = tag.clone();
        inserted_tag.name = format!("nullable-tag-{index}");
        inserted_tag.max_budget = value;
        inserted_tag.spend = value;
        db.create_tag(&inserted_tag).await?;
        let stored = db.get_tag(&inserted_tag.name).await?.ok_or_else(require)?;
        assert_eq!((stored.max_budget, stored.spend), (value, value));
        tag.max_budget = value;
        assert!(db.update_tag(&tag).await?);
        assert_eq!(
            db.get_tag(&tag.name).await?.ok_or_else(require)?.max_budget,
            value
        );

        let mut inserted_budget = budget.clone();
        inserted_budget.budget_id = format!("nullable-budget-{index}");
        inserted_budget.max_budget = value;
        inserted_budget.soft_budget = value;
        db.create_budget(&inserted_budget).await?;
        let stored = db
            .get_budget(&inserted_budget.budget_id)
            .await?
            .ok_or_else(require)?;
        assert_eq!((stored.max_budget, stored.soft_budget), (value, value));
        budget.max_budget = value;
        budget.soft_budget = value;
        assert!(db.update_budget(&budget).await?);
        let stored = db
            .get_budget(&budget.budget_id)
            .await?
            .ok_or_else(require)?;
        assert_eq!((stored.max_budget, stored.soft_budget), (value, value));

        let mut spend = log(
            "2026-07-31T23:59:59.123456789Z",
            &format!("nullable-cost-{index}"),
        );
        spend.cost = value;
        let event = SpendLogEvent::new(spend);
        assert_eq!(
            db.insert_spend_events(std::slice::from_ref(&event)).await?,
            1
        );
        assert_eq!(
            db.insert_spend_events(std::slice::from_ref(&event)).await?,
            0
        );
        let stored = db
            .query_spend_by_request_id(&format!("nullable-cost-{index}"))
            .await?
            .ok_or_else(require)?;
        assert_eq!(stored.cost, value);
    }
    Ok(())
}

#[tokio::test]
async fn cached_nullable_monetary_parameters_preserve_precision() -> Result<()> {
    let db = sqlite().await?;
    nullable_float_bind_contract(&db).await
}
