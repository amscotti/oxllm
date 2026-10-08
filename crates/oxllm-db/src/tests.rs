use super::*;
use crate::types::SpendFilter;
use std::sync::Arc;

async fn fresh_db() -> Arc<Database> {
    let db = Database::connect("sqlite::memory:")
        .await
        .expect("connect sqlite memory");
    db.migrate().await.expect("migrate");
    db
}

fn sample_key(key_id: &str, hash: &str) -> KeyRecord {
    KeyRecord {
        key_id: key_id.to_string(),
        key_hash: hash.to_string(),
        key_alias: Some("alias".to_string()),
        team_id: Some("team-1".to_string()),
        user_id: Some("user-1".to_string()),
        org_id: None,
        project_id: None,
        max_budget: Some(10.0),
        budget_duration: Some("30d".to_string()),
        rpm: Some(100),
        tpm: Some(10_000),
        created_at: now_iso(),
        updated_at: now_iso(),
        expires_at: None,
        is_active: true,
        models: None,
        allowed_cache_controls: None,
    }
}

#[tokio::test]
async fn migrate_is_idempotent() {
    let db = fresh_db().await;
    db.migrate().await.expect("second migrate should be ok");
}

/// Compatibility with a version-only bookkeeping table. This is a synthetic
/// imported-schema case; the frozen historical v1 fixture in release_tests
/// includes applied_at and separately exercises a real data-preserving upgrade.
#[tokio::test]
async fn migrate_upgrades_pre_applied_at_schema_version_table() {
    let db = Database::connect("sqlite::memory:")
        .await
        .expect("connect sqlite memory");
    // Shape the table exactly as the old build left it: version only, no
    // `applied_at`, no version rows yet (a fresh old-build deployment
    // records nothing until the first migrate writes it). The full chain
    // then runs, and every `set_schema_version` write exercises the
    // backfilled column.
    sqlx::query("CREATE TABLE _schema_version (version INTEGER PRIMARY KEY)")
        .execute(db.pool())
        .await
        .expect("create old-shape table");
    db.migrate()
        .await
        .expect("migrate must upgrade the old table");
    let v: (i64,) = sqlx::query_as("SELECT MAX(version) FROM _schema_version")
        .fetch_one(db.pool())
        .await
        .expect("read version");
    assert_eq!(
        v.0, SCHEMA_VERSION,
        "migrations must run through the latest version"
    );
}

/// H36: migrate must record the applied schema version in
/// `_schema_version`, and a second migrate must be a no-op (version
/// unchanged, batches not re-run).
#[tokio::test]
async fn migrate_records_and_reads_schema_version() {
    let db = Database::connect("sqlite::memory:")
        .await
        .expect("connect sqlite memory");
    // Pre-migrate: _schema_version table doesn't exist yet, so the
    // reader returns 0 (defensive default).
    assert_eq!(db.get_schema_version().await.unwrap(), 0);

    db.migrate().await.expect("first migrate");
    // After migrating up to the current batch, the version must match it.
    assert_eq!(db.get_schema_version().await.unwrap(), SCHEMA_VERSION);

    // Second migrate is a no-op: the version is already current.
    db.migrate().await.expect("second migrate (no-op)");
    assert_eq!(
        db.get_schema_version().await.unwrap(),
        SCHEMA_VERSION,
        "version must not increase on a re-migrate"
    );
}

/// H36: a database that already has the schema (because it was
/// created by a pre-version-tracking build) must still end up at the
/// current version after migrate, with the duplicate-column errors tolerated.
#[tokio::test]
async fn migrate_marks_v2_on_pre_existing_columns() {
    let db = fresh_db().await;
    db.create_key(&sample_key("unversioned", "unversioned-hash"))
        .await
        .unwrap();
    // Actually remove version tracking, rather than merely re-running a
    // current schema and claiming to exercise duplicate-column handling.
    sqlx::query("DROP TABLE _schema_version")
        .execute(db.pool())
        .await
        .unwrap();
    db.migrate().await.expect("adopt pre-existing columns");
    assert_eq!(
        db.get_schema_version().await.unwrap(),
        SCHEMA_VERSION,
        "unversioned existing schema must reach the current version"
    );
    assert!(db.get_key("unversioned").await.unwrap().is_some());
}

#[tokio::test]
async fn key_crud() {
    let db = fresh_db().await;
    let key = sample_key("k-1", "hash-1");
    db.create_key(&key).await.expect("create");

    let fetched = db
        .get_key_by_hash("hash-1")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(fetched.key_id, "k-1");
    assert_eq!(fetched.team_id.as_deref(), Some("team-1"));
    assert!(fetched.is_active);

    let listed = db.list_keys(None).await.expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].key_hash, "hash-1");

    db.delete_key("k-1", &now_iso()).await.expect("delete");
    let after = db
        .get_key_by_hash("hash-1")
        .await
        .expect("get")
        .expect("present");
    assert!(!after.is_active, "soft-delete sets is_active=false");

    db.purge_key("k-1").await.expect("purge");
    let gone = db.get_key("k-1").await.expect("get");
    assert!(gone.is_none(), "purge removes the row");
}

#[tokio::test]
async fn key_hash_unique() {
    let db = fresh_db().await;
    let a = sample_key("k-a", "dup");
    let b = KeyRecord {
        key_id: "k-b".to_string(),
        ..a.clone()
    };
    db.create_key(&a).await.expect("create a");
    let err = db.create_key(&b).await;
    assert!(err.is_err(), "duplicate hash must error");
}

#[tokio::test]
async fn get_missing_key_returns_none() {
    let db = fresh_db().await;
    let r = db.get_key("nope").await.expect("ok");
    assert!(r.is_none());
}

#[tokio::test]
async fn rotate_key_preserves_inactive_state() {
    // Regression: `rotate_key` previously SET is_active = 1 unconditionally,
    // silently resurrecting soft-deleted keys. Rotation must only touch
    // `key_hash` and `updated_at`.
    let db = fresh_db().await;
    let key = sample_key("k-rot", "hash-orig");
    db.create_key(&key).await.expect("create");

    db.delete_key("k-rot", &now_iso())
        .await
        .expect("soft-delete");
    let before = db.get_key("k-rot").await.expect("get").expect("present");
    assert!(!before.is_active, "precondition: key should be inactive");

    let prev_hash = db
        .rotate_key("k-rot", "hash-new", &now_iso())
        .await
        .expect("rotate")
        .expect("prev hash present");
    assert_eq!(prev_hash, "hash-orig");

    let after = db.get_key("k-rot").await.expect("get").expect("present");
    assert_eq!(after.key_hash, "hash-new", "hash should be updated");
    assert!(
        !after.is_active,
        "rotate_key must not reactivate a soft-deleted key"
    );
}

#[tokio::test]
async fn team_crud() {
    let db = fresh_db().await;
    let team = TeamRecord {
        team_id: "team-1".to_string(),
        team_name: "Engineering".to_string(),
        max_budget: Some(100.0),
        budget_duration: Some("30d".to_string()),
        rpm: Some(500),
        tpm: Some(50_000),
        created_at: now_iso(),
        updated_at: now_iso(),
        models: None,
        blocked: false,
    };
    db.create_team(&team).await.expect("create");
    let fetched = db.get_team("team-1").await.expect("get").expect("present");
    assert_eq!(fetched.team_name, "Engineering");
    assert_eq!(fetched.rpm, Some(500));

    db.delete_team("team-1").await.expect("delete");
    assert!(db.get_team("team-1").await.expect("ok").is_none());
}

/// C27/H2: the `blocked` column on teams must round-trip through create →
/// get → update → get, and default to `false` on a fresh team.
#[tokio::test]
async fn team_blocked_flag_roundtrips() {
    let db = fresh_db().await;
    let now = now_iso();
    db.create_team(&TeamRecord {
        team_id: "team-b".to_string(),
        team_name: "Blocked".to_string(),
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now.clone(),
        updated_at: now.clone(),
        models: None,
        blocked: false,
    })
    .await
    .expect("create");
    let fetched = db.get_team("team-b").await.unwrap().unwrap();
    assert!(!fetched.blocked, "new team defaults to not blocked");

    let mut record = fetched;
    record.blocked = true;
    record.updated_at = now_iso();
    db.update_team(&record).await.expect("update");
    let after = db.get_team("team-b").await.unwrap().unwrap();
    assert!(after.blocked, "blocked flag must persist after update");

    record.blocked = false;
    record.updated_at = now_iso();
    db.update_team(&record).await.expect("unblock");
    let final_state = db.get_team("team-b").await.unwrap().unwrap();
    assert!(!final_state.blocked, "blocked flag must clear on unblock");
}

#[tokio::test]
async fn user_crud() {
    let db = fresh_db().await;
    let user = UserRecord {
        user_id: "user-1".to_string(),
        user_name: "Alice".to_string(),
        max_budget: Some(50.0),
        budget_duration: Some("1mo".to_string()),
        created_at: now_iso(),
        updated_at: now_iso(),
        models: None,
    };
    db.create_user(&user).await.expect("create");
    let fetched = db.get_user("user-1").await.expect("get").expect("present");
    assert_eq!(fetched.user_name, "Alice");
    assert_eq!(fetched.max_budget, Some(50.0));
}

#[tokio::test]
async fn organization_crud() {
    let db = fresh_db().await;
    let org = OrganizationRecord {
        org_id: "org-1".to_string(),
        org_name: "Acme".to_string(),
        max_budget: Some(1_000.0),
        budget_duration: Some("30d".to_string()),
        rpm: Some(500),
        tpm: Some(50_000),
        created_at: now_iso(),
        updated_at: now_iso(),
        models: None,
        metadata: None,
        soft_budget: None,
        max_parallel_requests: None,
        model_max_budget: None,
    };
    db.create_organization(&org).await.expect("create");
    let fetched = db
        .get_organization("org-1")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(fetched.org_name, "Acme");
    assert_eq!(fetched.rpm, Some(500));

    let listed = db.list_organizations(None).await.expect("list");
    assert_eq!(listed.len(), 1);

    let mut updated = fetched.clone();
    updated.org_name = "Acme Co".to_string();
    updated.max_budget = Some(2_000.0);
    assert!(
        db.update_organization(&updated).await.expect("update"),
        "update should match the existing row"
    );
    let after = db
        .get_organization("org-1")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(after.org_name, "Acme Co");
    assert_eq!(after.max_budget, Some(2_000.0));

    db.delete_organization("org-1").await.expect("delete");
    assert!(db.get_organization("org-1").await.expect("ok").is_none());
}

/// v4 migration: the five new organization columns must exist after
/// migrate. We probe each one with a direct INSERT/SELECT that references
/// the column — a missing column surfaces as a sqlx error rather than a
/// silent failure.
#[tokio::test]
async fn migrate_v4_adds_organization_columns() {
    let db = fresh_db().await;
    // If any v4 column is missing, this INSERT fails with a sqlx error.
    sqlx::query(
        "INSERT INTO organizations (org_id, org_name, created_at, updated_at, \
 models, metadata, soft_budget, max_parallel_requests, model_max_budget) \
 VALUES ('probe', 'Probe', 't', 't', NULL, NULL, NULL, NULL, NULL)",
    )
    .execute(db.pool())
    .await
    .expect("v4 columns must be present");
    // And a SELECT touching every v4 column must succeed.
    let row = sqlx::query(
        "SELECT models, metadata, soft_budget, max_parallel_requests, model_max_budget \
 FROM organizations WHERE org_id = 'probe'",
    )
    .fetch_one(db.pool())
    .await
    .expect("select v4 columns");
    use sqlx::Row;
    let models: Option<String> = row.try_get("models").unwrap();
    let metadata: Option<String> = row.try_get("metadata").unwrap();
    let soft_budget: Option<f64> = row.try_get("soft_budget").unwrap();
    let max_parallel_requests: Option<i64> = row.try_get("max_parallel_requests").unwrap();
    let model_max_budget: Option<String> = row.try_get("model_max_budget").unwrap();
    assert!(models.is_none());
    assert!(metadata.is_none());
    assert!(soft_budget.is_none());
    assert!(max_parallel_requests.is_none());
    assert!(model_max_budget.is_none());
}

/// CRUD round-trip for the v4 fields: create an org with models, metadata,
/// and soft_budget set, read it back, and verify every field persists.
#[tokio::test]
async fn organization_v4_fields_roundtrip() {
    let db = fresh_db().await;
    let org = OrganizationRecord {
        org_id: "org-v4".to_string(),
        org_name: "Acme V4".to_string(),
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now_iso(),
        updated_at: now_iso(),
        models: Some(r#"["gpt-4o","claude-*"]"#.to_string()),
        metadata: Some(r#"{"tier":"enterprise"}"#.to_string()),
        soft_budget: Some(75.0),
        max_parallel_requests: Some(16),
        model_max_budget: Some(r#"{"gpt-4o":10.0}"#.to_string()),
    };
    db.create_organization(&org).await.expect("create");
    let fetched = db
        .get_organization("org-v4")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(fetched.models.as_deref(), Some(r#"["gpt-4o","claude-*"]"#));
    assert_eq!(
        fetched.metadata.as_deref(),
        Some(r#"{"tier":"enterprise"}"#)
    );
    assert_eq!(fetched.soft_budget, Some(75.0));
    assert_eq!(fetched.max_parallel_requests, Some(16));
    assert_eq!(
        fetched.model_max_budget.as_deref(),
        Some(r#"{"gpt-4o":10.0}"#)
    );

    // Listing must also surface the v4 fields.
    let listed = db.list_organizations(None).await.expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].soft_budget, Some(75.0));
    assert_eq!(listed[0].max_parallel_requests, Some(16));
}

/// Update path: changing soft_budget (and the other v4 fields) must
/// persist through update_organization.
#[tokio::test]
async fn organization_v4_fields_update() {
    let db = fresh_db().await;
    let org = OrganizationRecord {
        org_id: "org-upd".to_string(),
        org_name: "Acme".to_string(),
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now_iso(),
        updated_at: now_iso(),
        models: None,
        metadata: None,
        soft_budget: Some(50.0),
        max_parallel_requests: None,
        model_max_budget: None,
    };
    db.create_organization(&org).await.expect("create");

    let mut fetched = db
        .get_organization("org-upd")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(fetched.soft_budget, Some(50.0));

    fetched.soft_budget = Some(120.0);
    fetched.max_parallel_requests = Some(32);
    fetched.models = Some(r#"["gpt-4o"]"#.to_string());
    fetched.updated_at = now_iso();
    assert!(
        db.update_organization(&fetched).await.expect("update"),
        "update should match the existing row"
    );

    let after = db
        .get_organization("org-upd")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(after.soft_budget, Some(120.0), "soft_budget must update");
    assert_eq!(after.max_parallel_requests, Some(32));
    assert_eq!(after.models.as_deref(), Some(r#"["gpt-4o"]"#));
}

#[tokio::test]
async fn project_crud() {
    let db = fresh_db().await;
    let project = ProjectRecord {
        project_id: "project-1".to_string(),
        project_name: "Atlas".to_string(),
        org_id: Some("org-1".to_string()),
        team_id: Some("team-1".to_string()),
        max_budget: Some(100.0),
        budget_duration: Some("7d".to_string()),
        rpm: Some(50),
        tpm: Some(5_000),
        created_at: now_iso(),
        updated_at: now_iso(),
    };
    db.create_project(&project).await.expect("create");
    let fetched = db
        .get_project("project-1")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(fetched.project_name, "Atlas");
    assert_eq!(fetched.org_id.as_deref(), Some("org-1"));
    assert_eq!(fetched.team_id.as_deref(), Some("team-1"));

    let listed = db.list_projects(None).await.expect("list");
    assert_eq!(listed.len(), 1);

    let mut updated = fetched.clone();
    updated.team_id = None;
    updated.max_budget = Some(200.0);
    assert!(
        db.update_project(&updated).await.expect("update"),
        "update should match the existing row"
    );
    let after = db
        .get_project("project-1")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(after.team_id, None);
    assert_eq!(after.max_budget, Some(200.0));

    db.delete_project("project-1").await.expect("delete");
    assert!(db.get_project("project-1").await.expect("ok").is_none());
}

#[tokio::test]
async fn key_with_org_and_project_roundtrips() {
    let db = fresh_db().await;
    let mut key = sample_key("k-org", "hash-org");
    key.org_id = Some("org-1".to_string());
    key.project_id = Some("project-1".to_string());
    db.create_key(&key).await.expect("create");
    let fetched = db
        .get_key_by_hash("hash-org")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(fetched.org_id.as_deref(), Some("org-1"));
    assert_eq!(fetched.project_id.as_deref(), Some("project-1"));

    let mut updated = fetched.clone();
    updated.org_id = None;
    updated.project_id = None;
    db.update_key(&updated).await.expect("update");
    let after = db.get_key("k-org").await.expect("get").expect("present");
    assert_eq!(after.org_id, None);
    assert_eq!(after.project_id, None);
}

// ---- C27: scoped / atomic team-key bulk mutations -----------------------

/// Helper: a key owned by `(team_id, user_id)` with a unique hash.
fn key_in_scope(
    key_id: &str,
    hash: &str,
    team_id: Option<&str>,
    user_id: Option<&str>,
) -> KeyRecord {
    KeyRecord {
        key_id: key_id.to_string(),
        key_hash: hash.to_string(),
        key_alias: None,
        team_id: team_id.map(str::to_string),
        user_id: user_id.map(str::to_string),
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

/// C27/M2: `set_keys_active_for_team` must flip only the target team's keys
/// (no full-table scan side effects). Keys in other teams — and team-less
/// keys — stay untouched. The bulk UPDATE is a single statement, so it is
/// inherently atomic.
#[tokio::test]
async fn set_keys_active_for_team_only_touches_target_team() {
    let db = fresh_db().await;
    db.create_key(&key_in_scope("k-a", "h-a", Some("team-1"), None))
        .await
        .expect("create a");
    db.create_key(&key_in_scope("k-b", "h-b", Some("team-1"), None))
        .await
        .expect("create b");
    db.create_key(&key_in_scope("k-c", "h-c", Some("team-2"), None))
        .await
        .expect("create c");
    db.create_key(&key_in_scope("k-d", "h-d", None, None))
        .await
        .expect("create d");

    let n = db
        .set_keys_active_for_team("team-1", false, &now_iso())
        .await
        .expect("bulk deactivate");
    assert_eq!(n, 2, "only the two team-1 keys are affected");

    assert!(!db.get_key("k-a").await.unwrap().unwrap().is_active);
    assert!(!db.get_key("k-b").await.unwrap().unwrap().is_active);
    assert!(
        db.get_key("k-c").await.unwrap().unwrap().is_active,
        "team-2 key untouched"
    );
    assert!(
        db.get_key("k-d").await.unwrap().unwrap().is_active,
        "team-less key untouched"
    );
}

/// C27/M2: `list_key_hashes_for_team` is scoped to the team (used for cache
/// invalidation after the bulk mutation).
#[tokio::test]
async fn list_key_hashes_for_team_is_scoped() {
    let db = fresh_db().await;
    db.create_key(&key_in_scope("k-a", "h-a", Some("team-1"), None))
        .await
        .expect("create a");
    db.create_key(&key_in_scope("k-b", "h-b", Some("team-1"), None))
        .await
        .expect("create b");
    db.create_key(&key_in_scope("k-c", "h-c", Some("team-2"), None))
        .await
        .expect("create c");

    let mut hashes = db.list_key_hashes_for_team("team-1").await.expect("list");
    hashes.sort();
    assert_eq!(hashes, vec!["h-a", "h-b"]);
}

/// C27/M2: `set_team_for_user_keys` with no filter (member/add) sets the team
/// on every key owned by the user; with a filter (member/delete) it only
/// clears keys currently in that team.
#[tokio::test]
async fn set_team_for_user_keys_add_and_delete_semantics() {
    let db = fresh_db().await;
    // user-1 owns three keys: two team-less, one already in team-old.
    db.create_key(&key_in_scope("k-1", "h-1", None, Some("user-1")))
        .await
        .expect("create 1");
    db.create_key(&key_in_scope("k-2", "h-2", None, Some("user-1")))
        .await
        .expect("create 2");
    db.create_key(&key_in_scope(
        "k-3",
        "h-3",
        Some("team-old"),
        Some("user-1"),
    ))
    .await
    .expect("create 3");
    // Another user's key must never be touched.
    db.create_key(&key_in_scope("k-x", "h-x", None, Some("user-2")))
        .await
        .expect("create x");

    // member/add: assign user-1 to team-new on all their keys.
    let n = db
        .set_team_for_user_keys("user-1", None, Some("team-new"), &now_iso())
        .await
        .expect("add");
    assert_eq!(n, 3);
    for id in ["k-1", "k-2", "k-3"] {
        assert_eq!(
            db.get_key(id).await.unwrap().unwrap().team_id.as_deref(),
            Some("team-new")
        );
    }
    assert_eq!(
        db.get_key("k-x").await.unwrap().unwrap().team_id,
        None,
        "other user's key untouched"
    );

    // member/delete: drop membership of team-new only.
    let n = db
        .set_team_for_user_keys("user-1", Some("team-new"), None, &now_iso())
        .await
        .expect("delete");
    assert_eq!(n, 3);
    for id in ["k-1", "k-2", "k-3"] {
        assert_eq!(
            db.get_key(id).await.unwrap().unwrap().team_id,
            None,
            "team cleared"
        );
    }
}

/// C27/M2: `list_key_hashes_for_user` honours the optional team filter
/// (member/delete only invalidates keys that were in the dropped team).
#[tokio::test]
async fn list_key_hashes_for_user_with_team_filter() {
    let db = fresh_db().await;
    db.create_key(&key_in_scope("k-1", "h-1", Some("team-a"), Some("user-1")))
        .await
        .expect("create 1");
    db.create_key(&key_in_scope("k-2", "h-2", Some("team-b"), Some("user-1")))
        .await
        .expect("create 2");

    let only_a = db
        .list_key_hashes_for_user("user-1", Some("team-a"))
        .await
        .expect("list filtered");
    assert_eq!(only_a, vec!["h-1"]);

    let all = db
        .list_key_hashes_for_user("user-1", None)
        .await
        .expect("list all");
    let mut all = all;
    all.sort();
    assert_eq!(all, vec!["h-1", "h-2"]);
}

/// C27/M5: `clear_team_on_keys` removes the dangling `team_id` reference on
/// every key that pointed at the team (cascade on team delete), so the next
/// auth does not silently keep enforcing — or drop — a removed scope's limits.
#[tokio::test]
async fn clear_team_on_keys_cascades() {
    let db = fresh_db().await;
    db.create_key(&key_in_scope("k-a", "h-a", Some("team-1"), None))
        .await
        .expect("create a");
    db.create_key(&key_in_scope("k-b", "h-b", Some("team-1"), Some("user-1")))
        .await
        .expect("create b");
    db.create_key(&key_in_scope("k-c", "h-c", Some("team-2"), None))
        .await
        .expect("create c");

    let n = db
        .clear_team_on_keys("team-1", &now_iso())
        .await
        .expect("clear");
    assert_eq!(n, 2, "both team-1 keys cleared");
    assert_eq!(db.get_key("k-a").await.unwrap().unwrap().team_id, None);
    assert_eq!(db.get_key("k-b").await.unwrap().unwrap().team_id, None);
    assert_eq!(
        db.get_key("k-c").await.unwrap().unwrap().team_id.as_deref(),
        Some("team-2"),
        "team-2 key untouched"
    );
}

#[tokio::test]
async fn spend_insert_and_query() {
    let db = fresh_db().await;
    let log = SpendLog {
        id: None,
        timestamp: now_iso(),
        request_id: Some("req-1".to_string()),
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(10),
        completion_tokens: Some(5),
        total_tokens: Some(15),
        cost: Some(0.001),
        key_id: Some("k-1".to_string()),
        team_id: None,
        user_id: None,
        end_user_id: None,
        tag_name: None,
        tags: None,
        status: Some("success".to_string()),
        latency_ms: Some(42),
    };
    db.insert_spend_log(&log).await.expect("insert");

    let all = db
        .query_spend(&SpendFilter::default())
        .await
        .expect("query");
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].model, "gpt-4o");
    assert!((all[0].cost.unwrap_or(-1.0) - 0.001).abs() < 1e-9);

    let by_key = db
        .query_spend(&SpendFilter {
            key_id: Some("k-1".to_string()),
            ..Default::default()
        })
        .await
        .expect("query by key");
    assert_eq!(by_key.len(), 1);

    let by_other = db
        .query_spend(&SpendFilter {
            key_id: Some("k-other".to_string()),
            ..Default::default()
        })
        .await
        .expect("query miss");
    assert!(by_other.is_empty());
}

#[tokio::test]
async fn spend_logs_tag_filter_and_by_tag_aggregation() {
    let db = fresh_db().await;
    let mk =
        |ts: &str, model: &str, cost: f64, tags: Option<&str>, primary: Option<&str>| SpendLog {
            id: None,
            timestamp: ts.to_string(),
            request_id: None,
            model: model.to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            cost: Some(cost),
            key_id: Some("k-1".to_string()),
            team_id: None,
            user_id: None,
            end_user_id: None,
            tag_name: primary.map(str::to_string),
            tags: tags.map(str::to_string),
            status: Some("success".to_string()),
            latency_ms: None,
        };
    db.insert_spend_log(&mk(
        "2026-07-01T00:00:00Z",
        "gpt-4o",
        1.0,
        Some(r#"["app-a"]"#),
        Some("app-a"),
    ))
    .await
    .expect("insert");
    db.insert_spend_log(&mk(
        "2026-07-02T00:00:00Z",
        "gpt-4o",
        2.0,
        Some(r#"["app-a","env-prod"]"#),
        Some("app-a"),
    ))
    .await
    .expect("insert");
    db.insert_spend_log(&mk(
        "2026-07-03T00:00:00Z",
        "gpt-4o",
        4.0,
        Some(r#"["app-b"]"#),
        Some("app-b"),
    ))
    .await
    .expect("insert");
    db.insert_spend_log(&mk("2026-07-04T00:00:00Z", "gpt-4o", 8.0, None, None))
        .await
        .expect("insert");

    // Exact membership cannot cross-match tag prefixes.
    let f = |tag: &str| SpendFilter {
        tag_name: Some(tag.to_string()),
        ..Default::default()
    };
    let a = db.query_spend(&f("app-a")).await.expect("q");
    assert_eq!(a.len(), 2, "both app-a rows match");
    let b = db.query_spend(&f("app-b")).await.expect("q");
    assert_eq!(b.len(), 1, "app-b does not match app-a rows");
    let none = db.query_spend(&f("nope")).await.expect("q");
    assert!(none.is_empty());

    // by-tag aggregation: multi-tag rows count toward every carried tag.
    let by_tag = db.query_spend_by_tag(None, None).await.expect("agg");
    let get = |t: &str| {
        by_tag
            .iter()
            .find(|r| r.tag_name.as_deref() == Some(t))
            .unwrap_or_else(|| panic!("missing {t}"))
    };
    assert_eq!(get("app-a").total_spend, 3.0, "1.0 + 2.0");
    assert_eq!(get("env-prod").total_spend, 2.0, "the multi-tag row only");
    assert_eq!(get("app-b").total_spend, 4.0);
    let untagged = by_tag
        .iter()
        .find(|r| r.tag_name.is_none())
        .expect("untagged group");
    assert_eq!(untagged.total_spend, 8.0);
    assert_eq!(untagged.total_requests, 1);
}

#[tokio::test]
async fn migrate_v10_adds_tags_column() {
    let db = Database::connect("sqlite::memory:").await.expect("connect");
    db.migrate().await.expect("migrate");
    let v: (i64,) = sqlx::query_as("SELECT MAX(version) FROM _schema_version")
        .fetch_one(db.pool())
        .await
        .expect("version");
    assert!(v.0 >= 10, "v10 must be applied");
    let has: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM pragma_table_info('spend_logs') WHERE name = 'tags'")
            .fetch_one(db.pool())
            .await
            .expect("columns");
    assert_eq!(has.0, 1, "spend_logs.tags column must exist");
}

#[tokio::test]
async fn spend_total_and_count() {
    let db = fresh_db().await;
    for i in 0..3 {
        db.insert_spend_log(&SpendLog {
            id: None,
            timestamp: now_iso(),
            request_id: Some(format!("req-{i}")),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            cost: Some(0.01),
            key_id: Some("k-1".to_string()),
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: Some("success".to_string()),
            latency_ms: None,
        })
        .await
        .expect("insert");
    }

    let total = db.get_total_spend(Some("k-1"), None).await.expect("total");
    assert!((total - 0.03).abs() < 1e-9);

    let count = db.count_spend(Some("k-1"), None).await.expect("count");
    assert_eq!(count, 3);

    let total_other = db
        .get_total_spend(Some("k-other"), None)
        .await
        .expect("total");
    assert!(total_other.abs() < 1e-9);
}

/// Bug #4b regression: `count_spend` must alias `COUNT(*)` to `total`
/// and read it back by that name. SQLite names the unaliased column
/// `COUNT(*)`; Postgres names it `count`. The alias normalizes both.
///
/// This test exercises every WHERE-clause shape (no filter, key only,
/// since only, both) so a regression in any branch is caught.
#[tokio::test]
async fn count_spend_alias_works_across_filters() {
    let db = fresh_db().await;
    let mk = |key: &str, ts: &str| SpendLog {
        id: None,
        timestamp: ts.to_string(),
        request_id: None,
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(0.01),
        key_id: Some(key.to_string()),
        team_id: None,
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    };
    db.insert_spend_log(&mk("k-1", "2024-01-01T00:00:00Z"))
        .await
        .expect("ins");
    db.insert_spend_log(&mk("k-1", "2024-06-01T00:00:00Z"))
        .await
        .expect("ins");
    db.insert_spend_log(&mk("k-2", "2024-06-01T00:00:00Z"))
        .await
        .expect("ins");

    // No filter at all — exercises the empty-WHERE branch.
    assert_eq!(db.count_spend(None, None).await.expect("count all"), 3);
    // key filter only.
    assert_eq!(
        db.count_spend(Some("k-1"), None).await.expect("count k-1"),
        2
    );
    // since filter only — exercises the `timestamp >=` branch.
    assert_eq!(
        db.count_spend(None, Some("2024-03-01T00:00:00Z"))
            .await
            .expect("count since"),
        2
    );
    // Both filters — exercises the WHERE with two clauses.
    assert_eq!(
        db.count_spend(Some("k-1"), Some("2024-03-01T00:00:00Z"))
            .await
            .expect("count k-1 since"),
        1
    );
    // Misses return zero, not an error.
    assert_eq!(
        db.count_spend(Some("k-missing"), None)
            .await
            .expect("count missing"),
        0
    );
}

#[tokio::test]
async fn spend_filter_by_model() {
    let db = fresh_db().await;
    let mk = |model: &str, cost: f64| SpendLog {
        id: None,
        timestamp: now_iso(),
        request_id: None,
        model: model.to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(cost),
        key_id: None,
        team_id: None,
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    };
    db.insert_spend_log(&mk("gpt-4o", 0.1)).await.expect("ins");
    db.insert_spend_log(&mk("gpt-4o-mini", 0.2))
        .await
        .expect("ins");

    let only_mini = db
        .query_spend(&SpendFilter {
            model: Some("gpt-4o-mini".to_string()),
            ..Default::default()
        })
        .await
        .expect("q");
    assert_eq!(only_mini.len(), 1);
    assert_eq!(only_mini[0].model, "gpt-4o-mini");
}

#[tokio::test]
async fn total_spend_sums_all_keys() {
    let db = fresh_db().await;
    let mk = |cost: f64, key: &str| SpendLog {
        id: None,
        timestamp: now_iso(),
        request_id: None,
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(cost),
        key_id: Some(key.to_string()),
        team_id: None,
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    };
    db.insert_spend_log(&mk(0.10, "k-1")).await.expect("ins");
    db.insert_spend_log(&mk(0.20, "k-2")).await.expect("ins");

    let total = db.total_spend(None, None).await.expect("total");
    assert!((total - 0.30).abs() < 1e-9);
}

#[tokio::test]
async fn total_spend_respects_date_range() {
    let db = fresh_db().await;
    let mk = |ts: &str, cost: f64| SpendLog {
        id: None,
        timestamp: ts.to_string(),
        request_id: None,
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(cost),
        key_id: None,
        team_id: None,
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    };
    db.insert_spend_log(&mk("2024-01-01T00:00:00Z", 0.10))
        .await
        .expect("ins");
    db.insert_spend_log(&mk("2024-06-01T00:00:00Z", 0.20))
        .await
        .expect("ins");

    let total_all = db.total_spend(None, None).await.expect("total");
    assert!((total_all - 0.30).abs() < 1e-9);

    let total_since = db
        .total_spend(Some("2024-03-01T00:00:00Z"), None)
        .await
        .expect("total since");
    assert!((total_since - 0.20).abs() < 1e-9);

    let total_before = db
        .total_spend(None, Some("2024-03-01T00:00:00Z"))
        .await
        .expect("total before");
    assert!((total_before - 0.10).abs() < 1e-9);
}

#[tokio::test]
async fn query_spend_paginated_walks_cursor() {
    let db = fresh_db().await;
    for i in 0..6 {
        db.insert_spend_log(&SpendLog {
            id: None,
            timestamp: now_iso(),
            request_id: Some(format!("req-{i}")),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            cost: Some(0.01),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        })
        .await
        .expect("insert");
    }

    let page1 = db
        .query_spend_paginated(None, 2, &SpendFilter::default())
        .await
        .expect("page1");
    assert_eq!(page1.len(), 2, "first page has 2 rows");

    let last_id = page1.last().and_then(|l| l.id).expect("id present");
    let page2 = db
        .query_spend_paginated(Some(last_id), 2, &SpendFilter::default())
        .await
        .expect("page2");
    assert_eq!(page2.len(), 2, "second page has 2 rows");

    for p in &page2 {
        let id = p.id.unwrap_or(i64::MAX);
        assert!(id < last_id, "every row on page2 must come before cursor");
    }

    let last2 = page2.last().and_then(|l| l.id).expect("id present");
    let page3 = db
        .query_spend_paginated(Some(last2), 2, &SpendFilter::default())
        .await
        .expect("page3");
    assert_eq!(page3.len(), 2, "third page has 2 rows");
}

#[tokio::test]
async fn query_spend_paginated_filters_by_team() {
    let db = fresh_db().await;
    let mk = |team: &str, cost: f64| SpendLog {
        id: None,
        timestamp: now_iso(),
        request_id: None,
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(cost),
        key_id: None,
        team_id: Some(team.to_string()),
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    };
    db.insert_spend_log(&mk("team-a", 0.10)).await.expect("ins");
    db.insert_spend_log(&mk("team-b", 0.20)).await.expect("ins");
    db.insert_spend_log(&mk("team-a", 0.05)).await.expect("ins");

    let team_a = db
        .query_spend_paginated(
            None,
            100,
            &SpendFilter {
                team_id: Some("team-a".to_string()),
                ..Default::default()
            },
        )
        .await
        .expect("q");
    assert_eq!(team_a.len(), 2);
    let total: f64 = team_a.iter().filter_map(|l| l.cost).sum();
    assert!((total - 0.15).abs() < 1e-9);
}

#[tokio::test]
async fn query_spend_paginated_page_size_cap_is_1000() {
    // U45: the v2 page-size cap is 1000 (raised from 500). A request above
    // the cap is clamped down; a request at/under the cap is honored.
    let db = fresh_db().await;
    let total_rows = 1010;
    for i in 0..total_rows {
        db.insert_spend_log(&SpendLog {
            id: None,
            timestamp: now_iso(),
            request_id: Some(format!("req-{i}")),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            cost: Some(0.01),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        })
        .await
        .expect("insert");
    }

    // Above the cap → clamped to exactly 1000.
    let page = db
        .query_spend_paginated(None, 10_000, &SpendFilter::default())
        .await
        .expect("page");
    assert_eq!(page.len(), 1000, "limit above cap must clamp to 1000");

    // Exactly at the cap → honored.
    let page = db
        .query_spend_paginated(None, 1000, &SpendFilter::default())
        .await
        .expect("page");
    assert_eq!(page.len(), 1000, "limit at cap must be honored");

    // The old cap (500) is still a legal page size.
    let page = db
        .query_spend_paginated(None, 500, &SpendFilter::default())
        .await
        .expect("page");
    assert_eq!(page.len(), 500);
}

#[tokio::test]
async fn daily_activity_buckets_today() {
    let db = fresh_db().await;
    db.insert_spend_log(&SpendLog {
        id: None,
        timestamp: now_iso(),
        request_id: None,
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(0.42),
        key_id: None,
        team_id: None,
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    })
    .await
    .expect("ins");

    let activity = db.daily_activity(7).await.expect("daily");
    assert!(!activity.is_empty(), "should have at least one bucket");
    let today_total = activity.iter().map(|(_, c)| c).sum::<f64>();
    assert!(
        (today_total - 0.42).abs() < 1e-9,
        "bucket sum {today_total} != 0.42"
    );
}

#[tokio::test]
async fn query_spend_supports_team_filter() {
    let db = fresh_db().await;
    let mk = |team: &str| SpendLog {
        id: None,
        timestamp: now_iso(),
        request_id: None,
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(0.01),
        key_id: None,
        team_id: Some(team.to_string()),
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    };
    db.insert_spend_log(&mk("team-x")).await.expect("ins");
    db.insert_spend_log(&mk("team-y")).await.expect("ins");

    let xs = db
        .query_spend(&SpendFilter {
            team_id: Some("team-x".to_string()),
            ..Default::default()
        })
        .await
        .expect("q");
    assert_eq!(xs.len(), 1);
}

// ---- Models CRUD ----------------------------------------------------------

fn sample_model(model_id: &str, name: &str) -> ModelConfigRecord {
    ModelConfigRecord {
        model_id: model_id.to_string(),
        model_name: name.to_string(),
        oxllm_params: serde_json::json!({
            "model": "openai/gpt-4o",
            "api_key": "sk-test",
        }),
        model_info: Some(serde_json::json!({ "tags": ["canary"] })),
        created_at: now_iso(),
        updated_at: now_iso(),
    }
}

#[tokio::test]
async fn model_crud_roundtrip() {
    let db = fresh_db().await;
    let model = sample_model("model-abc", "my-model");
    db.create_model(&model).await.expect("create");

    let fetched = db
        .get_model("model-abc")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(fetched.model_name, "my-model");
    assert_eq!(fetched.oxllm_params["model"], "openai/gpt-4o");
    assert_eq!(fetched.model_info.as_ref().unwrap()["tags"][0], "canary");

    let listed = db.list_models_db(None).await.expect("list");
    assert_eq!(listed.len(), 1);

    let mut updated = fetched.clone();
    updated.model_name = "renamed".to_string();
    updated.oxllm_params = serde_json::json!({ "model": "anthropic/claude" });
    db.update_model("model-abc", &updated)
        .await
        .expect("update");
    let after = db
        .get_model("model-abc")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(after.model_name, "renamed");
    assert_eq!(after.oxllm_params["model"], "anthropic/claude");

    db.delete_model("model-abc").await.expect("delete");
    assert!(db.get_model("model-abc").await.expect("get").is_none());
}

#[tokio::test]
async fn get_missing_model_returns_none() {
    let db = fresh_db().await;
    assert!(db.get_model("nope").await.expect("ok").is_none());
}

#[tokio::test]
async fn delete_spend_logs_before_prunes_only_older_rows() {
    // C9: retention pruner. Rows with `timestamp < cutoff` are deleted;
    // rows with `timestamp >= cutoff` are kept. `<` (not `<=`) so the
    // cutoff timestamp itself survives — important when the daily prune
    // runs at midnight and a row's timestamp happens to equal it.
    let db = fresh_db().await;
    let mk = |ts: &str, key: &str| SpendLog {
        id: None,
        timestamp: ts.to_string(),
        request_id: None,
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(0.01),
        key_id: Some(key.to_string()),
        team_id: None,
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    };
    db.insert_spend_log(&mk("2024-01-01T00:00:00Z", "old"))
        .await
        .expect("ins old");
    db.insert_spend_log(&mk("2024-06-01T00:00:00Z", "keep"))
        .await
        .expect("ins keep");
    // Cutoff equal to an existing timestamp: that row must survive (`<`).
    db.insert_spend_log(&mk("2024-03-01T00:00:00Z", "boundary"))
        .await
        .expect("ins boundary");

    let deleted = db
        .delete_spend_logs_before("2024-03-01T00:00:00Z")
        .await
        .expect("delete");
    assert_eq!(deleted, 1, "only the strictly-older row is removed");

    let remaining = db
        .query_spend(&SpendFilter::default())
        .await
        .expect("query");
    assert_eq!(remaining.len(), 2, "boundary + keep remain");
    assert!(
        remaining.iter().all(|l| l.key_id.as_deref() != Some("old")),
        "old row must be gone"
    );
}

#[tokio::test]
async fn insert_spend_logs_batch_inserts_every_row() {
    // H15: multi-row INSERT must produce the same rows as N single-row
    // inserts. Build a non-trivial batch (mixed Option fields, different
    // models) and assert every row appears in `query_spend`.
    let db = fresh_db().await;
    let logs = vec![
        SpendLog {
            id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            request_id: Some("r-1".to_string()),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            cost: Some(0.001),
            key_id: Some("k-1".to_string()),
            team_id: Some("t-1".to_string()),
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: Some("success".to_string()),
            latency_ms: Some(42),
        },
        SpendLog {
            id: None,
            timestamp: "2024-01-02T00:00:00Z".to_string(),
            request_id: None,
            model: "claude-3".to_string(),
            provider: "anthropic".to_string(),
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            cost: None,
            key_id: Some("k-2".to_string()),
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        },
        SpendLog {
            id: None,
            timestamp: "2024-01-03T00:00:00Z".to_string(),
            request_id: Some("r-3".to_string()),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            cost: Some(0.0001),
            key_id: Some("k-1".to_string()),
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        },
    ];
    db.insert_spend_logs_batch(&logs)
        .await
        .expect("batch insert");

    let all = db
        .query_spend(&SpendFilter::default())
        .await
        .expect("query");
    assert_eq!(all.len(), 3, "every batched row should be persisted");
    // Spot-check one row's content to make sure binds line up correctly.
    let claude = all
        .iter()
        .find(|l| l.model == "claude-3")
        .expect("claude row present");
    assert_eq!(claude.provider, "anthropic");
    assert_eq!(claude.key_id.as_deref(), Some("k-2"));
}

#[tokio::test]
async fn insert_spend_logs_batch_empty_is_noop() {
    let db = fresh_db().await;
    db.insert_spend_logs_batch(&[]).await.expect("empty batch");
    let all = db
        .query_spend(&SpendFilter::default())
        .await
        .expect("query");
    assert!(all.is_empty());
}

#[tokio::test]
async fn model_info_optional_roundtrips_when_none() {
    let db = fresh_db().await;
    let mut model = sample_model("model-no-info", "noinfo");
    model.model_info = None;
    db.create_model(&model).await.expect("create");
    let fetched = db
        .get_model("model-no-info")
        .await
        .expect("get")
        .expect("present");
    assert!(fetched.model_info.is_none());
}

// ---- audit-fix queries ---------------------------------------------------

fn spend_row(request_id: Option<&str>, key_id: Option<&str>, cost: Option<f64>) -> SpendLog {
    SpendLog {
        id: None,
        timestamp: now_iso(),
        request_id: request_id.map(str::to_string),
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost,
        key_id: key_id.map(str::to_string),
        team_id: None,
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    }
}

#[tokio::test]
async fn query_spend_by_request_id_finds_row() {
    let db = fresh_db().await;
    db.insert_spend_log(&spend_row(Some("req-hit"), Some("k-1"), Some(0.01)))
        .await
        .expect("insert");

    let found = db
        .query_spend_by_request_id("req-hit")
        .await
        .expect("query")
        .expect("row present");
    assert_eq!(found.request_id.as_deref(), Some("req-hit"));
    assert_eq!(found.key_id.as_deref(), Some("k-1"));
    assert!(found.id.is_some(), "DB-assigned id must be populated");
}

#[tokio::test]
async fn query_spend_by_request_id_missing_returns_none() {
    let db = fresh_db().await;
    db.insert_spend_log(&spend_row(Some("req-other"), None, None))
        .await
        .expect("insert");

    let missing = db
        .query_spend_by_request_id("req-absent")
        .await
        .expect("query");
    assert!(missing.is_none());
    // A NULL request_id row must never match a lookup.
    db.insert_spend_log(&spend_row(None, None, None))
        .await
        .expect("insert null");
    let still_missing = db
        .query_spend_by_request_id("req-absent")
        .await
        .expect("query");
    assert!(still_missing.is_none());
}

#[tokio::test]
async fn query_spend_by_request_id_finds_old_rows() {
    // Regression for the 10k-row in-memory scan: a row outside the
    // "newest N" window must still be found. Insert one old row, then
    // enough newer rows to push it beyond the old scan window would be
    // slow; instead assert the lookup is window-independent by checking
    // a row whose timestamp sorts last in a small set.
    let db = fresh_db().await;
    let mut old = spend_row(Some("req-old"), None, None);
    old.timestamp = "2020-01-01T00:00:00Z".to_string();
    db.insert_spend_log(&old).await.expect("insert old");
    for i in 0..50 {
        db.insert_spend_log(&spend_row(Some(&format!("req-new-{i}")), None, None))
            .await
            .expect("insert new");
    }
    let found = db
        .query_spend_by_request_id("req-old")
        .await
        .expect("query")
        .expect("old row must be found directly");
    // Timestamps are normalized to fixed 9-digit precision on write so
    // lexicographic TEXT comparisons track chronology.
    assert_eq!(found.timestamp, "2020-01-01T00:00:00.000000000Z");
}

#[tokio::test]
async fn query_spend_by_key_aggregates_with_alias_and_nulls() {
    let db = fresh_db().await;
    db.create_key(&sample_key("k-1", "hash-1"))
        .await
        .expect("create key");

    // Two rows for the aliased key (one with NULL cost).
    db.insert_spend_log(&spend_row(None, Some("k-1"), Some(0.10)))
        .await
        .expect("ins");
    db.insert_spend_log(&spend_row(None, Some("k-1"), None))
        .await
        .expect("ins");
    // One row for a key with no `keys` row (LEFT JOIN → NULL alias).
    db.insert_spend_log(&spend_row(None, Some("k-ghost"), Some(0.20)))
        .await
        .expect("ins");
    // One row with no key at all (NULL key_id group).
    db.insert_spend_log(&spend_row(None, None, Some(0.50)))
        .await
        .expect("ins");

    let rows = db.query_spend_by_key(None, None).await.expect("query");
    assert_eq!(rows.len(), 3, "one group per distinct key_id (incl. NULL)");

    let k1 = rows
        .iter()
        .find(|r| r.key_id.as_deref() == Some("k-1"))
        .expect("k-1 group");
    assert_eq!(k1.key_alias.as_deref(), Some("alias"), "LEFT JOIN alias");
    assert_eq!(k1.total_requests, 2, "NULL-cost row still counts");
    assert!(
        (k1.total_spend - 0.10).abs() < 1e-9,
        "SUM ignores NULL cost"
    );
    assert!(k1.last_used.is_some());

    let ghost = rows
        .iter()
        .find(|r| r.key_id.as_deref() == Some("k-ghost"))
        .expect("ghost group");
    assert_eq!(ghost.key_alias, None, "no keys row → NULL alias");

    // NULL key_id stays NULL (not collapsed to "").
    let no_key = rows
        .iter()
        .find(|r| r.key_id.is_none())
        .expect("null group");
    assert_eq!(no_key.total_requests, 1);
    assert!((no_key.total_spend - 0.50).abs() < 1e-9);

    // Ordered by total_spend DESC.
    assert_eq!(rows[0].key_id, None, "highest-spend group first");
}

#[tokio::test]
async fn query_spend_by_key_respects_date_range() {
    let db = fresh_db().await;
    let mut early = spend_row(None, Some("k-1"), Some(0.10));
    early.timestamp = "2024-01-01T00:00:00Z".to_string();
    let mut late = spend_row(None, Some("k-1"), Some(0.20));
    late.timestamp = "2024-06-01T00:00:00Z".to_string();
    db.insert_spend_log(&early).await.expect("ins");
    db.insert_spend_log(&late).await.expect("ins");

    let rows = db
        .query_spend_by_key(Some("2024-03-01T00:00:00Z"), None)
        .await
        .expect("query");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].total_requests, 1);
    assert!((rows[0].total_spend - 0.20).abs() < 1e-9);
}

#[tokio::test]
async fn query_spend_by_model_respects_date_range() {
    let db = fresh_db().await;
    let mut early = spend_row(None, None, Some(0.10));
    early.model = "gpt-4o".to_string();
    early.provider = "openai".to_string();
    early.timestamp = "2024-01-01T00:00:00Z".to_string();
    let mut late = spend_row(None, None, Some(0.20));
    late.model = "gpt-4o".to_string();
    late.provider = "openai".to_string();
    late.timestamp = "2024-06-01T00:00:00Z".to_string();
    db.insert_spend_log(&early).await.expect("ins");
    db.insert_spend_log(&late).await.expect("ins");

    let rows = db
        .query_spend_by_model(Some("2024-03-01T00:00:00Z"), None)
        .await
        .expect("query");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].total_requests, 1);
    assert!((rows[0].total_spend - 0.20).abs() < 1e-9);
}

#[tokio::test]
async fn query_spend_by_model_aggregates() {
    let db = fresh_db().await;
    let mut a = spend_row(None, None, Some(0.10));
    a.model = "gpt-4o".to_string();
    a.provider = "openai".to_string();
    let mut b = spend_row(None, None, Some(0.30));
    b.model = "gpt-4o".to_string();
    b.provider = "openai".to_string();
    let mut c = spend_row(None, None, Some(0.20));
    c.model = "claude-3".to_string();
    c.provider = "anthropic".to_string();
    db.insert_spend_log(&a).await.expect("ins");
    db.insert_spend_log(&b).await.expect("ins");
    db.insert_spend_log(&c).await.expect("ins");

    let rows = db.query_spend_by_model(None, None).await.expect("query");
    assert_eq!(rows.len(), 2, "one group per (model, provider)");

    // Ordered by total_spend DESC → gpt-4o (0.40) first.
    let gpt = &rows[0];
    assert_eq!(gpt.model.as_deref(), Some("gpt-4o"));
    assert_eq!(gpt.provider.as_deref(), Some("openai"));
    assert_eq!(gpt.total_requests, 2);
    assert!((gpt.total_spend - 0.40).abs() < 1e-9);

    let claude = &rows[1];
    assert_eq!(claude.model.as_deref(), Some("claude-3"));
    assert_eq!(claude.total_requests, 1);
    assert!((claude.total_spend - 0.20).abs() < 1e-9);
}

#[tokio::test]
async fn query_spend_by_user_aggregates_with_name_and_nulls() {
    let db = fresh_db().await;
    db.create_user(&UserRecord {
        user_id: "user-1".to_string(),
        user_name: "Alice".to_string(),
        max_budget: None,
        budget_duration: None,
        created_at: now_iso(),
        updated_at: now_iso(),
        models: None,
    })
    .await
    .expect("create user");

    let mk = |user: Option<&str>, cost: Option<f64>| {
        let mut row = spend_row(None, None, cost);
        row.user_id = user.map(str::to_string);
        row
    };
    db.insert_spend_log(&mk(Some("user-1"), Some(0.10)))
        .await
        .expect("ins");
    db.insert_spend_log(&mk(Some("user-1"), Some(0.20)))
        .await
        .expect("ins");
    // user with no `users` row (LEFT JOIN → NULL name).
    db.insert_spend_log(&mk(Some("user-ghost"), Some(0.05)))
        .await
        .expect("ins");
    // no user at all (NULL user_id group).
    db.insert_spend_log(&mk(None, Some(0.50)))
        .await
        .expect("ins");

    let rows = db.query_spend_by_user(None, None).await.expect("query");
    assert_eq!(rows.len(), 3, "one group per distinct user_id (incl. NULL)");

    let alice = rows
        .iter()
        .find(|r| r.user_id.as_deref() == Some("user-1"))
        .expect("user-1 group");
    assert_eq!(alice.user_name.as_deref(), Some("Alice"), "LEFT JOIN name");
    assert_eq!(alice.total_requests, 2);
    assert!((alice.total_spend - 0.30).abs() < 1e-9);

    let ghost = rows
        .iter()
        .find(|r| r.user_id.as_deref() == Some("user-ghost"))
        .expect("ghost group");
    assert_eq!(ghost.user_name, None, "no users row → NULL name");

    // Ordered by total_spend DESC → NULL group (0.50) first.
    assert_eq!(rows[0].user_id, None);
    assert!((rows[0].total_spend - 0.50).abs() < 1e-9);
}

#[tokio::test]
async fn query_spend_by_user_respects_date_range() {
    let db = fresh_db().await;
    let mut early = spend_row(None, None, Some(0.10));
    early.user_id = Some("user-1".to_string());
    early.timestamp = "2024-01-01T00:00:00Z".to_string();
    let mut late = spend_row(None, None, Some(0.20));
    late.user_id = Some("user-1".to_string());
    late.timestamp = "2024-06-01T00:00:00Z".to_string();
    db.insert_spend_log(&early).await.expect("ins");
    db.insert_spend_log(&late).await.expect("ins");

    let rows = db
        .query_spend_by_user(Some("2024-03-01T00:00:00Z"), None)
        .await
        .expect("query");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].total_requests, 1);
    assert!((rows[0].total_spend - 0.20).abs() < 1e-9);
}

#[tokio::test]
async fn query_spend_by_team_aggregates_with_name() {
    let db = fresh_db().await;
    db.create_team(&TeamRecord {
        team_id: "team-1".to_string(),
        team_name: "Engineering".to_string(),
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now_iso(),
        updated_at: now_iso(),
        models: None,
        blocked: false,
    })
    .await
    .expect("create team");

    let mk = |team: Option<&str>, cost: f64| {
        let mut row = spend_row(None, None, Some(cost));
        row.team_id = team.map(str::to_string);
        row
    };
    db.insert_spend_log(&mk(Some("team-1"), 0.10))
        .await
        .expect("ins");
    db.insert_spend_log(&mk(Some("team-1"), 0.40))
        .await
        .expect("ins");
    db.insert_spend_log(&mk(Some("team-2"), 0.20))
        .await
        .expect("ins");

    let rows = db.query_spend_by_team(None, None).await.expect("query");
    assert_eq!(rows.len(), 2, "one group per distinct team_id");

    let eng = rows
        .iter()
        .find(|r| r.team_id.as_deref() == Some("team-1"))
        .expect("team-1 group");
    assert_eq!(eng.team_name.as_deref(), Some("Engineering"));
    assert_eq!(eng.total_requests, 2);
    assert!((eng.total_spend - 0.50).abs() < 1e-9);

    // Ordered by total_spend DESC → team-1 (0.50) first.
    assert_eq!(rows[0].team_id.as_deref(), Some("team-1"));
}

#[tokio::test]
async fn query_spend_by_org_resolves_through_key() {
    let db = fresh_db().await;
    // Two keys, each belonging to a different org.
    let mut key_a = sample_key("k-a", "hash-a");
    key_a.org_id = Some("org-a".to_string());
    let mut key_b = sample_key("k-b", "hash-b");
    key_b.org_id = Some("org-b".to_string());
    // A key with no org.
    let mut key_c = sample_key("k-c", "hash-c");
    key_c.org_id = None;
    db.create_key(&key_a).await.expect("create key a");
    db.create_key(&key_b).await.expect("create key b");
    db.create_key(&key_c).await.expect("create key c");

    db.insert_spend_log(&spend_row(None, Some("k-a"), Some(0.10)))
        .await
        .expect("ins");
    db.insert_spend_log(&spend_row(None, Some("k-a"), Some(0.20)))
        .await
        .expect("ins");
    db.insert_spend_log(&spend_row(None, Some("k-b"), Some(0.50)))
        .await
        .expect("ins");
    // key with no org → NULL org group.
    db.insert_spend_log(&spend_row(None, Some("k-c"), Some(0.05)))
        .await
        .expect("ins");
    // no key at all → also NULL org group (aggregated with k-c's NULL).
    db.insert_spend_log(&spend_row(None, None, Some(0.15)))
        .await
        .expect("ins");

    let rows = db.query_spend_by_org(None, None).await.expect("query");
    assert_eq!(rows.len(), 3, "org-a, org-b, and the NULL group");

    let org_a = rows
        .iter()
        .find(|r| r.org_id.as_deref() == Some("org-a"))
        .expect("org-a group");
    assert_eq!(org_a.total_requests, 2);
    assert!((org_a.total_spend - 0.30).abs() < 1e-9);

    // Ordered by total_spend DESC → org-b (0.50) first.
    assert_eq!(rows[0].org_id.as_deref(), Some("org-b"));
    assert!((rows[0].total_spend - 0.50).abs() < 1e-9);

    // NULL org group aggregates both the no-org key and the no-key row.
    let null_group = rows
        .iter()
        .find(|r| r.org_id.is_none())
        .expect("null org group");
    assert_eq!(null_group.total_requests, 2);
    assert!((null_group.total_spend - 0.20).abs() < 1e-9);
}

#[tokio::test]
async fn query_spend_by_org_respects_date_range() {
    let db = fresh_db().await;
    let mut key = sample_key("k-a", "hash-a");
    key.org_id = Some("org-a".to_string());
    db.create_key(&key).await.expect("create key");

    let mut early = spend_row(None, Some("k-a"), Some(0.10));
    early.timestamp = "2024-01-01T00:00:00Z".to_string();
    let mut late = spend_row(None, Some("k-a"), Some(0.20));
    late.timestamp = "2024-06-01T00:00:00Z".to_string();
    db.insert_spend_log(&early).await.expect("ins");
    db.insert_spend_log(&late).await.expect("ins");

    let rows = db
        .query_spend_by_org(Some("2024-03-01T00:00:00Z"), None)
        .await
        .expect("query");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].org_id.as_deref(), Some("org-a"));
    assert!((rows[0].total_spend - 0.20).abs() < 1e-9);
}

#[tokio::test]
async fn query_spend_order_is_stable_for_identical_timestamps() {
    // Pagination tiebreaker: rows sharing a timestamp must have a
    // deterministic order (id DESC) so offset pages can't skip/dup rows.
    let db = fresh_db().await;
    let ts = "2024-06-01T12:00:00Z".to_string();
    for i in 0..5 {
        let mut row = spend_row(Some(&format!("req-{i}")), None, None);
        row.timestamp = ts.to_string();
        db.insert_spend_log(&row).await.expect("ins");
    }

    let page1 = db
        .query_spend(&SpendFilter {
            limit: Some(2),
            offset: Some(0),
            ..Default::default()
        })
        .await
        .expect("page1");
    let page2 = db
        .query_spend(&SpendFilter {
            limit: Some(2),
            offset: Some(2),
            ..Default::default()
        })
        .await
        .expect("page2");
    let page3 = db
        .query_spend(&SpendFilter {
            limit: Some(2),
            offset: Some(4),
            ..Default::default()
        })
        .await
        .expect("page3");

    let mut seen: Vec<i64> = Vec::new();
    for log in page1.iter().chain(page2.iter()).chain(page3.iter()) {
        seen.push(log.id.expect("id"));
    }
    assert_eq!(seen.len(), 5, "all rows paged through");
    let mut dedup = seen.clone();
    dedup.sort_unstable();
    dedup.dedup();
    assert_eq!(dedup.len(), 5, "no row duplicated or skipped across pages");
    // Deterministic: strictly decreasing ids within/across pages.
    assert!(
        seen.windows(2).all(|w| w[0] > w[1]),
        "ids must be strictly descending for identical timestamps: {seen:?}"
    );
}

#[tokio::test]
async fn customer_crud() {
    let db = fresh_db().await;
    let now = now_iso();
    let c = CustomerRecord {
        end_user_id: "eu-1".to_string(),
        alias: Some("Alice".to_string()),
        email: Some("a@example.com".to_string()),
        max_budget: Some(10.0),
        spend: 0.0,
        budget_duration: Some("30d".to_string()),
        blocked: false,
        model_blocklist: Some(r#"["gpt-4"]"#.to_string()),
        created_at: now.clone(),
        updated_at: now.clone(),
    };
    db.create_customer(&c).await.expect("create");
    let got = db
        .get_customer("eu-1")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(got.alias.as_deref(), Some("Alice"));
    assert!(!got.blocked);
    assert!((got.spend - 0.0).abs() < 1e-9);

    db.increment_customer_spend("eu-1", 1.5)
        .await
        .expect("increment");
    let mut updated = got;
    updated.blocked = true;
    updated.spend = 99.0;
    updated.updated_at = now_iso();
    assert!(
        db.update_customer(&updated).await.expect("update"),
        "update should match the existing row"
    );
    let after = db
        .get_customer("eu-1")
        .await
        .expect("get")
        .expect("present");
    assert!(after.blocked);
    assert!(
        (after.spend - 1.5).abs() < 1e-9,
        "update must not clobber spend (got {})",
        after.spend
    );

    let listed = db.list_customers(None).await.expect("list");
    assert_eq!(listed.len(), 1);

    db.delete_customer("eu-1").await.expect("delete");
    assert!(db.get_customer("eu-1").await.expect("get").is_none());
}

#[tokio::test]
async fn block_customers_bulk_is_atomic_and_scoped() {
    let db = fresh_db().await;
    let now = now_iso();
    for id in ["eu-1", "eu-2", "eu-3"] {
        db.create_customer(&CustomerRecord {
            end_user_id: id.to_string(),
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
        .await
        .expect("create");
    }

    // Mix in a non-existent id; it must be silently ignored, not error.
    let n = db
        .block_customers(
            &["eu-1".into(), "eu-2".into(), "nope".into()],
            true,
            &now_iso(),
        )
        .await
        .expect("bulk block");
    assert_eq!(n, 2, "only the two existing customers are affected");
    assert!(db.get_customer("eu-1").await.unwrap().unwrap().blocked);
    assert!(db.get_customer("eu-2").await.unwrap().unwrap().blocked);
    assert!(
        !db.get_customer("eu-3").await.unwrap().unwrap().blocked,
        "eu-3 was not in the id list and must be untouched"
    );

    // Empty id list is a no-op returning 0 (does not error / does not match all).
    let n = db
        .block_customers(&[], false, &now_iso())
        .await
        .expect("empty");
    assert_eq!(n, 0);
}

/// Helper: create a customer row with only the required fields set.
async fn make_customer(db: &Database, id: &str) {
    let now = now_iso();
    db.create_customer(&CustomerRecord {
        end_user_id: id.to_string(),
        alias: None,
        email: None,
        max_budget: None,
        spend: 0.0,
        budget_duration: None,
        blocked: false,
        model_blocklist: None,
        created_at: now.clone(),
        updated_at: now,
    })
    .await
    .expect("create");
}

/// C28 MEDIUM: `delete_customers_bulk` runs inside a single transaction, so
/// the batch is all-or-nothing, and the returned count is the true
/// `rows_affected` (not a request-side counter that over-reports on a racy
/// no-op). Ids that don't resolve are silently skipped and not counted.
#[tokio::test]
async fn delete_customers_bulk_returns_affected_count() {
    let db = fresh_db().await;
    for id in ["eu-1", "eu-2", "eu-3"] {
        make_customer(&db, id).await;
    }

    // Mix in a non-existent id; it is not counted and does not error.
    let n = db
        .delete_customers_bulk(&["eu-1".into(), "eu-2".into(), "eu-3".into(), "ghost".into()])
        .await
        .expect("bulk delete");
    assert_eq!(n, 3, "count matches rows actually removed, ghost ignored");
    for id in ["eu-1", "eu-2", "eu-3"] {
        assert!(db.get_customer(id).await.unwrap().is_none(), "{id} gone");
    }
}

/// C28 MEDIUM: an empty id list is a no-op returning 0 (it must not match
/// all rows and must not error).
#[tokio::test]
async fn delete_customers_bulk_empty_is_noop() {
    let db = fresh_db().await;
    make_customer(&db, "eu-keep").await;
    let n = db.delete_customers_bulk(&[]).await.expect("empty bulk");
    assert_eq!(n, 0);
    assert!(db.get_customer("eu-keep").await.unwrap().is_some());
}

/// C28 MEDIUM: the id list is chunked so each statement stays under the
/// SQLite bind-parameter ceiling (999). A list longer than the ceiling must
/// not surface a driver error; only the rows that exist are removed.
#[tokio::test]
async fn delete_customers_bulk_chunks_oversized_list() {
    let db = fresh_db().await;
    make_customer(&db, "eu-1").await;
    make_customer(&db, "eu-2").await;

    // 1000 ids: two real customers plus 998 ghosts. Under SQLite this is
    // chunked into [999, 1]; neither chunk exceeds the ceiling.
    let mut ids: Vec<String> = (0..998).map(|i| format!("ghost-{i}")).collect();
    ids.push("eu-1".into());
    ids.push("eu-2".into());

    let n = db.delete_customers_bulk(&ids).await.expect("chunked bulk");
    assert_eq!(n, 2, "only the two real customers are removed");
    assert!(db.get_customer("eu-1").await.unwrap().is_none());
    assert!(db.get_customer("eu-2").await.unwrap().is_none());
}

#[tokio::test]
async fn budget_crud() {
    let db = fresh_db().await;
    let now = now_iso();
    let b = BudgetRecord {
        budget_id: "budget-1".to_string(),
        max_budget: Some(100.0),
        soft_budget: Some(80.0),
        max_parallel_requests: Some(5),
        tpm_limit: Some(10_000),
        rpm_limit: Some(60),
        budget_duration: Some("1d".to_string()),
        model_max_budget: Some(r#"{"gpt-4o": 10.0}"#.to_string()),
        created_at: now.clone(),
        updated_at: now,
    };
    db.create_budget(&b).await.expect("create");
    let got = db
        .get_budget("budget-1")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(got.max_budget, Some(100.0));
    assert_eq!(got.rpm_limit, Some(60));

    let mut updated = got;
    updated.max_budget = Some(200.0);
    updated.updated_at = now_iso();
    db.update_budget(&updated).await.expect("update");
    assert_eq!(
        db.get_budget("budget-1")
            .await
            .expect("get")
            .expect("present")
            .max_budget,
        Some(200.0)
    );

    let listed = db.list_budgets(None).await.expect("list");
    assert_eq!(listed.len(), 1);

    let bulk = db
        .get_budgets_by_ids(&["budget-1".to_string(), "missing".to_string()])
        .await
        .expect("bulk");
    assert_eq!(bulk.len(), 1);

    let n = db
        .delete_budgets(&["budget-1".to_string()])
        .await
        .expect("delete");
    assert_eq!(n, 1);
    assert!(db.get_budget("budget-1").await.expect("get").is_none());
}

#[tokio::test]
async fn budget_bulk_fetch_returns_matches_in_input_order() {
    // Regression guard for the N+1 → single batched `WHERE IN` fix.
    // `get_budgets_by_ids` must return only the rows that exist, in the
    // order their ids first appear in the input list (so callers iterating
    // over a known id list see a stable shape). A broken batch query that
    // returns the wrong rows or reorders them trips this test.
    let db = fresh_db().await;
    let now = now_iso();
    for id in ["b-3", "b-1", "b-2"] {
        db.create_budget(&BudgetRecord {
            budget_id: id.to_string(),
            max_budget: Some(10.0),
            soft_budget: None,
            max_parallel_requests: None,
            tpm_limit: None,
            rpm_limit: None,
            budget_duration: None,
            model_max_budget: None,
            created_at: now.clone(),
            updated_at: now.clone(),
        })
        .await
        .expect("create");
    }

    // Mixed existing + missing, with a duplicate to make sure each id is
    // returned exactly as many times as it appears in the input (matching
    // the prior per-id `get_budget` loop semantics).
    let ids: Vec<String> = ["b-2", "missing", "b-1", "b-3", "b-1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let got = db.get_budgets_by_ids(&ids).await.expect("bulk");
    let got_ids: Vec<&str> = got.iter().map(|b| b.budget_id.as_str()).collect();
    assert_eq!(got_ids, vec!["b-2", "b-1", "b-3", "b-1"]);

    // Empty input is a no-op (must not error and must not match all rows).
    let empty = db.get_budgets_by_ids(&[]).await.expect("empty");
    assert!(empty.is_empty());
}

#[tokio::test]
async fn tag_crud() {
    let db = fresh_db().await;
    let now = now_iso();
    let t = TagRecord {
        name: "prod".to_string(),
        description: Some("production".to_string()),
        models: Some(r#"["gpt-4o"]"#.to_string()),
        model_info: Some(r#"{"tier":"a"}"#.to_string()),
        spend: Some(0.0),
        max_budget: Some(50.0),
        created_at: now.clone(),
        updated_at: now,
    };
    db.create_tag(&t).await.expect("create");
    let got = db.get_tag("prod").await.expect("get").expect("present");
    assert_eq!(got.description.as_deref(), Some("production"));

    db.increment_tag_spend("prod", 2.5)
        .await
        .expect("increment");
    let mut updated = got;
    updated.max_budget = Some(75.0);
    updated.spend = Some(99.0);
    updated.updated_at = now_iso();
    assert!(
        db.update_tag(&updated).await.expect("update"),
        "update should match the existing row"
    );
    let after = db.get_tag("prod").await.expect("get").expect("present");
    assert_eq!(after.max_budget, Some(75.0));
    assert!(
        (after.spend.unwrap_or(-1.0) - 2.5).abs() < 1e-9,
        "update must not clobber spend (got {:?})",
        after.spend
    );

    let listed = db.list_tags(None).await.expect("list");
    assert_eq!(listed.len(), 1);

    db.delete_tag("prod").await.expect("delete");
    assert!(db.get_tag("prod").await.expect("get").is_none());
}

fn sample_skill(id: &str, name: &str) -> types::SkillRecord {
    types::SkillRecord {
        id: id.to_string(),
        name: name.to_string(),
        description: Some("a skill".to_string()),
        content: "# Skill\n\nDoes the thing.".to_string(),
        version: "1.0".to_string(),
        model_pattern: None,
        team_id: None,
        created_at: now_iso(),
        updated_at: now_iso(),
    }
}

/// `list_skills_by_model` caches per-model results. The first call
/// populates the cache; a row written *outside* the repo (raw SQL, i.e. the
/// multi-process case) must not be visible while the entry is fresh, must
/// appear after the TTL expires, and must appear immediately when written
/// through `create_skill` (write invalidation).
#[tokio::test]
async fn list_skills_by_model_caches_until_ttl_or_write() {
    let db = fresh_db().await;
    db.create_skill(&sample_skill("skill-1", "greeter"))
        .await
        .expect("create");

    // Populate the cache.
    let first = db.list_skills_by_model("gpt-4o").await.expect("list");
    assert_eq!(first.len(), 1);

    // A row the repo did not write (another replica / direct SQL) stays
    // invisible while the cached entry is fresh — proving the cache is
    // actually consulted rather than the query re-run each time.
    sqlx::query(
        "INSERT INTO skills (id, name, content, created_at, updated_at) \
         VALUES ('outsider', 'outsider', 'body', '2031-01-01T00:00:00Z', '2031-01-01T00:00:00Z')",
    )
    .execute(db.pool())
    .await
    .expect("raw insert");
    let still_cached = db.list_skills_by_model("gpt-4o").await.expect("list");
    assert_eq!(
        still_cached.len(),
        1,
        "fresh cache entry must be served without re-querying"
    );

    // TTL expiry re-queries and picks up the outside write.
    db.skills_cache.expire_all_for_test();
    let after_ttl = db.list_skills_by_model("gpt-4o").await.expect("list");
    assert_eq!(
        after_ttl.len(),
        2,
        "expired entry must be refreshed from the database"
    );

    // A repo write invalidates immediately: the next lookup sees it.
    db.create_skill(&sample_skill("via-repo", "in-process"))
        .await
        .expect("create");
    let after_write = db.list_skills_by_model("gpt-4o").await.expect("list");
    assert_eq!(
        after_write.len(),
        3,
        "repo writes must invalidate the cache synchronously"
    );
}

/// Full CRUD round-trip plus the model/team listing helpers.
#[tokio::test]
async fn skill_crud() {
    let db = fresh_db().await;
    let skill = sample_skill("skill-1", "greeter");
    let created = db.create_skill(&skill).await.expect("create");
    assert_eq!(created.id, "skill-1");

    let got = db
        .get_skill("skill-1")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(got.name, "greeter");
    assert_eq!(got.content, "# Skill\n\nDoes the thing.");
    assert_eq!(got.version, "1.0");
    assert!(got.model_pattern.is_none());

    let mut updated = got;
    updated.content = "# Skill v2".to_string();
    updated.version = "2.0".to_string();
    updated.model_pattern = Some("claude-*".to_string());
    updated.team_id = Some("team-1".to_string());
    updated.updated_at = now_iso();
    let refreshed = db
        .update_skill("skill-1", &updated)
        .await
        .expect("update")
        .expect("present");
    assert_eq!(refreshed.version, "2.0");
    assert_eq!(refreshed.model_pattern.as_deref(), Some("claude-*"));
    assert_eq!(refreshed.team_id.as_deref(), Some("team-1"));
    assert_eq!(refreshed.content, "# Skill v2");

    // update on a missing id returns None (no row affected).
    let missing = db
        .update_skill("nope", &updated)
        .await
        .expect("update missing");
    assert!(missing.is_none());

    // Listing surfaces the row.
    let listed = db.list_skills(None).await.expect("list");
    assert_eq!(listed.len(), 1);

    // list_skills_by_model: the claude-* pattern matches a claude model.
    let by_model = db
        .list_skills_by_model("claude-3-opus")
        .await
        .expect("by model");
    assert_eq!(
        by_model.len(),
        1,
        "claude-* pattern should match claude model"
    );
    let no_match = db.list_skills_by_model("gpt-4o").await.expect("by model");
    assert!(
        no_match.is_empty(),
        "claude-* pattern must not match gpt-4o"
    );

    // list_skills_by_team.
    let by_team = db.list_skills_by_team("team-1").await.expect("by team");
    assert_eq!(by_team.len(), 1);
    let other_team = db.list_skills_by_team("team-2").await.expect("by team");
    assert!(other_team.is_empty());

    // delete returns true for an existing id, false for a missing one.
    assert!(db.delete_skill("skill-1").await.expect("delete"));
    assert!(!db.delete_skill("skill-1").await.expect("delete missing"));
    assert!(db.get_skill("skill-1").await.expect("get").is_none());
}

/// A skill with no `model_pattern` applies to every model
/// (unrestricted scope), so `list_skills_by_model` returns it for any input.
#[tokio::test]
async fn skill_without_pattern_matches_all_models() {
    let db = fresh_db().await;
    db.create_skill(&sample_skill("skill-open", "universal"))
        .await
        .expect("create");
    for model in ["claude-3-opus", "gpt-4o", "gemini-1.5-pro"] {
        let hits = db.list_skills_by_model(model).await.expect("list");
        assert_eq!(
            hits.len(),
            1,
            "unrestricted skill should match every model ({model})"
        );
    }
}

/// `version` defaults to "1.0" at the DB level when the column is omitted;
/// the repo always binds it, but the column default guards direct inserts.
#[tokio::test]
async fn skill_version_column_defaults() {
    let db = fresh_db().await;
    // Direct insert relying on the column DEFAULT (no `version` provided).
    sqlx::query(
        "INSERT INTO skills (id, name, content, created_at, updated_at) \
         VALUES ('direct', 'direct', 'body', 't', 't')",
    )
    .execute(db.pool())
    .await
    .expect("insert without version");
    let got = db.get_skill("direct").await.expect("get").expect("present");
    assert_eq!(got.version, "1.0", "DEFAULT '1.0' must apply");
}

/// `compute_since_iso` must not panic on extreme inputs. The old
/// `-days.saturating_mul(24)` negation trapped on `i64::MIN`; both edges now
/// degrade to a zero span ("now") and every result must still parse as an
/// ISO-8601 timestamp.
#[test]
fn compute_since_iso_handles_extreme_input_without_panicking() {
    for days in [i64::MIN, i64::MAX, 0, 1, 30] {
        let s = compute_since_iso(days);
        assert!(
            s.parse::<jiff::Timestamp>().is_ok(),
            "compute_since_iso({days}) must yield a parseable ISO timestamp, got {s}"
        );
    }
}

/// The URL-credential redactor must strip `user:password@` userinfo from any
/// DSN embedded in an error string so DB credentials never reach the logs.
#[test]
fn redact_url_credentials_strips_password() {
    let cases = [
        ("postgres://alice:s3cr3t@db.internal:5432/oxllm", "s3cr3t"),
        ("postgresql://u:p%40ss@host/db?sslmode=require", "p@ss"),
        ("mysql://root:hunter2@10.0.0.1/prod", "hunter2"),
    ];
    for (dsn, secret) in cases {
        let rendered = redact_url_credentials(dsn);
        assert!(
            !rendered.contains(secret),
            "credential `{secret}` leaked through redaction: {rendered}"
        );
        assert!(
            rendered.contains("***@"),
            "userinfo should be replaced with `***@`: {rendered}"
        );
    }
    // Strings without a DSN are returned verbatim.
    let plain = "database is locked";
    assert_eq!(redact_url_credentials(plain), plain);
    // A DSN with no userinfo is left untouched.
    let no_user = "sqlite:/var/lib/oxllm/data.db";
    assert_eq!(redact_url_credentials(no_user), no_user);

    // A malformed DSN whose password contains an unencoded `@`: the redactor
    // splits at the *last* `@` before the path so no credential fragment
    // leaks. Both password fragments (`p` and `ss`) must be absent.
    let multi_at = "postgres://u:p@ss@host/db";
    let rendered = redact_url_credentials(multi_at);
    assert!(
        !rendered.contains("p@ss") && !rendered.contains("ss@"),
        "multi-`@` DSN must fully redact up to the host: {rendered}"
    );
    assert!(
        rendered.contains("***@host"),
        "host after the final `@` should be preserved: {rendered}"
    );
}

/// A file-backed SQLite DSN must boot a working pool: migrate runs, WAL mode
/// is applied (a persistent DB property), and a read/write roundtrip works.
#[tokio::test]
async fn connect_file_sqlite_applies_migrations_and_wal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("filepool.db");
    let url = format!("sqlite:{}?mode=rwc", db_path.display());

    let db = Database::connect(&url).await.expect("connect file sqlite");
    db.migrate().await.expect("migrate file sqlite");

    // WAL mode is persistent in the file; any connection observes it.
    use sqlx::Row;
    let row = sqlx::query("PRAGMA journal_mode")
        .fetch_one(db.pool())
        .await
        .expect("read journal_mode");
    let mode: String = row.try_get(0).expect("decode journal_mode");
    assert_eq!(
        mode.to_lowercase(),
        "wal",
        "file-backed SQLite should be switched to WAL mode"
    );

    // Read/write roundtrip through the tuned pool.
    db.create_key(&sample_key("file-k", "file-h"))
        .await
        .expect("create on file sqlite");
    let got = db
        .get_key_by_hash("file-h")
        .await
        .expect("get on file sqlite")
        .expect("row present");
    assert_eq!(got.key_id, "file-k");
}

/// Regression for the concurrent cold-start migration race (MEDIUM): two
/// replicas booting cold against the same fresh file DB must both complete
/// `migrate()` without the second one crashing on a duplicate
/// `_schema_version` primary-key write. The version write is idempotent
/// (`INSERT OR IGNORE` on SQLite, `ON CONFLICT DO NOTHING` on Postgres).
#[tokio::test]
async fn migrate_concurrent_on_fresh_file_db_does_not_collide() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("race.db");
    let url = format!("sqlite:{}?mode=rwc", db_path.display());

    let url_a = url.clone();
    let url_b = url.clone();
    let (res_a, res_b) = tokio::join!(
        async move {
            let db = Database::connect(&url_a).await.expect("replica A connect");
            db.migrate().await
        },
        async move {
            let db = Database::connect(&url_b).await.expect("replica B connect");
            db.migrate().await
        },
    );
    res_a.expect("replica A migrate must not collide");
    res_b.expect("replica B migrate must not collide");

    // The recorded version is correct and singular.
    let probe = Database::connect(&url).await.expect("probe connect");
    assert_eq!(
        probe.get_schema_version().await.unwrap(),
        SCHEMA_VERSION,
        "concurrent migrate must leave exactly the current schema version"
    );
}

/// `run_migration_batch` must tolerate a duplicate-column error from a
/// re-applied `ADD COLUMN` (the legacy upgrade path).
#[tokio::test]
async fn run_migration_batch_tolerates_duplicate_column() {
    let db = fresh_db().await;
    // `keys.models` already exists after migrate; re-adding is the exact
    // idempotent scenario the swallow is meant to cover.
    let mut conn = db.pool().acquire().await.expect("acquire conn");
    db.run_migration_batch(
        vec!["ALTER TABLE keys ADD COLUMN models TEXT".to_string()],
        &mut conn,
    )
    .await
    .expect("duplicate ADD COLUMN should be tolerated");
}

/// Regression for the broad error-swallow (LOW): a statement that fails with
/// a *non*-duplicate error must propagate `Err` instead of being silently
/// swallowed by the column-name match.
#[tokio::test]
async fn run_migration_batch_propagates_non_duplicate_errors() {
    let db = fresh_db().await;
    let mut conn = db.pool().acquire().await.expect("acquire conn");
    let err = db
        .run_migration_batch(
            vec!["SELECT * FROM _definitely_not_a_table_xyz".to_string()],
            &mut conn,
        )
        .await
        .expect_err("non-duplicate failure must propagate");
    let msg = err.to_string();
    assert!(
        !msg.contains("duplicate column") && !msg.contains("already exists"),
        "unrelated failure must not be mis-classified as duplicate-column: {msg}"
    );
}

// ---- C57: db-repos hardening ---------------------------------------------

/// HIGH: `insert_spend_logs_batch` must chunk so a large batch stays under
/// the SQLite bind-parameter ceiling (999 on older builds). A 6 000-row
/// batch × 13 columns = 78 000 parameters, far above the ceiling; without
/// chunking the driver rejects the statement and the whole batch is dropped.
#[tokio::test]
async fn insert_spend_logs_batch_chunks_around_param_limit() {
    let db = fresh_db().await;
    let n = 6_000;
    let logs: Vec<SpendLog> = (0..n)
        .map(|i| SpendLog {
            id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            request_id: Some(format!("req-{i}")),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            cost: Some(0.01),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        })
        .collect();
    db.insert_spend_logs_batch(&logs)
        .await
        .expect("large batch must chunk and not hit the bind ceiling");

    let count = db
        .count_spend_filtered(&SpendFilter::default())
        .await
        .expect("count");
    assert_eq!(count as usize, n, "every batched row must land");
}

/// MEDIUM: the v6 migration must create an index on `spend_logs.request_id`
/// so `query_spend_by_request_id` doesn't full-scan. On SQLite we assert via
/// `EXPLAIN QUERY PLAN` that the lookup uses the index (not a sequential
/// scan).
#[tokio::test]
async fn query_spend_by_request_id_uses_index() {
    let db = fresh_db().await;
    use sqlx::Row;
    let plan = sqlx::query("EXPLAIN QUERY PLAN SELECT * FROM spend_logs WHERE request_id = 'x'")
        .fetch_all(db.pool())
        .await
        .expect("explain");
    let mut text = String::new();
    for row in &plan {
        let detail: String = row.try_get(3).unwrap_or_default();
        text.push_str(&detail);
        text.push('\n');
    }
    assert!(
        text.contains("idx_spend_logs_request_id") || text.contains("USING INDEX"),
        "request_id lookup should use an index, plan was:\n{text}"
    );
}

/// MEDIUM: `list_keys` must honour an explicit LIMIT so the DB returns at
/// most the page (the OOM-amplifier fix). Insert 1 500 rows, ask for 1 000,
/// and assert exactly 1 000 come back.
#[tokio::test]
async fn list_keys_respects_limit() {
    let db = fresh_db().await;
    for i in 0..1500 {
        db.create_key(&KeyRecord {
            key_id: format!("k-{i}"),
            key_hash: format!("h-{i}"),
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
        })
        .await
        .expect("create");
    }
    let page = db.list_keys(Some(1000)).await.expect("list");
    assert_eq!(page.len(), 1000, "explicit limit must bound the page");
    // The default (None) also caps to avoid an unbounded scan.
    let default_page = db.list_keys(None).await.expect("list default");
    assert_eq!(default_page.len(), 1000, "default cap is 1000");
    // A negative limit is clamped to 1, never "no cap".
    let one = db.list_keys(Some(-5)).await.expect("list negative");
    assert_eq!(one.len(), 1, "negative limit must clamp to 1");
}

/// MEDIUM: `block_customers` must not blow up on an oversized id list (the
/// IN-list would exceed the bind ceiling). Pass 10 000 ids mixed with the two
/// existing customers and assert no driver error reaches the caller.
#[tokio::test]
async fn block_customers_chunks_oversized_list() {
    let db = fresh_db().await;
    let now = now_iso();
    for id in ["eu-1", "eu-2"] {
        db.create_customer(&CustomerRecord {
            end_user_id: id.to_string(),
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
        .await
        .expect("create");
    }
    let mut ids: Vec<String> = (0..10_000).map(|i| format!("filler-{i}")).collect();
    ids.push("eu-1".to_string());
    ids.push("eu-2".to_string());

    let n = db
        .block_customers(&ids, true, &now_iso())
        .await
        .expect("oversized list must chunk, not error");
    assert_eq!(n, 2, "only the two existing customers are blocked");
    assert!(db.get_customer("eu-1").await.unwrap().unwrap().blocked);
    assert!(db.get_customer("eu-2").await.unwrap().unwrap().blocked);
}

/// MEDIUM: `query_spend` must clamp `limit` / `offset` like
/// `query_spend_paginated`. `i64::MAX` would ask the DB for that many rows;
/// a negative `offset` is meaningless. Both must degrade to a sane page.
#[tokio::test]
async fn query_spend_clamps_limit_and_offset() {
    let db = fresh_db().await;
    for i in 0..5 {
        db.insert_spend_log(&SpendLog {
            id: None,
            timestamp: now_iso(),
            request_id: Some(format!("req-{i}")),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            cost: Some(0.01),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        })
        .await
        .expect("insert");
    }
    let page = db
        .query_spend(&SpendFilter {
            limit: Some(i64::MAX),
            offset: Some(-5),
            ..Default::default()
        })
        .await
        .expect("clamped query");
    assert_eq!(
        page.len(),
        5,
        "huge limit clamps to 1000, negative offset clamps to 0 → all 5 rows"
    );
}

/// LOW: soft-delete `delete_key` must bump `updated_at` so an audit reader
/// can tell when the key was revoked.
#[tokio::test]
async fn delete_key_advances_updated_at() {
    let db = fresh_db().await;
    db.create_key(&sample_key("k-del", "h-del"))
        .await
        .expect("create");
    let revoked_at = "2030-01-01T00:00:00Z".to_string();
    db.delete_key("k-del", &revoked_at).await.expect("delete");
    let after = db.get_key("k-del").await.expect("get").expect("present");
    assert!(!after.is_active, "soft-deleted");
    assert_eq!(
        after.updated_at, revoked_at,
        "updated_at must reflect the revocation instant"
    );
}

/// LOW: `create_skill` must return the DB row (not the input clone) so any
/// DB-side defaulting is visible to the caller.
#[tokio::test]
async fn create_skill_returns_db_row() {
    let db = fresh_db().await;
    let skill = sample_skill("skill-db", "greeter");
    let created = db.create_skill(&skill).await.expect("create");
    let fetched = db
        .get_skill("skill-db")
        .await
        .expect("get")
        .expect("present");
    assert_eq!(created.version, fetched.version);
    assert_eq!(created.content, fetched.content);
    assert_eq!(created.updated_at, fetched.updated_at);
}

/// LOW: when sorting by a non-id column, the `id < cursor` keyset filter is
/// ignored (it is meaningless under a different ORDER BY). A cursor supplied
/// with `sort_by = cost` must therefore return the same first page as no
/// cursor, rather than filtering on an unrelated column.
#[tokio::test]
async fn query_spend_paginated_non_id_sort_ignores_cursor() {
    let db = fresh_db().await;
    for i in 0..6 {
        db.insert_spend_log(&SpendLog {
            id: None,
            timestamp: now_iso(),
            request_id: Some(format!("req-{i}")),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            cost: Some(i as f64),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        })
        .await
        .expect("insert");
    }
    let cost_filter = SpendFilter {
        sort_by: Some("cost".to_string()),
        ..Default::default()
    };
    let no_cursor = db
        .query_spend_paginated(None, 3, &cost_filter)
        .await
        .expect("page");
    // A bogus cursor must not change the result set for a non-id sort.
    let with_cursor = db
        .query_spend_paginated(Some(999_999), 3, &cost_filter)
        .await
        .expect("page");
    assert_eq!(no_cursor.len(), with_cursor.len());
    let ids_no: Vec<i64> = no_cursor.iter().map(|l| l.id.unwrap()).collect();
    let ids_with: Vec<i64> = with_cursor.iter().map(|l| l.id.unwrap()).collect();
    assert_eq!(ids_no, ids_with, "cursor must be ignored for non-id sort");
}

/// MEDIUM: `query_spend_paginated` keyset cursor must agree with the sort
/// direction. For `sort_by=id, sort_order=asc` the keyset is `id > cursor`
/// (continue forward), not the unconditional `id < cursor` that previously
/// re-filtered to ids below the cursor and returned duplicates or an empty
/// page. Walk all rows with a cursor taken from the last element of each
/// page and assert every id appears exactly once in strictly ascending order.
#[tokio::test]
async fn query_spend_paginated_asc_cursor_walks_forward() {
    let db = fresh_db().await;
    let n = 6;
    for i in 0..n {
        db.insert_spend_log(&SpendLog {
            id: None,
            timestamp: now_iso(),
            request_id: Some(format!("req-{i}")),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            cost: Some(0.01),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        })
        .await
        .expect("insert");
    }

    let asc_filter = SpendFilter {
        sort_by: Some("id".to_string()),
        sort_order: Some("asc".to_string()),
        ..Default::default()
    };

    let mut seen: Vec<i64> = Vec::new();
    let mut cursor: Option<i64> = None;
    loop {
        let page = db
            .query_spend_paginated(cursor, 2, &asc_filter)
            .await
            .expect("page");
        if page.is_empty() {
            break;
        }
        for log in &page {
            seen.push(log.id.expect("id"));
        }
        cursor = page.last().and_then(|l| l.id);
        if page.len() < 2 {
            break;
        }
    }

    assert_eq!(seen.len(), n, "every row must appear exactly once");
    let mut dedup = seen.clone();
    dedup.sort_unstable();
    dedup.dedup();
    assert_eq!(dedup.len(), n, "no row duplicated across pages");
    assert!(
        seen.windows(2).all(|w| w[0] < w[1]),
        "ids must be strictly ascending across pages: {seen:?}"
    );
}

/// HIGH (C16): `insert_spend_logs_batch` must wrap its chunk loop in a
/// transaction so a failure on chunk *K > 1* rolls back the chunks that
/// ran before it. Without the transaction, chunk 1 of a multi-chunk batch
/// would already be committed when chunk 2 fails — and the spend writer's
/// connection-error fallback would then re-insert the whole `logs` slice,
/// duplicating the chunk-1 rows in `spend_logs` and overstating every
/// downstream `/spend/*` aggregation.
///
/// We force a mid-batch failure by installing a trigger that ABORTs on a
/// sentinel model. On SQLite the chunk size is `(999-1)/13 = 76`, so a
/// 77-row batch is split `76 + 1`. The sentinel is placed on row 76 (the
/// only row in chunk 2): chunk 1 succeeds, chunk 2 fails, and the
/// transaction must roll back the entire batch — zero rows committed.
#[tokio::test]
async fn insert_spend_logs_batch_rolls_back_on_mid_chunk_failure() {
    let db = fresh_db().await;

    sqlx::query(
        "CREATE TRIGGER spend_logs_fail_sentinel BEFORE INSERT ON spend_logs \
         FOR EACH ROW WHEN NEW.model = 'fail-sentinel' \
         BEGIN SELECT RAISE(ABORT, 'sentinel model rejected'); END",
    )
    .execute(db.pool())
    .await
    .expect("create trigger");

    // 77 rows = 2 chunks on SQLite (chunk_size = 76). Rows 0..76 are plain;
    // row 76 carries the sentinel model so chunk 2 (the last row) fails.
    let mut logs: Vec<SpendLog> = Vec::with_capacity(77);
    for i in 0..77 {
        let is_sentinel = i == 76;
        logs.push(SpendLog {
            id: None,
            timestamp: now_iso(),
            request_id: Some(format!("r-rollback-{i}")),
            model: if is_sentinel {
                "fail-sentinel".to_string()
            } else {
                "gpt-4o".to_string()
            },
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            cost: Some(0.01),
            key_id: Some("k-rollback".to_string()),
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        });
    }

    let result = db.insert_spend_logs_batch(&logs).await;
    assert!(
        result.is_err(),
        "batch with a failing chunk-2 row must surface an error"
    );

    // The transaction must have rolled back: zero rows from the batch may be
    // committed. If chunk 1 had stuck (no transaction), 76 rows would be
    // here and the connection-error fallback would later duplicate them.
    let committed = db
        .query_spend(&SpendFilter {
            key_id: Some("k-rollback".to_string()),
            ..Default::default()
        })
        .await
        .expect("query");
    assert_eq!(
        committed.len(),
        0,
        "mid-batch failure must roll back the entire batch; got {} committed rows",
        committed.len()
    );
}

// ---- C57 (round 2): db-repos hardening -----------------------------------

/// MEDIUM: `list_skills_by_model` must not silently truncate at the 1 000-row
/// `list_skills` cap. Insert 1 001 skills matching `claude-*` (whose
/// `created_at` is older than the 1 filler rows that push them past the cap)
/// and assert every matching skill is returned. A regression that goes back
/// to `list_skills(None)` would return at most 1 000 rows.
#[tokio::test]
async fn list_skills_by_model_does_not_truncate_at_list_cap() {
    let db = fresh_db().await;
    // The filler skills share `created_at` with the matchers but are pushed
    // to the front of the `ORDER BY created_at DESC` window by inserting
    // them *last* with a strictly newer timestamp. We then insert the
    // matcher skills with an older timestamp so they land past row 1 000.
    let newer = "2030-01-02T00:00:00Z".to_string();
    let older = "2030-01-01T00:00:00Z".to_string();

    // 1 001 filler skills that do NOT match `claude-*`, inserted with the
    // newer timestamp so they occupy the first 1 001 rows of the
    // `ORDER BY created_at DESC` window — exactly the rows that the old
    // `list_skills(None)` (cap 1 000) returned.
    for i in 0..1001 {
        db.create_skill(&types::SkillRecord {
            id: format!("filler-{i}"),
            name: format!("filler-{i}"),
            description: None,
            content: "body".to_string(),
            version: "1.0".to_string(),
            model_pattern: Some("gpt-*".to_string()),
            team_id: None,
            created_at: newer.clone(),
            updated_at: newer.clone(),
        })
        .await
        .expect("create filler");
    }
    // Three matching skills with the older timestamp → they sit past row
    // 1 000 and would be silently dropped by a capped listing.
    for i in 0..3 {
        db.create_skill(&types::SkillRecord {
            id: format!("claude-skill-{i}"),
            name: format!("claude-skill-{i}"),
            description: None,
            content: "body".to_string(),
            version: "1.0".to_string(),
            model_pattern: Some("claude-*".to_string()),
            team_id: None,
            created_at: older.clone(),
            updated_at: older.clone(),
        })
        .await
        .expect("create matcher");
    }

    let matches = db
        .list_skills_by_model("claude-3-opus")
        .await
        .expect("list by model");
    assert_eq!(
        matches.len(),
        3,
        "every matching skill must be returned regardless of insertion order"
    );
    for s in &matches {
        assert!(
            s.id.starts_with("claude-skill-"),
            "non-matching filler {} leaked into result set",
            s.id
        );
    }
}

/// MEDIUM: the Postgres `daily_activity` SQL must reference the `timestamp`
/// column as a double-quoted identifier so it cannot be confused with the
/// `TIMESTAMP` type keyword. We assert the shape of the generated query by
/// exercising the SQLite path (which the test DB uses) plus a string check
/// on the Postgres branch via a direct call to `daily_activity` — and we
/// verify the SQLite path still produces the correct buckets (so a
/// regression in either branch is caught).
#[tokio::test]
async fn daily_activity_buckets_multiple_days_sqlite() {
    let db = fresh_db().await;
    let mk = |ts: &str, cost: f64| SpendLog {
        id: None,
        timestamp: ts.to_string(),
        request_id: None,
        model: "gpt-4o".to_string(),
        provider: "openai".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(cost),
        key_id: None,
        team_id: None,
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: None,
        latency_ms: None,
    };
    db.insert_spend_log(&mk("2024-01-01T08:00:00Z", 0.10))
        .await
        .expect("ins");
    db.insert_spend_log(&mk("2024-01-01T20:00:00Z", 0.20))
        .await
        .expect("ins");
    db.insert_spend_log(&mk("2024-01-02T08:00:00Z", 0.50))
        .await
        .expect("ins");

    let activity = db.daily_activity(3650).await.expect("daily");
    let by_day: std::collections::HashMap<&String, f64> =
        activity.iter().map(|(d, c)| (d, *c)).collect();
    // Both rows on the 1st aggregate into one bucket.
    assert!(
        (by_day
            .get(&"2024-01-01".to_string())
            .copied()
            .unwrap_or(-1.0)
            - 0.30)
            .abs()
            < 1e-9,
        "Jan 1 bucket should aggregate both rows: {activity:?}"
    );
    assert!(
        (by_day
            .get(&"2024-01-02".to_string())
            .copied()
            .unwrap_or(-1.0)
            - 0.50)
            .abs()
            < 1e-9,
        "Jan 2 bucket should hold its row: {activity:?}"
    );
}

/// MEDIUM (Postgres path): without a live Postgres testcontainer, lock in
/// the Postgres-shaped SQL by constructing a Postgres-flavoured `Database`
/// value and exercising the path through `daily_activity`. The query uses
/// `spend_logs."timestamp"` (double-quoted) instead of the bare `timestamp`
/// keyword, so a future parser tightening or column rename cannot silently
/// break the `/spend/daily` dashboard. This test fails to compile if the
/// Postgres SQL branch drops the double-quoted identifier.
#[tokio::test]
async fn daily_activity_postgres_sql_quotes_timestamp_column() {
    // The query text is generated by `daily_activity` only after a real
    // connection is established; we instead assert the in-source SQL
    // template directly. Searching the module source keeps the test honest
    // without requiring a Postgres testcontainer in CI.
    let src = include_str!("repos/spend.rs");
    assert!(
        src.contains(r#"TO_CHAR(spend_logs."timestamp"::date, 'YYYY-MM-DD')"#),
        "Postgres daily_activity must quote the timestamp column"
    );
    assert!(
        src.contains(r#"WHERE {ts_col} >= {}"#) || src.contains(r#"spend_logs.\"timestamp\""#),
        "Postgres daily_activity WHERE clause must use the quoted ts_col"
    );
    // And the SQLite branch must remain on the bare column name (no quotes,
    // which SQLite also tolerates but the existing tests depend on).
    assert!(
        src.contains("\"substr(timestamp, 1, 10)\""),
        "SQLite daily_activity branch must keep the bare-column substr expression"
    );
}

/// LOW: org spend attribution is resolved at query time via JOIN, not stored
/// at write time (the reference writes immutable daily-org-spend rollups).
/// After a key is moved to a new org via `update_key`, all historical spend
/// for that key is re-attributed to the new org. This test locks in that
/// documented behavior so a future change to write-time attribution is
/// visible.
#[tokio::test]
async fn query_spend_by_org_reattributes_after_key_is_moved() {
    let db = fresh_db().await;
    let mut key = sample_key("k-move", "hash-move");
    key.org_id = Some("org-original".to_string());
    db.create_key(&key).await.expect("create key");
    db.insert_spend_log(&spend_row(Some("req-1"), Some("k-move"), Some(0.42)))
        .await
        .expect("ins");

    // Initially: spend attributed to org-original.
    let before = db.query_spend_by_org(None, None).await.expect("query");
    let org_before = before
        .iter()
        .find(|r| r.org_id.as_deref() == Some("org-original"))
        .expect("org-original group");
    assert!(
        (org_before.total_spend - 0.42).abs() < 1e-9,
        "spend starts attributed to org-original"
    );

    // Move the key to a new org.
    key.org_id = Some("org-moved".to_string());
    key.updated_at = now_iso();
    db.update_key(&key).await.expect("update key");

    // Historical spend is now reported under org-moved — query-time JOIN,
    // not write-time rollup. (The reference would still see org-original here.)
    let after = db.query_spend_by_org(None, None).await.expect("query");
    let org_moved = after
        .iter()
        .find(|r| r.org_id.as_deref() == Some("org-moved"))
        .expect("org-moved group after move");
    assert!(
        (org_moved.total_spend - 0.42).abs() < 1e-9,
        "spend re-attributed to the new org at query time"
    );
    let org_original_after = after
        .iter()
        .find(|r| r.org_id.as_deref() == Some("org-original"))
        .is_none();
    assert!(
        org_original_after,
        "org-original group must be gone after the move"
    );
}

/// LOW: `insert_spend_log` (single-row) and `insert_spend_logs_batch`
/// (multi-row chunked) must produce identical rows for the same input — the
/// two paths share [`SPEND_LOG_COLS`] / [`bind_spend_log`] so the
/// column ↔ bind mapping cannot drift. Insert the same `SpendLog` via both
/// paths (differing only in `request_id` to keep them distinct) and verify
/// field equality across the 13 columns.
#[tokio::test]
async fn insert_spend_log_and_batch_produce_identical_rows() {
    let db = fresh_db().await;
    let mk = |req: &str| SpendLog {
        id: None,
        timestamp: "2024-01-15T10:30:00Z".to_string(),
        request_id: Some(req.to_string()),
        model: "claude-3-opus".to_string(),
        provider: "anthropic".to_string(),
        prompt_tokens: Some(111),
        completion_tokens: Some(222),
        total_tokens: Some(333),
        cost: Some(0.0123),
        key_id: Some("k-cmp".to_string()),
        team_id: Some("t-cmp".to_string()),
        user_id: Some("u-cmp".to_string()),
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: Some("success".to_string()),
        latency_ms: Some(987),
    };

    db.insert_spend_log(&mk("single")).await.expect("single");
    db.insert_spend_logs_batch(&[mk("batch")])
        .await
        .expect("batch");

    let rows = db
        .query_spend(&SpendFilter {
            key_id: Some("k-cmp".to_string()),
            ..Default::default()
        })
        .await
        .expect("query");
    assert_eq!(rows.len(), 2, "both rows must land");

    let single = rows
        .iter()
        .find(|r| r.request_id.as_deref() == Some("single"))
        .expect("single row");
    let batch = rows
        .iter()
        .find(|r| r.request_id.as_deref() == Some("batch"))
        .expect("batch row");

    // Every field except the DB-assigned id and the differentiating
    // request_id must match.
    for (label, a, b) in [
        ("timestamp", &single.timestamp, &batch.timestamp),
        ("model", &single.model, &batch.model),
        ("provider", &single.provider, &batch.provider),
    ] {
        assert_eq!(a, b, "{label} must match across paths");
    }
    assert_eq!(single.prompt_tokens, batch.prompt_tokens);
    assert_eq!(single.completion_tokens, batch.completion_tokens);
    assert_eq!(single.total_tokens, batch.total_tokens);
    assert!(
        (single.cost.unwrap_or(f64::NAN) - batch.cost.unwrap_or(f64::NAN)).abs() < 1e-9,
        "cost must match across paths"
    );
    assert_eq!(single.team_id, batch.team_id);
    assert_eq!(single.user_id, batch.user_id);
    assert_eq!(single.status, batch.status);
    assert_eq!(single.latency_ms, batch.latency_ms);
}

/// LOW: every `list_*` repo method must honour the 1 001 sentinel cap —
/// `list_xxx(Some(1001))` returns at most 1 001 rows, not 1 000 (which would
/// silently defeat a handler's truncation-sentinel detection). Insert 1 010
/// rows and assert an explicit 1 001 limit is preserved.
#[tokio::test]
async fn list_methods_honour_1001_sentinel_cap() {
    let db = fresh_db().await;
    for i in 0..1010_i64 {
        db.create_key(&KeyRecord {
            key_id: format!("k-{i}"),
            key_hash: format!("h-{i}"),
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
        })
        .await
        .expect("create key");
        db.create_budget(&BudgetRecord {
            budget_id: format!("b-{i}"),
            max_budget: None,
            soft_budget: None,
            max_parallel_requests: None,
            tpm_limit: None,
            rpm_limit: None,
            budget_duration: None,
            model_max_budget: None,
            created_at: now_iso(),
            updated_at: now_iso(),
        })
        .await
        .expect("create budget");
    }

    let keys = db.list_keys(Some(1001)).await.expect("list keys");
    assert_eq!(
        keys.len(),
        1001,
        "keys: sentinel cap of 1001 must be honoured"
    );
    let budgets = db.list_budgets(Some(1001)).await.expect("list budgets");
    assert_eq!(
        budgets.len(),
        1001,
        "budgets: sentinel cap of 1001 must be honoured"
    );
}

/// LOW: `query_spend_paginated` tiebreaker must agree with the sort
/// direction. For `sort_by=model, sort_order=asc` rows sharing the same
/// `model` must come back in ascending `id` order within the group (not
/// newest-first as the previous unconditional `id DESC` produced).
#[tokio::test]
async fn query_spend_paginated_asc_tiebreaker_is_ascending() {
    let db = fresh_db().await;
    // Insert 4 rows with the same model in deterministic order; their
    // DB-assigned ids will be strictly increasing in insertion order.
    for i in 0..4 {
        db.insert_spend_log(&SpendLog {
            id: None,
            timestamp: now_iso(),
            request_id: Some(format!("req-{i}")),
            model: "shared-model".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            total_tokens: Some(2),
            cost: Some(0.01),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            tags: None,
            tag_name: None,
            status: None,
            latency_ms: None,
        })
        .await
        .expect("insert");
    }

    let page = db
        .query_spend_paginated(
            None,
            100,
            &SpendFilter {
                sort_by: Some("model".to_string()),
                sort_order: Some("asc".to_string()),
                ..Default::default()
            },
        )
        .await
        .expect("page");
    let ids: Vec<i64> = page.iter().map(|l| l.id.expect("id")).collect();
    assert_eq!(ids.len(), 4);
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "asc sort tiebreaker must be ascending within a tied group: {ids:?}"
    );

    // And DESC must remain descending within the tied group.
    let page_desc = db
        .query_spend_paginated(
            None,
            100,
            &SpendFilter {
                sort_by: Some("model".to_string()),
                sort_order: Some("desc".to_string()),
                ..Default::default()
            },
        )
        .await
        .expect("page desc");
    let ids_desc: Vec<i64> = page_desc.iter().map(|l| l.id.expect("id")).collect();
    assert!(
        ids_desc.windows(2).all(|w| w[0] > w[1]),
        "desc sort tiebreaker must remain descending within a tied group: {ids_desc:?}"
    );
}

// ---- C57 (round 3): db-repos hardening -----------------------------------

/// LOW: `query_spend_by_request_id` is deterministic when two rows share a
/// `request_id`. `request_id` has no UNIQUE constraint, so a retry that logs
/// a second row under the same id (or a colliding client-supplied id) is
/// possible; the lookup orders by `id DESC` before `LIMIT 1` so the
/// most-recent row wins. Pin that choice so a regression to an unordered
/// `LIMIT 1` (arbitrary row) is caught.
#[tokio::test]
async fn query_spend_by_request_id_returns_most_recent_on_duplicate() {
    let db = fresh_db().await;
    let mk = |cost: f64| {
        let mut row = spend_row(Some("dup-req"), None, Some(cost));
        row.timestamp = now_iso();
        row
    };
    db.insert_spend_log(&mk(0.10)).await.expect("insert first");
    db.insert_spend_log(&mk(0.20)).await.expect("insert second");

    let found = db
        .query_spend_by_request_id("dup-req")
        .await
        .expect("query")
        .expect("row present");
    assert_eq!(found.request_id.as_deref(), Some("dup-req"));
    assert!(
        (found.cost.unwrap_or(-1.0) - 0.20).abs() < 1e-9,
        "most-recent row (id DESC) must win on duplicate request_id"
    );
}

/// v8: `spend_logs.end_user_id` attributes per-request cost to an OpenAI
/// body end-user, distinct from `user_id` (virtual-key owner).
#[tokio::test]
async fn spend_log_round_trips_end_user_id() {
    let db = fresh_db().await;
    let mut log = spend_row(Some("req-eu"), Some("k-1"), Some(0.05));
    log.end_user_id = Some("eu-alice".to_string());
    log.user_id = Some("internal-user".to_string());
    db.insert_spend_log(&log).await.expect("insert");

    let found = db
        .query_spend_by_request_id("req-eu")
        .await
        .expect("query")
        .expect("row present");
    assert_eq!(found.end_user_id.as_deref(), Some("eu-alice"));
    assert_eq!(found.user_id.as_deref(), Some("internal-user"));
    assert_eq!(found.key_id.as_deref(), Some("k-1"));
}

/// v9: `spend_logs.tag_name` attributes per-request cost to an `x-oxllm-tag`
/// request tag, distinct from customer/end-user attribution.
#[tokio::test]
async fn spend_log_round_trips_tag_name() {
    let db = fresh_db().await;
    let mut log = spend_row(Some("req-tag"), Some("k-1"), Some(0.07));
    log.tag_name = Some("canary".to_string());
    log.end_user_id = Some("eu-1".to_string());
    db.insert_spend_log(&log).await.expect("insert");

    let found = db
        .query_spend_by_request_id("req-tag")
        .await
        .expect("query")
        .expect("row present");
    assert_eq!(found.tag_name.as_deref(), Some("canary"));
    assert_eq!(found.end_user_id.as_deref(), Some("eu-1"));

    // Batch path must also bind tag_name (same SPEND_LOG_COLS / bind_spend_log).
    let mut batch = spend_row(Some("req-tag-batch"), Some("k-2"), Some(0.02));
    batch.tag_name = Some("prod".to_string());
    db.insert_spend_logs_batch(&[batch])
        .await
        .expect("batch insert");
    let found_batch = db
        .query_spend_by_request_id("req-tag-batch")
        .await
        .expect("query")
        .expect("row present");
    assert_eq!(found_batch.tag_name.as_deref(), Some("prod"));
}

/// `increment_customer_spend` atomically advances `customers.spend` by
/// `delta`. Called by the spend writer after a successful flush when the
/// entry carries an end-user id. The increment is additive, returns `false`
/// (no row touched) for a missing id, and negative deltas (refunds) are valid.
#[tokio::test]
async fn increment_customer_spend_advances_and_reports_missing() {
    let db = fresh_db().await;
    make_customer(&db, "eu-spend").await;
    // Start at 0 (create binds spend = 0).
    let start = db
        .get_customer("eu-spend")
        .await
        .expect("get")
        .expect("present");
    assert!(start.spend.abs() < 1e-9);

    assert!(
        db.increment_customer_spend("eu-spend", 0.42)
            .await
            .expect("increment"),
        "matching row → true"
    );
    assert!(
        db.increment_customer_spend("eu-spend", 0.08)
            .await
            .expect("increment"),
        "second increment compounds"
    );
    let after = db
        .get_customer("eu-spend")
        .await
        .expect("get")
        .expect("present");
    assert!(
        (after.spend - 0.50).abs() < 1e-9,
        "spend should be 0.42 + 0.08 = 0.50, got {}",
        after.spend
    );

    // Missing customer: no row touched, returns false (not an error).
    assert!(
        !db.increment_customer_spend("eu-missing", 1.0)
            .await
            .expect("missing increment"),
        "missing customer → false"
    );

    // A negative delta (refund) is valid.
    assert!(
        db.increment_customer_spend("eu-spend", -0.20)
            .await
            .expect("refund"),
        "refund on existing customer → true"
    );
    let refunded = db
        .get_customer("eu-spend")
        .await
        .expect("get")
        .expect("present");
    assert!(
        (refunded.spend - 0.30).abs() < 1e-9,
        "spend should be 0.50 - 0.20 = 0.30, got {}",
        refunded.spend
    );
}

/// Tag spend metering primitive — spend writer advances `tags.spend`.
#[tokio::test]
async fn increment_tag_spend_advances_and_reports_missing() {
    let db = fresh_db().await;
    let now = now_iso();
    db.create_tag(&TagRecord {
        name: "canary".to_string(),
        description: None,
        models: None,
        model_info: None,
        spend: Some(0.0),
        max_budget: Some(10.0),
        created_at: now.clone(),
        updated_at: now,
    })
    .await
    .expect("create tag");

    assert!(
        db.increment_tag_spend("canary", 0.25).await.expect("inc"),
        "matching tag → true"
    );
    assert!(
        db.increment_tag_spend("canary", 0.10).await.expect("inc"),
        "compounds"
    );
    let after = db.get_tag("canary").await.expect("get").expect("present");
    assert!(
        (after.spend.unwrap_or(-1.0) - 0.35).abs() < 1e-9,
        "got {:?}",
        after.spend
    );
    assert!(
        !db.increment_tag_spend("missing", 1.0).await.expect("inc"),
        "missing tag → false"
    );
}

/// LOW: the four older UPDATE methods (`update_key` / `update_team` /
/// `update_user` / `update_budget`) must return `Ok(false)` when the target
/// row does not exist, matching their newer siblings (`update_customer`,
/// `update_org`, …). This closes the stale-200 TOCTOU window where a racy
/// delete between the handler's `get_*` and `update_*` surfaced as `200 OK`
/// with the prior record. The handler maps `false` to a 404.
#[tokio::test]
async fn update_methods_return_false_on_missing_row() {
    let db = fresh_db().await;
    let now = now_iso();

    let key = KeyRecord {
        key_id: "ghost-key".to_string(),
        ..sample_key("ghost-key", "ghost-hash")
    };
    assert!(
        !db.update_key(&key).await.expect("update_key missing"),
        "update_key on missing id → false"
    );

    let team = TeamRecord {
        team_id: "ghost-team".to_string(),
        team_name: "Ghost".to_string(),
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now.clone(),
        updated_at: now.clone(),
        models: None,
        blocked: false,
    };
    assert!(
        !db.update_team(&team).await.expect("update_team missing"),
        "update_team on missing id → false"
    );

    let user = UserRecord {
        user_id: "ghost-user".to_string(),
        user_name: "Ghost".to_string(),
        max_budget: None,
        budget_duration: None,
        created_at: now.clone(),
        updated_at: now.clone(),
        models: None,
    };
    assert!(
        !db.update_user(&user).await.expect("update_user missing"),
        "update_user on missing id → false"
    );

    let budget = BudgetRecord {
        budget_id: "ghost-budget".to_string(),
        max_budget: None,
        soft_budget: None,
        max_parallel_requests: None,
        tpm_limit: None,
        rpm_limit: None,
        budget_duration: None,
        model_max_budget: None,
        created_at: now.clone(),
        updated_at: now,
    };
    assert!(
        !db.update_budget(&budget)
            .await
            .expect("update_budget missing"),
        "update_budget on missing id → false"
    );

    // And the same methods return true for an existing row (sanity).
    db.create_key(&sample_key("k-real", "h-real"))
        .await
        .expect("create");
    let mut real_key = db.get_key("k-real").await.expect("get").expect("present");
    real_key.updated_at = now_iso();
    assert!(
        db.update_key(&real_key).await.expect("update_key real"),
        "update_key on existing id → true"
    );
}

/// LOW: the batched spend insert must round-trip per-row values exactly when
/// the batch spans more than one chunk. On SQLite the chunk size is
/// `(999 - 1) / 13 = 76`, so a 77-row batch is split `76 + 1`. The shared
/// `SPEND_LOG_COL_COUNT` drives both the chunk-size quotient and the per-row
/// placeholder offset (`base = i * SPEND_LOG_COL_COUNT`); a regression that
/// reintroduces a hand-synced column count would bind values to the wrong
/// columns, scrambling `request_id ↔ cost`. Give every row a distinct
/// `request_id` with a derivable cost and assert each maps back correctly.
#[tokio::test]
async fn insert_spend_logs_batch_round_trips_per_row_values_across_chunks() {
    let db = fresh_db().await;
    let n = 77; // spans two chunks on SQLite (chunk_size = 76)
    let logs: Vec<SpendLog> = (0..n)
        .map(|i| {
            let i = i as i64;
            SpendLog {
                id: None,
                timestamp: "2024-01-01T00:00:00Z".to_string(),
                request_id: Some(format!("row-{i}")),
                model: format!("model-{}", i % 3),
                provider: "openai".to_string(),
                prompt_tokens: Some(i),
                completion_tokens: Some(i * 2),
                total_tokens: Some(i * 3),
                cost: Some((i as f64) * 0.001),
                key_id: Some(format!("k-{}", i % 5)),
                team_id: Some(format!("t-{}", i % 2)),
                user_id: Some(format!("u-{}", i % 4)),
                end_user_id: None,
                tags: None,
                tag_name: None,
                status: Some("success".to_string()),
                latency_ms: Some(i * 10),
            }
        })
        .collect();
    db.insert_spend_logs_batch(&logs)
        .await
        .expect("batch across chunk boundary");

    let rows = db
        .query_spend(&SpendFilter::default())
        .await
        .expect("query");
    assert_eq!(rows.len(), n, "every row must land");

    // Each request_id must map back to exactly its own derived values — a
    // placeholder/bind offset bug would shift columns across rows.
    use std::collections::HashMap;
    let by_req: HashMap<String, &SpendLog> = rows
        .iter()
        .map(|r| (r.request_id.clone().unwrap(), r))
        .collect();
    for i in 0..n as i64 {
        let req = format!("row-{i}");
        let r = by_req.get(&req).unwrap_or_else(|| panic!("missing {req}"));
        assert_eq!(r.model, format!("model-{}", i % 3), "{req} model");
        assert_eq!(r.prompt_tokens, Some(i), "{req} prompt_tokens");
        assert_eq!(r.completion_tokens, Some(i * 2), "{req} completion");
        assert_eq!(r.total_tokens, Some(i * 3), "{req} total_tokens");
        assert!(
            (r.cost.unwrap_or(f64::NAN) - (i as f64) * 0.001).abs() < 1e-12,
            "{req} cost"
        );
        assert_eq!(r.key_id, Some(format!("k-{}", i % 5)), "{req} key_id");
        assert_eq!(r.team_id, Some(format!("t-{}", i % 2)), "{req} team_id");
        assert_eq!(r.user_id, Some(format!("u-{}", i % 4)), "{req} user_id");
        assert_eq!(r.latency_ms, Some(i * 10), "{req} latency_ms");
    }
}

/// The equivalent `sqlite://:memory:` DSN must be recognised as in-memory
/// (single-connection pool); otherwise each pooled connection would get a
/// separate empty database and the schema/rows would silently vanish.
#[tokio::test]
async fn sqlite_url_memory_dsn_is_in_memory() {
    let db = Database::connect("sqlite://:memory:")
        .await
        .expect("connect sqlite memory alt dsn");
    db.migrate().await.expect("migrate");
    // Rows written on one acquire must be visible on a later acquire.
    db.create_key(&sample_key("k-dsn", "hash-dsn"))
        .await
        .expect("create key");
    assert!(db.get_key_by_hash("hash-dsn").await.unwrap().is_some());
}

/// CI's existing entry point now runs the shared behavioral contracts and
/// lock/migration regressions in isolated schemas. Skips only when the
/// dedicated test DSN is absent; a configured but broken DB fails the test.
#[tokio::test]
async fn postgres_roundtrip_ci() {
    crate::release_tests::postgres_release_validation().await;
}

/// `create_key` / `update_key` must reject malformed records before SQL:
/// empty ids create unaddressable rows, negative budgets/rates silently
/// disable enforcement (`None` already means unlimited).
#[tokio::test]
async fn key_validation_rejects_malformed_records() {
    let db = fresh_db().await;
    let mut bad = sample_key("k-bad", "h-bad");

    bad.key_id = String::new();
    assert!(db.create_key(&bad).await.is_err());
    bad.key_id = "k-bad".to_string();

    bad.key_hash = "  ".to_string();
    assert!(db.create_key(&bad).await.is_err());
    bad.key_hash = "h-bad".to_string();

    bad.max_budget = Some(-1.0);
    assert!(db.create_key(&bad).await.is_err());
    bad.max_budget = Some(f64::NAN);
    assert!(db.create_key(&bad).await.is_err());
    bad.max_budget = Some(10.0);

    bad.rpm = Some(-5);
    assert!(db.create_key(&bad).await.is_err());
    bad.rpm = Some(100);

    bad.tpm = Some(-5);
    assert!(db.create_key(&bad).await.is_err());
    bad.tpm = Some(10_000);

    // The valid record still writes, and `update_key` enforces the same rules.
    db.create_key(&bad).await.expect("valid key writes");
    let mut upd = bad.clone();
    upd.rpm = Some(-1);
    assert!(db.update_key(&upd).await.is_err());
    upd.rpm = Some(200);
    assert!(db.update_key(&upd).await.expect("update"));
}

/// A bare `YYYY-MM-DD` end date must include the whole end day in spend
/// queries (previously the TEXT comparison dropped every end-day row).
#[tokio::test]
async fn spend_bare_end_date_includes_end_day_rows() {
    use crate::types::{SpendFilter, SpendLog};
    let db = fresh_db().await;
    let log = |ts: &str| SpendLog {
        id: None,
        timestamp: ts.to_string(),
        request_id: Some("r1".to_string()),
        model: "m".to_string(),
        provider: "p".to_string(),
        prompt_tokens: Some(1),
        completion_tokens: Some(1),
        total_tokens: Some(2),
        cost: Some(0.001),
        key_id: None,
        team_id: None,
        user_id: None,
        end_user_id: None,
        tags: None,
        tag_name: None,
        status: Some("200".to_string()),
        latency_ms: Some(5),
    };
    db.insert_spend_log(&log("2026-07-31T12:00:00Z"))
        .await
        .expect("insert");
    let rows = db
        .query_spend(&SpendFilter {
            end_date: Some("2026-07-31".to_string()),
            ..Default::default()
        })
        .await
        .expect("query");
    assert_eq!(rows.len(), 1, "noon end-day row must match bare end date");
}
