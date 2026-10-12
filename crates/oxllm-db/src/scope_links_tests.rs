//! Behavioral scope-link contracts and real cross-pool transaction races.

use std::sync::Arc;
use std::time::Duration;

use crate::{
    Database, KeyRecord, OrganizationRecord, ProjectRecord, ScopeEntity, ScopeLinkError,
    TeamRecord, UserRecord, now_iso,
};

const LITERAL_MODELS: &str = "[\"  \",\" model \",\"gpt-*\"]";

struct Graph {
    team_a: String,
    team_b: String,
    user: String,
    org: String,
    owned: String,
    shared: String,
}

async fn seed(db: &Database, prefix: &str) -> Graph {
    let graph = Graph {
        team_a: format!("{prefix}-team-a"),
        team_b: format!("{prefix}-team-b"),
        user: format!("{prefix}-user"),
        org: format!("{prefix}-org"),
        owned: format!("{prefix}-owned"),
        shared: format!("{prefix}-shared"),
    };
    for id in [&graph.team_a, &graph.team_b] {
        db.create_team(&TeamRecord {
            team_id: id.clone(),
            team_name: id.clone(),
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
        .unwrap();
    }
    db.create_user(&UserRecord {
        user_id: graph.user.clone(),
        user_name: graph.user.clone(),
        max_budget: None,
        budget_duration: None,
        created_at: now_iso(),
        updated_at: now_iso(),
        models: None,
    })
    .await
    .unwrap();
    db.create_organization(&OrganizationRecord {
        org_id: graph.org.clone(),
        org_name: graph.org.clone(),
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now_iso(),
        updated_at: now_iso(),
        models: None,
        metadata: None,
        soft_budget: None,
        max_parallel_requests: None,
        model_max_budget: None,
    })
    .await
    .unwrap();
    for (id, team) in [
        (&graph.owned, Some(graph.team_a.as_str())),
        (&graph.shared, None),
    ] {
        db.create_project_scoped(&project(id, team, Some(&graph.org)))
            .await
            .unwrap();
    }
    graph
}

fn project(id: &str, team: Option<&str>, org: Option<&str>) -> ProjectRecord {
    ProjectRecord {
        project_id: id.to_string(),
        project_name: id.to_string(),
        org_id: org.map(str::to_string),
        team_id: team.map(str::to_string),
        max_budget: Some(50.0),
        budget_duration: Some("1d".into()),
        rpm: Some(10),
        tpm: None,
        created_at: now_iso(),
        updated_at: now_iso(),
    }
}

fn key(id: &str, graph: &Graph, team: Option<&str>, project: Option<&str>) -> KeyRecord {
    KeyRecord {
        key_id: id.to_string(),
        key_hash: format!("hash-{id}"),
        key_alias: Some("original".into()),
        team_id: team.map(str::to_string),
        user_id: Some(graph.user.clone()),
        org_id: Some(graph.org.clone()),
        project_id: project.map(str::to_string),
        max_budget: Some(5.0),
        budget_duration: Some("1d".into()),
        rpm: Some(4),
        tpm: Some(40),
        created_at: now_iso(),
        updated_at: now_iso(),
        expires_at: None,
        is_active: true,
        models: Some(LITERAL_MODELS.into()),
        allowed_cache_controls: Some("[]".into()),
    }
}

fn missing(error: &anyhow::Error, expected: ScopeEntity) {
    assert!(
        matches!(error.downcast_ref::<ScopeLinkError>(), Some(ScopeLinkError::MissingScope { entity, .. }) if *entity == expected),
        "expected missing {expected:?}, got {error:#}"
    );
}

fn mismatch(error: &anyhow::Error) {
    assert!(
        matches!(
            error.downcast_ref::<ScopeLinkError>(),
            Some(ScopeLinkError::ProjectTeamMismatch { .. })
        ),
        "expected a team/project mismatch, got {error:#}"
    );
}

async fn key_contract(db: &Database) {
    let graph = seed(db, "attach").await;
    for (index, team) in [None, Some(graph.team_b.as_str())].into_iter().enumerate() {
        let row = key(
            &format!("rejected-{index}"),
            &graph,
            team,
            Some(&graph.owned),
        );
        mismatch(&db.create_key_scoped(&row).await.unwrap_err());
        assert!(db.get_key(&row.key_id).await.unwrap().is_none());
    }
    for (index, team) in [
        None,
        Some(graph.team_a.as_str()),
        Some(graph.team_b.as_str()),
    ]
    .into_iter()
    .enumerate()
    {
        db.create_key_scoped(&key(
            &format!("shared-{index}"),
            &graph,
            team,
            Some(&graph.shared),
        ))
        .await
        .unwrap();
    }
    for entity in [
        ScopeEntity::Team,
        ScopeEntity::User,
        ScopeEntity::Organization,
        ScopeEntity::Project,
    ] {
        let mut row = key(
            "missing-parent",
            &graph,
            Some(&graph.team_a),
            Some(&graph.owned),
        );
        match entity {
            ScopeEntity::Team => row.team_id = Some("absent".into()),
            ScopeEntity::User => row.user_id = Some("absent".into()),
            ScopeEntity::Organization => row.org_id = Some("absent".into()),
            ScopeEntity::Project => row.project_id = Some("absent".into()),
        }
        missing(&db.create_key_scoped(&row).await.unwrap_err(), entity);
        assert!(db.get_key("missing-parent").await.unwrap().is_none());
    }
    let row = key("mutable", &graph, Some(&graph.team_a), Some(&graph.owned));
    db.create_key_scoped(&row).await.unwrap();
    let original = serde_json::to_value(&row).unwrap();
    for team in [None, Some(graph.team_b.clone())] {
        let error = db
            .update_key_scoped("mutable", |row| {
                row.team_id = team;
                row.key_alias = Some("must-rollback".into());
                row.max_budget = Some(0.0);
            })
            .await
            .unwrap_err();
        mismatch(&error);
        assert_eq!(
            serde_json::to_value(db.get_key("mutable").await.unwrap().unwrap()).unwrap(),
            original
        );
    }
    let error = db
        .update_key_scoped("mutable", |row| {
            row.project_id = Some("deleted-project".into())
        })
        .await
        .unwrap_err();
    missing(&error, ScopeEntity::Project);
    let updated = db
        .update_key_scoped("mutable", |row| {
            row.project_id = Some(graph.shared.clone());
            row.team_id = Some(graph.team_b.clone());
            row.key_alias = Some("changed".into());
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.models.as_deref(), Some(LITERAL_MODELS));
    assert_eq!(updated.allowed_cache_controls.as_deref(), Some("[]"));
    assert_eq!(updated.key_hash, row.key_hash);
    assert!(
        db.update_key_scoped("absent-key", |_| panic!("missing rows must not be mutated"))
            .await
            .unwrap()
            .is_none()
    );

    // Import writers intentionally retain their ability to load legacy links.
    let imported = key("imported", &graph, None, Some(&graph.owned));
    db.create_key(&imported).await.unwrap();
    assert!(db.update_key(&imported).await.unwrap());
    assert_eq!(
        db.set_key_active("imported", false, &now_iso())
            .await
            .unwrap(),
        Some(imported.key_hash)
    );
}

async fn project_contract(db: &Database) {
    let graph = seed(db, "reparent").await;
    db.create_key_scoped(&key(
        "owned-key",
        &graph,
        Some(&graph.team_a),
        Some(&graph.owned),
    ))
    .await
    .unwrap();
    db.set_key_active("owned-key", false, &now_iso())
        .await
        .unwrap();
    let original = db.get_project(&graph.owned).await.unwrap().unwrap();
    let error = db
        .update_project_scoped(&graph.owned, |row| {
            row.team_id = Some(graph.team_b.clone());
            row.max_budget = Some(0.0);
        })
        .await
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<ScopeLinkError>(),
        Some(ScopeLinkError::ConflictingProjectKeys { .. })
    ));
    assert_eq!(
        serde_json::to_value(db.get_project(&graph.owned).await.unwrap().unwrap()).unwrap(),
        serde_json::to_value(original).unwrap()
    );
    db.update_project_scoped(&graph.owned, |row| row.team_id = None)
        .await
        .unwrap();
    db.update_key_scoped("owned-key", |row| row.team_id = Some(graph.team_b.clone()))
        .await
        .unwrap();
    let owned = db
        .update_project_scoped(&graph.owned, |row| row.team_id = Some(graph.team_b.clone()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(owned.team_id.as_deref(), Some(graph.team_b.as_str()));
    db.create_key_scoped(&key("teamless-key", &graph, None, Some(&graph.shared)))
        .await
        .unwrap();
    assert!(
        db.update_project_scoped(&graph.shared, |row| row.team_id =
            Some(graph.team_a.clone()))
            .await
            .unwrap_err()
            .downcast_ref::<ScopeLinkError>()
            .is_some()
    );
    for entity in [ScopeEntity::Team, ScopeEntity::Organization] {
        let mut row = project("missing-project-parent", None, None);
        match entity {
            ScopeEntity::Team => row.team_id = Some("absent".into()),
            ScopeEntity::Organization => row.org_id = Some("absent".into()),
            _ => unreachable!(),
        }
        missing(&db.create_project_scoped(&row).await.unwrap_err(), entity);
        assert!(db.get_project(&row.project_id).await.unwrap().is_none());
    }
    assert!(
        db.update_project_scoped("deleted-project", |_| panic!("missing project mutated"))
            .await
            .unwrap()
            .is_none()
    );
}

async fn bulk_contract(db: &Database) {
    let graph = seed(db, "bulk").await;
    for (id, project) in [("bulk-owned", &graph.owned), ("bulk-shared", &graph.shared)] {
        db.create_key_scoped(&key(id, &graph, Some(&graph.team_a), Some(project)))
            .await
            .unwrap();
    }
    for (filter, target) in [
        (None, Some(graph.team_b.as_str())),
        (Some(graph.team_a.as_str()), None),
    ] {
        let error = db
            .set_team_for_user_keys_scoped(&graph.user, filter, target, "must-not-be-written")
            .await
            .unwrap_err();
        mismatch(&error);
        for id in ["bulk-owned", "bulk-shared"] {
            let row = db.get_key(id).await.unwrap().unwrap();
            assert_eq!(row.team_id.as_deref(), Some(graph.team_a.as_str()));
            assert_ne!(row.updated_at, "must-not-be-written");
        }
    }
    db.update_key_scoped("bulk-owned", |row| row.project_id = None)
        .await
        .unwrap();
    let mutation = db
        .set_team_for_user_keys_scoped(&graph.user, None, Some(&graph.team_b), &now_iso())
        .await
        .unwrap();
    assert_eq!(mutation.affected, 2);
    let mut hashes = mutation.key_hashes;
    hashes.sort();
    assert_eq!(hashes, ["hash-bulk-owned", "hash-bulk-shared"]);
    let mutation = db
        .set_team_for_user_keys_scoped(&graph.user, Some(&graph.team_b), None, &now_iso())
        .await
        .unwrap();
    assert_eq!(mutation.affected, 2);
    for id in ["bulk-owned", "bulk-shared"] {
        let row = db.get_key(id).await.unwrap().unwrap();
        assert!(row.team_id.is_none());
        assert_eq!(row.models.as_deref(), Some(LITERAL_MODELS));
    }
    missing(
        &db.set_team_for_user_keys_scoped("deleted-user", None, Some(&graph.team_a), &now_iso())
            .await
            .unwrap_err(),
        ScopeEntity::User,
    );
    missing(
        &db.set_team_for_user_keys_scoped(&graph.user, Some("deleted-team"), None, &now_iso())
            .await
            .unwrap_err(),
        ScopeEntity::Team,
    );
    missing(
        &db.set_team_for_user_keys_scoped(&graph.user, None, Some("deleted-team"), &now_iso())
            .await
            .unwrap_err(),
        ScopeEntity::Team,
    );
}

async fn deletion_contract(db: &Database) {
    let graph = seed(db, "delete").await;
    let row = key(
        "deletion-key",
        &graph,
        Some(&graph.team_a),
        Some(&graph.owned),
    );
    db.create_key_scoped(&row).await.unwrap();
    let removed = db.delete_team_scoped(&graph.team_a).await.unwrap();
    assert!(removed.deleted);
    assert_eq!(removed.key_hashes, std::slice::from_ref(&row.key_hash));
    assert_eq!(removed.project_ids, std::slice::from_ref(&graph.owned));
    assert!(
        db.get_key(&row.key_id)
            .await
            .unwrap()
            .unwrap()
            .team_id
            .is_none()
    );
    assert!(
        db.get_project(&graph.owned)
            .await
            .unwrap()
            .unwrap()
            .team_id
            .is_none()
    );
    missing(
        &db.create_key_scoped(&row).await.unwrap_err(),
        ScopeEntity::Team,
    );
    db.delete_project_and_detach(&graph.owned).await.unwrap();
    db.delete_user_and_detach(&graph.user).await.unwrap();
    db.delete_organization_and_detach(&graph.org).await.unwrap();
    let updated = db
        .update_key_scoped(&row.key_id, |row| {
            row.key_alias = Some("after-deletes".into())
        })
        .await
        .unwrap()
        .unwrap();
    assert!(updated.project_id.is_none() && updated.user_id.is_none() && updated.org_id.is_none());
    assert_eq!(updated.models.as_deref(), Some(LITERAL_MODELS));
    let before_purge = updated.key_hash;
    assert_eq!(
        db.rotate_key(&row.key_id, "rotated", &now_iso())
            .await
            .unwrap(),
        Some(before_purge)
    );
    assert_eq!(
        db.purge_key_scoped(&row.key_id).await.unwrap(),
        Some("rotated".into())
    );
    assert!(
        db.update_key_scoped(&row.key_id, |_| {})
            .await
            .unwrap()
            .is_none()
    );

    let foreign = seed(db, "foreign-delete").await;
    db.create_key(&key(
        "legacy-foreign",
        &foreign,
        Some(&foreign.team_b),
        Some(&foreign.owned),
    ))
    .await
    .unwrap();
    mismatch(&db.delete_team_scoped(&foreign.team_b).await.unwrap_err());
    assert!(db.get_team(&foreign.team_b).await.unwrap().is_some());
    assert_eq!(
        db.get_key("legacy-foreign").await.unwrap().unwrap().team_id,
        Some(foreign.team_b)
    );
}

async fn sqlite() -> Arc<Database> {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    db.migrate().await.unwrap();
    db
}

#[tokio::test]
async fn scope_links_sqlite_key_attachments_and_updates() {
    key_contract(sqlite().await.as_ref()).await;
}

#[tokio::test]
async fn scope_links_sqlite_project_reparenting() {
    project_contract(sqlite().await.as_ref()).await;
}

#[tokio::test]
async fn scope_links_sqlite_bulk_moves_are_all_or_nothing() {
    bulk_contract(sqlite().await.as_ref()).await;
}

#[tokio::test]
async fn scope_links_sqlite_scope_deletes_detach_without_stale_restoration() {
    deletion_contract(sqlite().await.as_ref()).await;
}

async fn race_contract(db: Arc<Database>, peer: Arc<Database>) {
    for index in 0..8 {
        let graph = seed(&db, &format!("race-{index}")).await;
        let row = key(
            &format!("racing-key-{index}"),
            &graph,
            Some(&graph.team_a),
            Some(&graph.shared),
        );
        let (attached, reparented) = tokio::join!(
            db.create_key_scoped(&row),
            peer.update_project_scoped(&graph.shared, |row| row.team_id =
                Some(graph.team_b.clone()))
        );
        assert_ne!(
            attached.is_ok(),
            reparented.is_ok(),
            "exactly one conflicting mutation can commit"
        );
        if let Err(error) = attached {
            mismatch(&error);
        }
        if let Err(error) = reparented {
            assert!(matches!(
                error.downcast_ref::<ScopeLinkError>(),
                Some(ScopeLinkError::ConflictingProjectKeys { .. })
            ));
        }

        let move_id = format!("move-key-{index}");
        db.create_key_scoped(&key(&move_id, &graph, Some(&graph.team_a), None))
            .await
            .unwrap();
        db.update_project_scoped(&graph.shared, |row| row.team_id = None)
            .await
            .unwrap();
        db.update_key_scoped(&move_id, |row| row.project_id = Some(graph.shared.clone()))
            .await
            .unwrap();
        // Remove the other possible attachment so only this user's bulk move
        // and the reparent contest the same ownerless project.
        db.purge_key(&format!("racing-key-{index}")).await.unwrap();
        let now = now_iso();
        let (moved, reparented) = tokio::join!(
            db.set_team_for_user_keys_scoped(&graph.user, None, Some(&graph.team_b), &now),
            peer.update_project_scoped(&graph.shared, |row| row.team_id =
                Some(graph.team_a.clone()))
        );
        assert_ne!(moved.is_ok(), reparented.is_ok());
        if let Err(error) = moved {
            mismatch(&error);
        }
        if let Err(error) = reparented {
            assert!(matches!(
                error.downcast_ref::<ScopeLinkError>(),
                Some(ScopeLinkError::ConflictingProjectKeys { .. })
            ));
        }
        let current_key = db.get_key(&move_id).await.unwrap().unwrap();
        let current_project = db.get_project(&graph.shared).await.unwrap().unwrap();
        assert!(
            current_project.team_id.is_none() || current_project.team_id == current_key.team_id
        );

        db.update_project_scoped(&graph.shared, |row| row.team_id = None)
            .await
            .unwrap();
        db.update_key_scoped(&move_id, |row| row.team_id = Some(graph.team_a.clone()))
            .await
            .unwrap();
        let (changed, reparented) = tokio::join!(
            db.update_key_scoped(&move_id, |row| row.team_id = Some(graph.team_b.clone())),
            peer.update_project_scoped(&graph.shared, |row| {
                row.team_id = Some(graph.team_a.clone())
            })
        );
        assert_ne!(changed.is_ok(), reparented.is_ok());
        if let Err(error) = changed {
            mismatch(&error);
        }
        if let Err(error) = reparented {
            assert!(matches!(
                error.downcast_ref::<ScopeLinkError>(),
                Some(ScopeLinkError::ConflictingProjectKeys { .. })
            ));
        }

        let (deleted, edited) = tokio::join!(
            db.delete_project_and_detach(&graph.shared),
            peer.update_key_scoped(&move_id, |row| row.key_alias = Some("after-race".into()))
        );
        assert!(deleted.unwrap().deleted);
        edited.unwrap().unwrap();
        assert!(
            db.get_key(&move_id)
                .await
                .unwrap()
                .unwrap()
                .project_id
                .is_none()
        );

        let creation = key(
            &format!("delete-race-{index}"),
            &graph,
            Some(&graph.team_a),
            Some(&graph.owned),
        );
        let (attached, removed) = tokio::join!(
            db.create_key_scoped(&creation),
            peer.delete_team_scoped(&graph.team_a)
        );
        assert!(removed.unwrap().deleted);
        if let Err(error) = attached {
            missing(&error, ScopeEntity::Team);
        }
        if let Some(row) = db.get_key(&creation.key_id).await.unwrap() {
            assert!(row.team_id.is_none());
            assert!(
                db.get_project(&graph.owned)
                    .await
                    .unwrap()
                    .unwrap()
                    .team_id
                    .is_none()
            );
        }

        let creation = key(
            &format!("user-delete-race-{index}"),
            &graph,
            Some(&graph.team_b),
            None,
        );
        let (attached, removed) = tokio::join!(
            db.create_key_scoped(&creation),
            peer.delete_user_and_detach(&graph.user)
        );
        assert!(removed.unwrap().deleted);
        if let Err(error) = attached {
            missing(&error, ScopeEntity::User);
        }
        if let Some(row) = db.get_key(&creation.key_id).await.unwrap() {
            assert!(row.user_id.is_none());
        }
    }
}

async fn locked_snapshot_contract(db: Arc<Database>, peer: Arc<Database>) {
    let graph = seed(&db, "snapshot").await;
    db.create_key_scoped(&key(
        "snapshot-key",
        &graph,
        Some(&graph.team_a),
        Some(&graph.owned),
    ))
    .await
    .unwrap();
    let mut holder = db.begin_scope_links().await.unwrap();
    let q = format!(
        "UPDATE keys SET project_id = NULL WHERE project_id = {}",
        db.ph(1)
    );
    sqlx::query(&q)
        .bind(&graph.owned)
        .execute(&mut *holder)
        .await
        .unwrap();
    let q = format!("DELETE FROM projects WHERE project_id = {}", db.ph(1));
    sqlx::query(&q)
        .bind(&graph.owned)
        .execute(&mut *holder)
        .await
        .unwrap();
    let waiting_db = peer.clone();
    let mut waiting = tokio::spawn(async move {
        waiting_db
            .update_key_scoped("snapshot-key", |row| {
                row.key_alias = Some("fresh-snapshot".into())
            })
            .await
    });
    assert!(
        tokio::time::timeout(Duration::from_millis(80), &mut waiting)
            .await
            .is_err()
    );
    holder.commit().await.unwrap();
    let row = tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        row.project_id.is_none(),
        "a waiting metadata update must not restore the deleted scope"
    );
    assert_eq!(row.models.as_deref(), Some(LITERAL_MODELS));

    let holder = db.begin_scope_links().await.unwrap();
    let waiting_db = peer.clone();
    let row = key("canceled-key", &graph, Some(&graph.team_a), None);
    let canceled_row = row.clone();
    let mut canceled =
        tokio::spawn(async move { waiting_db.create_key_scoped(&canceled_row).await });
    assert!(
        tokio::time::timeout(Duration::from_millis(80), &mut canceled)
            .await
            .is_err()
    );
    canceled.abort();
    assert!(canceled.await.unwrap_err().is_cancelled());
    holder.rollback().await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), peer.create_key_scoped(&row))
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scope_links_sqlite_independent_pools_serialize_races_and_cancel_safely() {
    let directory = tempfile::tempdir().unwrap();
    let url = format!(
        "sqlite://{}?mode=rwc",
        directory.path().join("scope-links.sqlite").display()
    );
    let db = Database::connect(&url).await.unwrap();
    db.migrate().await.unwrap();
    let peer = Database::connect(&url).await.unwrap();
    tokio::time::timeout(Duration::from_secs(20), async {
        race_contract(db.clone(), peer.clone()).await;
        locked_snapshot_contract(db.clone(), peer.clone()).await;
    })
    .await
    .unwrap();
    db.pool().close().await;
    peer.pool().close().await;
}

#[cfg(feature = "postgres")]
async fn postgres_fixture() -> Option<(crate::release_tests::PostgresFixture, String)> {
    let url = match std::env::var("OXLLM_TEST_POSTGRES_URL") {
        Ok(url) if !url.trim().is_empty() => url,
        _ => {
            eprintln!("skipping PostgreSQL scope-link regression: OXLLM_TEST_POSTGRES_URL not set");
            return None;
        }
    };
    let fixture = crate::release_tests::PostgresFixture::new(&url, 4).await;
    Some((fixture, url))
}

#[cfg(feature = "postgres")]
#[tokio::test]
async fn scope_links_postgres_native_mutations() {
    let Some((fixture, _)) = postgres_fixture().await else {
        return;
    };
    let db = fixture.db.clone();
    let result = tokio::spawn(async move {
        db.migrate().await.unwrap();
        key_contract(&db).await;
        project_contract(&db).await;
        bulk_contract(&db).await;
        deletion_contract(&db).await;
    })
    .await;
    fixture.cleanup().await;
    result.unwrap();
}

#[cfg(feature = "postgres")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scope_links_postgres_independent_pools_refresh_waiting_snapshots() {
    let Some((fixture, url)) = postgres_fixture().await else {
        return;
    };
    let schema: (String,) = sqlx::query_as("SELECT CAST(current_schema() AS TEXT)")
        .fetch_one(fixture.db.pool())
        .await
        .unwrap();
    let pool = sqlx::pool::PoolOptions::<sqlx::Any>::new()
        .max_connections(4)
        .after_connect(move |conn, _| {
            let q = format!("SET search_path TO {}", schema.0);
            Box::pin(async move {
                sqlx::query(&q).execute(&mut *conn).await?;
                sqlx::query("SET default_transaction_isolation = 'repeatable read'")
                    .execute(&mut *conn)
                    .await?;
                Ok(())
            })
        })
        .connect(&url)
        .await
        .unwrap();
    let peer = Arc::new(Database::new(pool, crate::PlaceholderStyle::Dollar));
    let db = fixture.db.clone();
    let contender = peer.clone();
    let result = tokio::spawn(async move {
        db.migrate().await.unwrap();
        tokio::time::timeout(Duration::from_secs(25), async {
            race_contract(db.clone(), contender.clone()).await;
            locked_snapshot_contract(db.clone(), contender.clone()).await;
        })
        .await
        .unwrap();
    })
    .await;
    peer.pool().close().await;
    fixture.cleanup().await;
    result.unwrap();
}
