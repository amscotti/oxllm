//! Transaction-scoped serialization and validation of native scope mutations.
//!
//! Management writers acquire the graph lock before reading any scope or key.
//! The same lock covers scope deletion, so validation cannot race an attachment,
//! project reparent, bulk reassignment, or removal of a referenced scope.

use anyhow::Result;
use sqlx::{Any, AnyConnection, Row, Transaction};

use crate::{Database, KeyRecord, PlaceholderStyle, ProjectRecord, ScopeEntity, ScopeLinkError};

/// Shared across processes and distinct from the schema-preparation lock.
pub(crate) const SCOPE_LINKS_LOCK_KEY: i64 = 0x004f_5853_434f_5045;

impl Database {
    pub(crate) async fn begin_scope_links(&self) -> Result<Transaction<'_, Any>> {
        match self.placeholder_style() {
            PlaceholderStyle::Dollar => {
                let mut tx = self.pool().begin().await?;
                // A pool can inherit REPEATABLE READ from the database. The
                // blocking advisory SELECT must not freeze an old snapshot:
                // ownership reads after it need to observe the previous commit.
                sqlx::query("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    .execute(&mut *tx)
                    .await?;
                sqlx::query("SELECT pg_advisory_xact_lock($1)")
                    .bind(SCOPE_LINKS_LOCK_KEY)
                    .execute(&mut *tx)
                    .await?;
                Ok(tx)
            }
            // Reserve the writer before taking a read snapshot. This works
            // across independent pools/processes, including SQLite WAL mode.
            PlaceholderStyle::Question => Ok(self.pool().begin_with("BEGIN IMMEDIATE").await?),
        }
    }

    pub(crate) async fn require_scope_on(
        &self,
        conn: &mut AnyConnection,
        entity: ScopeEntity,
        id: &str,
    ) -> Result<()> {
        let table = match entity {
            ScopeEntity::Team => "teams",
            ScopeEntity::User => "users",
            ScopeEntity::Organization => "organizations",
            ScopeEntity::Project => "projects",
        };
        let q = format!(
            "SELECT {} FROM {table} WHERE {} = {}",
            entity.field_name(),
            entity.field_name(),
            self.ph(1)
        );
        if sqlx::query(&q)
            .bind(id)
            .fetch_optional(conn)
            .await?
            .is_none()
        {
            return Err(ScopeLinkError::MissingScope {
                entity,
                id: id.to_string(),
            }
            .into());
        }
        Ok(())
    }

    pub(crate) async fn validate_key_scope_links_on(
        &self,
        conn: &mut AnyConnection,
        key: &KeyRecord,
    ) -> Result<()> {
        for (entity, id) in [
            (ScopeEntity::Team, key.team_id.as_deref()),
            (ScopeEntity::User, key.user_id.as_deref()),
            (ScopeEntity::Organization, key.org_id.as_deref()),
        ] {
            if let Some(id) = id {
                self.require_scope_on(&mut *conn, entity, id).await?;
            }
        }
        if let Some(project_id) = key.project_id.as_deref() {
            let q = format!(
                "SELECT team_id FROM projects WHERE project_id = {}",
                self.ph(1)
            );
            let row = sqlx::query(&q)
                .bind(project_id)
                .fetch_optional(&mut *conn)
                .await?
                .ok_or_else(|| ScopeLinkError::MissingScope {
                    entity: ScopeEntity::Project,
                    id: project_id.to_string(),
                })?;
            let owner: Option<String> = row.try_get("team_id")?;
            if let Some(owner) = owner
                && key.team_id.as_deref() != Some(owner.as_str())
            {
                return Err(ScopeLinkError::ProjectTeamMismatch {
                    project_id: project_id.to_string(),
                    project_team_id: owner,
                    key_team_id: key.team_id.clone(),
                }
                .into());
            }
        }
        Ok(())
    }

    pub(crate) async fn validate_project_scope_links_on(
        &self,
        conn: &mut AnyConnection,
        project: &ProjectRecord,
    ) -> Result<()> {
        for (entity, id) in [
            (ScopeEntity::Organization, project.org_id.as_deref()),
            (ScopeEntity::Team, project.team_id.as_deref()),
        ] {
            if let Some(id) = id {
                self.require_scope_on(&mut *conn, entity, id).await?;
            }
        }
        if let Some(team_id) = project.team_id.as_deref() {
            let q = format!(
                "SELECT key_id FROM keys WHERE project_id = {} \
                 AND (team_id IS NULL OR team_id <> {}) LIMIT 1",
                self.ph(1),
                self.ph(2)
            );
            if sqlx::query(&q)
                .bind(&project.project_id)
                .bind(team_id)
                .fetch_optional(&mut *conn)
                .await?
                .is_some()
            {
                return Err(ScopeLinkError::ConflictingProjectKeys {
                    project_id: project.project_id.clone(),
                    team_id: team_id.to_string(),
                }
                .into());
            }
        }
        Ok(())
    }

    pub(crate) async fn validate_user_key_team_links_on(
        &self,
        conn: &mut AnyConnection,
        user_id: &str,
        filter_team: Option<&str>,
        new_team: Option<&str>,
    ) -> Result<()> {
        // Member endpoints retain their team-before-user missing-entity order,
        // including a missing team on an otherwise empty bulk operation.
        for team_id in new_team.into_iter().chain(filter_team) {
            self.require_scope_on(&mut *conn, ScopeEntity::Team, team_id)
                .await?;
        }
        self.require_scope_on(&mut *conn, ScopeEntity::User, user_id)
            .await?;
        let filter = filter_team
            .map(|_| format!(" AND k.team_id = {}", self.ph(2)))
            .unwrap_or_default();
        let target_index = if filter_team.is_some() { 3 } else { 2 };
        let mismatch = new_team
            .map(|_| format!("p.team_id <> {}", self.ph(target_index)))
            .unwrap_or_else(|| "1 = 1".to_string());
        let q = format!(
            "SELECT k.project_id, p.project_id AS existing_project_id, p.team_id \
             FROM keys k LEFT JOIN projects p ON p.project_id = k.project_id \
             WHERE k.user_id = {}{filter} AND k.project_id IS NOT NULL \
             AND (p.project_id IS NULL OR (p.team_id IS NOT NULL AND {mismatch})) LIMIT 1",
            self.ph(1)
        );
        let mut query = sqlx::query(&q).bind(user_id);
        if let Some(team_id) = filter_team {
            query = query.bind(team_id);
        }
        if let Some(team_id) = new_team {
            query = query.bind(team_id);
        }
        if let Some(row) = query.fetch_optional(&mut *conn).await? {
            return Err(project_link_error(&row, new_team)?);
        }
        Ok(())
    }

    pub(crate) async fn validate_deleted_team_links_on(
        &self,
        conn: &mut AnyConnection,
        team_id: &str,
    ) -> Result<()> {
        // Projects owned by the deleted team become ownerless in the same
        // commit. Only a foreign owner (or an imported dangling project link)
        // would conflict with the resulting NULL team on an affected key.
        let q = format!(
            "SELECT k.project_id, p.project_id AS existing_project_id, p.team_id \
             FROM keys k LEFT JOIN projects p ON p.project_id = k.project_id \
             WHERE k.team_id = {} AND k.project_id IS NOT NULL \
             AND (p.project_id IS NULL OR (p.team_id IS NOT NULL AND p.team_id <> {})) LIMIT 1",
            self.ph(1),
            self.ph(2)
        );
        if let Some(row) = sqlx::query(&q)
            .bind(team_id)
            .bind(team_id)
            .fetch_optional(conn)
            .await?
        {
            return Err(project_link_error(&row, None)?);
        }
        Ok(())
    }
}

fn project_link_error(row: &sqlx::any::AnyRow, key_team_id: Option<&str>) -> Result<anyhow::Error> {
    let project_id: String = row.try_get("project_id")?;
    let existing: Option<String> = row.try_get("existing_project_id")?;
    Ok(if existing.is_none() {
        ScopeLinkError::MissingScope {
            entity: ScopeEntity::Project,
            id: project_id,
        }
        .into()
    } else {
        ScopeLinkError::ProjectTeamMismatch {
            project_id,
            project_team_id: row.try_get("team_id")?,
            key_team_id: key_team_id.map(str::to_string),
        }
        .into()
    })
}
