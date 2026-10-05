//! Atomic scope deletion and referential cleanup. Deleting a scope detaches
//! its members; it does not revoke otherwise-active keys or delete projects.

use anyhow::Result;
use sqlx::Row;

use crate::{Database, ScopeDeletion};

pub(super) enum Scope {
    Organization,
    Project,
    Team,
    User,
}

impl Database {
    pub(super) async fn delete_scope(&self, scope: Scope, id: &str) -> Result<ScopeDeletion> {
        let (table, column, detach_projects) = match scope {
            Scope::Organization => ("organizations", "org_id", true),
            Scope::Project => ("projects", "project_id", false),
            Scope::Team => ("teams", "team_id", true),
            Scope::User => ("users", "user_id", false),
        };
        let mut tx = self.pool().begin().await?;
        // Identifiers are selected exclusively by Scope, never client input.
        let q = format!("DELETE FROM {table} WHERE {column} = {}", self.ph(1));
        let deleted = sqlx::query(&q)
            .bind(id)
            .execute(&mut *tx)
            .await?
            .rows_affected()
            > 0;
        if !deleted {
            tx.rollback().await?;
            return Ok(ScopeDeletion::default());
        }

        let now = crate::now_iso();
        let q = format!(
            "UPDATE keys SET {column} = NULL, updated_at = {} \
             WHERE {column} = {} RETURNING key_hash",
            self.ph(1),
            self.ph(2)
        );
        let rows = sqlx::query(&q)
            .bind(&now)
            .bind(id)
            .fetch_all(&mut *tx)
            .await?;
        let key_hashes = rows
            .iter()
            .map(|row| row.try_get("key_hash"))
            .collect::<Result<Vec<String>, _>>()?;

        let mut project_ids = Vec::new();
        if detach_projects {
            let q = format!(
                "UPDATE projects SET {column} = NULL, updated_at = {} \
                 WHERE {column} = {} RETURNING project_id",
                self.ph(1),
                self.ph(2)
            );
            let rows = sqlx::query(&q)
                .bind(&now)
                .bind(id)
                .fetch_all(&mut *tx)
                .await?;
            project_ids = rows
                .iter()
                .map(|row| row.try_get("project_id"))
                .collect::<Result<Vec<String>, _>>()?;
        }
        tx.commit().await?;
        Ok(ScopeDeletion {
            deleted,
            key_hashes,
            project_ids,
        })
    }
}
