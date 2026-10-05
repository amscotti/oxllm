//! `skills` table CRUD — versioned instruction files (SKILL.md content)
//! that extend a model's capabilities.

use anyhow::{Result, anyhow};
use sqlx::Row;

use crate::Database;
use crate::rows::row_to_skill;
use crate::types::SkillRecord;

/// How long a [`Database::list_skills_by_model`] result stays cached before
/// the next call re-queries the database. The lookup runs on every chat
/// request when a DB is configured; without the cache each request pays a
/// full-table scan of the (slim `id` + `model_pattern`) skills projection
/// plus a DB round-trip, growing with the skills table. In-process writes
/// (`create`/`update`/`delete_skill`) invalidate immediately, so the TTL
/// only bounds staleness from writes this process did not observe (e.g. a
/// second proxy replica, or direct SQL).
pub(crate) const SKILLS_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(60);

/// Upper bound on distinct cached model names. The `model` string is
/// client-supplied and only presence-checked before the skills lookup, so
/// adversarial traffic could otherwise grow the cache without limit. When
/// the cap is hit, the whole map is dropped (entries for the actually-hot
/// models repopulate on their next request).
pub(crate) const SKILLS_CACHE_MAX_MODELS: usize = 256;

/// TTL cache for [`Database::list_skills_by_model`], keyed by model name.
/// Interior-mutable so the (single) `Database` handle can serve concurrent
/// chat requests without re-querying within the TTL window.
pub(crate) struct SkillsCache {
    entries: std::sync::Mutex<std::collections::HashMap<String, CacheEntry>>,
}

struct CacheEntry {
    /// When this entry stops being served. `Instant` (monotonic) so wall
    /// clock jumps cannot resurrect expired entries.
    expires_at: std::time::Instant,
    records: Vec<SkillRecord>,
}

impl SkillsCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    fn lookup(&self, model: &str) -> Option<Vec<SkillRecord>> {
        let mut entries = self.entries.lock().expect("skills cache lock");
        let now = std::time::Instant::now();
        match entries.get(model) {
            Some(entry) if now < entry.expires_at => Some(entry.records.clone()),
            // Drop the stale copy while we hold the lock so it cannot be
            // served again and does not linger until the next write.
            Some(_) => {
                entries.remove(model);
                None
            }
            None => None,
        }
    }

    fn store(&self, model: &str, records: Vec<SkillRecord>) {
        let mut entries = self.entries.lock().expect("skills cache lock");
        if !entries.contains_key(model) && entries.len() >= SKILLS_CACHE_MAX_MODELS {
            entries.clear();
        }
        entries.insert(
            model.to_string(),
            CacheEntry {
                expires_at: std::time::Instant::now() + SKILLS_CACHE_TTL,
                records,
            },
        );
    }

    /// Drop every entry. Called by the skills write paths so this process's
    /// reads never observe its own just-written state stale.
    pub(crate) fn invalidate(&self) {
        self.entries.lock().expect("skills cache lock").clear();
    }

    /// Test hook: expire every entry so the next lookup misses. Uses
    /// `checked_sub` because a hypothetical platform whose monotonic clock
    /// cannot look `1s` into the past cannot express "already expired".
    #[cfg(test)]
    pub(crate) fn expire_all_for_test(&self) {
        let past = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("monotonic clock supports 1s lookback");
        let mut entries = self.entries.lock().expect("skills cache lock");
        for entry in entries.values_mut() {
            entry.expires_at = past;
        }
    }
}

impl Database {
    pub async fn create_skill(&self, skill: &SkillRecord) -> Result<SkillRecord> {
        let q = format!(
            "INSERT INTO skills (id, name, description, content, version, model_pattern, \
             team_id, created_at, updated_at) VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {})",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6),
            self.ph(7),
            self.ph(8),
            self.ph(9)
        );
        sqlx::query(&q)
            .bind(&skill.id)
            .bind(&skill.name)
            .bind(skill.description.clone())
            .bind(&skill.content)
            .bind(&skill.version)
            .bind(skill.model_pattern.clone())
            .bind(skill.team_id.clone())
            .bind(&skill.created_at)
            .bind(&skill.updated_at)
            .execute(self.pool())
            .await?;
        // In-process skill writes must be visible to the next
        // `list_skills_by_model` (the chat hot path reads through a TTL
        // cache), so drop cached lookups alongside the insert.
        self.skills_cache.invalidate();
        // Re-read the row rather than echoing back the input clone, so
        // the returned record matches what [`Self::get_skill`] /
        // [`Self::update_skill`] would return for the same id. (The
        // `version` column's `DEFAULT '1.0'` never applies here — the
        // INSERT binds `version` explicitly.)
        self.get_skill(&skill.id)
            .await?
            .ok_or_else(|| anyhow!("skill '{}' missing after insert", skill.id))
    }

    pub async fn get_skill(&self, id: &str) -> Result<Option<SkillRecord>> {
        let q = format!(
            "SELECT id, name, description, content, version, model_pattern, team_id, created_at, \
             updated_at FROM skills WHERE id = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q).bind(id).fetch_optional(self.pool()).await?;
        row.map(|r| row_to_skill(&r)).transpose()
    }

    /// List skills newest-first, capped at `limit` rows (default 1 000
    /// when `None`). The upper clamp is 1 001 so a handler can request one
    /// extra row as a sentinel to detect truncation without the result
    /// being silently clamped back down. See [`crate::Database::list_keys`]
    /// for why the cap lives in the repo.
    pub async fn list_skills(&self, limit: Option<i64>) -> Result<Vec<SkillRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT id, name, description, content, version, model_pattern, team_id, \
             created_at, updated_at FROM skills ORDER BY created_at DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_skill).collect()
    }

    /// Skills whose `model_pattern` matches `model`. A skill with no
    /// `model_pattern` applies to every model (unrestricted). Otherwise the
    /// pattern is a `*`-wildcard glob (e.g. `"claude-*"`). Pattern matching
    /// runs in Rust (not SQL) so the query stays portable across the
    /// Postgres and SQLite backends.
    ///
    /// Loads every skill row (no `LIMIT`) before filtering: the cap in
    /// [`Self::list_skills`] exists to keep handler-facing list endpoints
    /// from materialising a huge table, but here the in-Rust glob filter
    /// immediately reduces the set, so capping the candidate pool would
    /// silently drop any skill past row 1 000 (ordered `created_at DESC`)
    /// — newer skills would shadow older model-scoped skills and never
    /// match.
    ///
    /// Hot-path cache: this method runs on every chat request when a DB is
    /// configured, so results are cached per model for
    /// [`SKILLS_CACHE_TTL`] (see [`SkillsCache`]). The skills write methods
    /// invalidate the cache, so CRUD through this handle is visible
    /// immediately; only writes from other processes can be stale for up
    /// to the TTL.
    pub async fn list_skills_by_model(&self, model: &str) -> Result<Vec<SkillRecord>> {
        if let Some(cached) = self.skills_cache.lookup(model) {
            return Ok(cached);
        }
        let records = self.list_skills_by_model_uncached(model).await?;
        self.skills_cache.store(model, records.clone());
        Ok(records)
    }

    /// The uncached DB lookup backing [`Self::list_skills_by_model`].
    ///
    /// Hot routing path: run the glob filter over a slim candidate query
    /// (`id` + `model_pattern` only) instead of materialising every row's
    /// large SKILL.md `content`, then fetch full rows — content included
    /// — just for the matches. Unbounded: see the cap note on
    /// [`Self::list_skills_by_model`].
    async fn list_skills_by_model_uncached(&self, model: &str) -> Result<Vec<SkillRecord>> {
        let q = "SELECT id, model_pattern FROM skills";
        let rows = sqlx::query(q).fetch_all(self.pool()).await?;
        let mut matching: Vec<String> = Vec::new();
        for row in &rows {
            let id: String = row
                .try_get("id")
                .map_err(|e| anyhow!("failed to read skills.id: {e}"))?;
            let pattern: Option<String> = row
                .try_get("model_pattern")
                .map_err(|e| anyhow!("failed to read skills.model_pattern: {e}"))?;
            let hit = match pattern.as_deref() {
                None => true,
                Some(pattern) => glob_match(pattern, model),
            };
            if hit {
                matching.push(id);
            }
        }
        if matching.is_empty() {
            return Ok(Vec::new());
        }
        // Keep each `IN (...)` list under the SQLite 999-bind ceiling.
        let mut out: Vec<SkillRecord> = Vec::new();
        for chunk in matching.chunks(500) {
            let placeholders: String = (0..chunk.len())
                .map(|i| self.ph(i + 1))
                .collect::<Vec<_>>()
                .join(", ");
            let q = format!(
                "SELECT id, name, description, content, version, model_pattern, team_id, \
                 created_at, updated_at FROM skills WHERE id IN ({placeholders})"
            );
            let mut query = sqlx::query(&q);
            for id in chunk {
                query = query.bind(id.as_str());
            }
            let rows = query.fetch_all(self.pool()).await?;
            out.extend(rows.iter().map(row_to_skill).collect::<Result<Vec<_>>>()?);
        }
        // Newest-first, matching the pre-filter ordering.
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(b.id.cmp(&a.id)));
        Ok(out)
    }

    pub async fn list_skills_by_team(&self, team_id: &str) -> Result<Vec<SkillRecord>> {
        let q = format!(
            "SELECT id, name, description, content, version, model_pattern, team_id, created_at, \
             updated_at FROM skills WHERE team_id = {} ORDER BY created_at DESC",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(team_id).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_skill).collect()
    }

    /// Overwrite the mutable fields of the skill identified by `id`. Returns
    /// the refreshed record, or `None` when no row matched `id`.
    ///
    /// The `None` case means "no row present *after* the update": between
    /// the `UPDATE` and the re-`SELECT` a concurrent `delete_skill` can
    /// remove the row, so a successful update can still yield `Ok(None)`.
    /// Callers that interpret `None` as "skill didn't exist, no change"
    /// should be aware of this window.
    pub async fn update_skill(&self, id: &str, skill: &SkillRecord) -> Result<Option<SkillRecord>> {
        let q = format!(
            "UPDATE skills SET name = {}, description = {}, content = {}, version = {}, \
             model_pattern = {}, team_id = {}, updated_at = {} WHERE id = {}",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6),
            self.ph(7),
            self.ph(8)
        );
        let res = sqlx::query(&q)
            .bind(&skill.name)
            .bind(skill.description.clone())
            .bind(&skill.content)
            .bind(&skill.version)
            .bind(skill.model_pattern.clone())
            .bind(skill.team_id.clone())
            .bind(&skill.updated_at)
            .bind(id)
            .execute(self.pool())
            .await?;
        if res.rows_affected() > 0 {
            // Invalidate the `list_skills_by_model` TTL cache so this
            // handle's next chat-request lookup sees the new content.
            self.skills_cache.invalidate();
        }
        if res.rows_affected() == 0 {
            return Ok(None);
        }
        self.get_skill(id).await
    }

    /// Delete the skill identified by `id`. Returns `true` when a row was
    /// removed, `false` when no such skill existed.
    pub async fn delete_skill(&self, id: &str) -> Result<bool> {
        let q = format!("DELETE FROM skills WHERE id = {}", self.ph(1));
        let res = sqlx::query(&q).bind(id).execute(self.pool()).await?;
        // Invalidate the `list_skills_by_model` TTL cache so deleted skills
        // stop being injected on the next chat request.
        if res.rows_affected() > 0 {
            self.skills_cache.invalidate();
        }
        Ok(res.rows_affected() > 0)
    }
}

/// Simple glob matcher supporting `*` wildcards. A `*` matches any sequence
/// (including the empty sequence); a pattern without `*` requires an exact
/// match. Mirrors the wildcard semantics used by the server's model-ACL
/// matcher so `model_pattern` behaves consistently with model allow-lists.
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == text {
        return true;
    }
    if !pattern.contains('*') {
        return false;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !text[pos..].starts_with(part) {
                return false;
            }
            pos += part.len();
        } else if i == parts.len() - 1 {
            return text[pos..].ends_with(part);
        } else {
            match text[pos..].find(part) {
                Some(idx) => pos += idx + part.len(),
                None => return false,
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::glob_match;

    #[test]
    fn glob_match_exact() {
        assert!(glob_match("claude-3", "claude-3"));
        assert!(!glob_match("claude-3", "claude-4"));
    }

    #[test]
    fn glob_match_wildcard() {
        assert!(glob_match("claude-*", "claude-3-opus"));
        assert!(glob_match("claude-*", "claude-"));
        assert!(!glob_match("claude-*", "gpt-4o"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("pre-*-post", "pre-mid-post"));
    }

    /// C57 coverage gap: multi-wildcard patterns, unicode model names, and
    /// patterns ending mid-character.
    #[test]
    fn glob_match_multi_wildcard_and_unicode() {
        // Three wildcards: `*a*b*c*` matches any string containing a, b, c
        // in order (each possibly with surrounding text).
        assert!(glob_match("*a*b*c*", "xxayybzzcww"));
        assert!(glob_match("*a*b*c*", "abc"));
        assert!(!glob_match("*a*b*c*", "acb"));
        // Unicode model names round-trip byte-wise.
        assert!(glob_match("模型-*", "模型-大"));
        assert!(!glob_match("模型-*", "other"));
        // Pattern ending mid-character: suffix must match.
        assert!(glob_match("gpt-*o", "gpt-4o"));
        assert!(!glob_match("gpt-*o", "gpt-4o-mini"));
    }
}
