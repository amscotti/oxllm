//! Rolling-upgrade reads and bounded, resumable spend metadata maintenance.
//!
//! DDL readiness does not depend on finishing a historical scan. Unrepaired
//! rows use exact JSON membership and nanosecond-preserving UTC read views;
//! old writers leave those rows pending even after an earlier sweep finished.

use anyhow::Result;
use sqlx::Row;

use crate::types::spend_tag_names;
use crate::{Database, PlaceholderStyle, SPEND_METADATA_REPAIR_BATCH_SIZE, SpendMetadataRepair};

/// Public spend columns in SELECT order, shared with the rolling-upgrade view.
const LEGACY_SPEND_LOG_SELECT_COLS: &str = "id, timestamp, request_id, model, provider, \
 prompt_tokens, completion_tokens, total_tokens, cost, key_id, team_id, user_id, \
 end_user_id, tag_name, tags, status, latency_ms";

const DEPLOYMENT_SPEND_LOG_SELECT_COLS: &str = "id, timestamp, request_id, model, provider, \
 prompt_tokens, completion_tokens, total_tokens, cost, key_id, team_id, user_id, \
 end_user_id, tag_name, tags, status, latency_ms, deployment_id";

const MODEL_GROUP_SPEND_LOG_SELECT_COLS: &str = "id, timestamp, request_id, model, provider, \
 prompt_tokens, completion_tokens, total_tokens, cost, key_id, team_id, user_id, \
 end_user_id, tag_name, tags, status, latency_ms, deployment_id, requested_model, model_group";

pub(crate) const SPEND_LOG_SELECT_COLS: &str = "id, timestamp, request_id, model, provider, \
 prompt_tokens, completion_tokens, total_tokens, cost, key_id, team_id, user_id, \
 end_user_id, tag_name, tags, status, latency_ms, deployment_id, requested_model, model_group, session_id";

const STARTUP_REPAIR_BATCHES: usize = 4;

impl Database {
    /// Inspect at most 256 historical rows and commit their repaired timestamps
    /// and memberships together. A database cursor resumes the sweep on later
    /// calls/startups. No DDL or migration advisory lock is held here.
    /// Durable customer/tag counters are never replayed by metadata repair.
    ///
    /// PostgreSQL skips contending workers/rows instead of waiting on their row
    /// locks. SQLite reserves its writer lock before reading the batch, avoiding
    /// a deferred-transaction upgrade race. Skipped rows remain correct through
    /// the read fallback and are revisited on the next sweep.
    pub async fn repair_spend_metadata_batch(&self, limit: i64) -> Result<SpendMetadataRepair> {
        let limit = limit.clamp(1, SPEND_METADATA_REPAIR_BATCH_SIZE);
        let mut tx = match self.placeholder_style() {
            PlaceholderStyle::Dollar => self.pool().begin().await?,
            PlaceholderStyle::Question => self.pool().begin_with("BEGIN IMMEDIATE").await?,
        };
        let lock = match self.placeholder_style() {
            PlaceholderStyle::Dollar => {
                sqlx::query("SET LOCAL lock_timeout = '100ms'")
                    .execute(&mut *tx)
                    .await?;
                " FOR UPDATE SKIP LOCKED"
            }
            PlaceholderStyle::Question => "",
        };
        let q = format!(
            "SELECT CAST(last_id AS BIGINT) AS last_id FROM _spend_metadata_repair \
             WHERE name = 'spend-v1'{lock}"
        );
        let state = sqlx::query(&q).fetch_optional(&mut *tx).await?;
        let Some(state) = state else {
            tx.rollback().await?;
            return Ok(SpendMetadataRepair {
                scanned_rows: 0,
                repaired_rows: 0,
                has_more: true,
            });
        };
        let cursor: Option<i64> = state.try_get("last_id")?;
        let where_sql = cursor
            .map(|_| format!(" WHERE id > {}", self.ph(1)))
            .unwrap_or_default();
        let limit_idx = if cursor.is_some() { 2 } else { 1 };
        let q = format!(
            "SELECT CAST(id AS BIGINT) AS id, timestamp, tags, tag_name, \
             CAST(metadata_version AS BIGINT) AS metadata_version \
             FROM spend_logs{where_sql} ORDER BY id LIMIT {}{lock}",
            self.ph(limit_idx)
        );
        let mut query = sqlx::query(&q);
        if let Some(id) = cursor {
            query = query.bind(id);
        }
        let rows = query.bind(limit + 1).fetch_all(&mut *tx).await?;
        let has_more = rows.len() > limit as usize;
        let scanned_rows = rows.len().min(limit as usize);
        let mut last_id = cursor;
        let mut pending_ids = Vec::new();
        let mut timestamps = Vec::new();
        let mut memberships = Vec::new();
        for row in rows.iter().take(limit as usize) {
            let id: i64 = row.try_get("id")?;
            last_id = Some(id);
            if row.try_get::<i64, _>("metadata_version")? == 1 {
                continue;
            }
            pending_ids.push(id);
            let timestamp: String = row.try_get("timestamp")?;
            let normalized = crate::normalize_iso(&timestamp);
            if timestamp != normalized {
                timestamps.push((id, normalized));
            }
            let tags: Option<String> = row.try_get("tags")?;
            let primary: Option<String> = row.try_get("tag_name")?;
            memberships.extend(
                spend_tag_names(tags.as_deref(), primary.as_deref())
                    .into_iter()
                    .map(|name| (id, name)),
            );
        }
        if !timestamps.is_empty() {
            let values = (0..timestamps.len())
                .map(|i| {
                    format!(
                        "(CAST({} AS BIGINT), CAST({} AS TEXT))",
                        self.ph(i * 2 + 1),
                        self.ph(i * 2 + 2)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let q = format!(
                "WITH normalized(id, ts) AS (VALUES {values}) \
                 UPDATE spend_logs SET timestamp = normalized.ts \
                 FROM normalized WHERE spend_logs.id = normalized.id"
            );
            let mut query = sqlx::query(&q);
            for (id, timestamp) in timestamps {
                query = query.bind(id).bind(timestamp);
            }
            query.execute(&mut *tx).await?;
        }
        // An imported UPDATE may have invalidated formerly indexed metadata.
        // Replace its memberships, rather than retaining stale attribution.
        if !pending_ids.is_empty() {
            let ids = (0..pending_ids.len())
                .map(|i| self.ph(i + 1))
                .collect::<Vec<_>>()
                .join(", ");
            let q = format!("DELETE FROM spend_log_tags WHERE spend_log_id IN ({ids})");
            let mut query = sqlx::query(&q);
            for id in &pending_ids {
                query = query.bind(id);
            }
            query.execute(&mut *tx).await?;
            self.insert_spend_tag_memberships(&mut tx, &memberships)
                .await?;
            // Timestamp invalidation triggers run before this final marker.
            let q = format!("UPDATE spend_logs SET metadata_version = 1 WHERE id IN ({ids})");
            let mut query = sqlx::query(&q);
            for id in &pending_ids {
                query = query.bind(id);
            }
            query.execute(&mut *tx).await?;
        }
        let q = format!(
            "UPDATE _spend_metadata_repair SET last_id = {} WHERE name = 'spend-v1'",
            self.ph(1)
        );
        sqlx::query(&q)
            .bind(if has_more { last_id } else { None })
            .execute(&mut *tx)
            .await?;
        tx.commit().await.map_err(crate::CommitFailed)?;
        Ok(SpendMetadataRepair {
            scanned_rows,
            repaired_rows: pending_ids.len(),
            has_more,
        })
    }

    /// Opportunistic maintenance only: read correctness and schema readiness
    /// are already established by the committed DDL/read views. A busy or bad
    /// historical row must not prevent a new replica from serving correct reads.
    pub(crate) async fn repair_spend_metadata_on_startup(&self) {
        for _ in 0..STARTUP_REPAIR_BATCHES {
            match self
                .repair_spend_metadata_batch(SPEND_METADATA_REPAIR_BATCH_SIZE)
                .await
            {
                Ok(page) if page.has_more && page.scanned_rows != 0 => {
                    tokio::task::yield_now().await;
                }
                Ok(_) => break,
                Err(error) => {
                    tracing::warn!(
                        error = %crate::redact_url_credentials(&error.to_string()),
                        "spend metadata repair deferred; rolling-upgrade read fallback remains active"
                    );
                    break;
                }
            }
        }
    }
}

fn read_columns(timestamp: &str, columns: &str) -> String {
    columns
        .split(',')
        .map(str::trim)
        .map(|column| {
            if column == "timestamp" {
                format!("{timestamp} AS timestamp")
            } else {
                column.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn statements_for(style: PlaceholderStyle) -> Vec<String> {
    let add = match style {
        PlaceholderStyle::Dollar => "ADD COLUMN IF NOT EXISTS",
        PlaceholderStyle::Question => "ADD COLUMN",
    };
    let mut statements = vec![
        format!("ALTER TABLE spend_logs {add} metadata_version INTEGER NOT NULL DEFAULT 0"),
        "CREATE TABLE IF NOT EXISTS _spend_metadata_repair (\
         name TEXT PRIMARY KEY, last_id BIGINT)"
            .to_string(),
        "INSERT INTO _spend_metadata_repair (name, last_id) VALUES ('spend-v1', NULL) \
         ON CONFLICT (name) DO NOTHING"
            .to_string(),
    ];
    match style {
        PlaceholderStyle::Dollar => {
            statements.extend([
                // Parse whole seconds separately: PostgreSQL timestamps round
                // fractional seconds to microseconds, which loses nanoseconds.
                r#"CREATE OR REPLACE FUNCTION oxllm_spend_timestamp(value TEXT)
                   RETURNS TEXT LANGUAGE plpgsql IMMUTABLE STRICT AS $$
                   DECLARE parts TEXT[]; whole TIMESTAMP; body TEXT; day TEXT;
                     clock TEXT; zone TEXT; hours INTEGER; minutes INTEGER; seconds INTEGER;
                     zone_hours INTEGER; zone_minutes INTEGER; zone_seconds INTEGER; delta INTEGER := 0;
                   BEGIN
                     IF length(value) = 30 AND value ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[.][0-9]{9}Z$'
                        AND substring(value FROM 18 FOR 2) <> '60' THEN RETURN value; END IF;
                     body := value;
                     IF position('[' IN body) > 0 AND right(body, 1) = ']' THEN
                       body := split_part(body, '[', 1);
                     END IF;
                     body := replace(body, '−', '-');
                     parts := regexp_match(body,
                       '^([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{8})[Tt ]([0-9]{2}(?::[0-9]{2}(?::[0-9]{2})?|[0-9]{2}(?:[0-9]{2})?)?)(?:[.,]([0-9]{1,9}))?([Zz]|[+-][0-9]{2}(?::[0-9]{2}(?::[0-9]{2})?|[0-9]{2}(?:[0-9]{2})?)?)$');
                     IF parts IS NULL THEN RETURN value; END IF;
                     clock := replace(parts[2], ':', '');
                     IF parts[3] IS NOT NULL AND length(clock) <> 6 THEN RETURN value; END IF;
                     hours := substring(clock FROM 1 FOR 2)::INTEGER;
                     minutes := COALESCE(NULLIF(substring(clock FROM 3 FOR 2), '')::INTEGER, 0);
                     seconds := COALESCE(NULLIF(substring(clock FROM 5 FOR 2), '')::INTEGER, 0);
                     IF hours > 23 OR minutes > 59 OR seconds > 60 THEN RETURN value; END IF;
                     -- Match the Rust parser's leap-second clamp rather than rounding into the next day.
                     seconds := LEAST(seconds, 59);
                     zone := replace(parts[4], ':', '');
                     IF zone NOT IN ('Z', 'z') THEN
                       zone_hours := substring(zone FROM 2 FOR 2)::INTEGER;
                       zone_minutes := COALESCE(NULLIF(substring(zone FROM 4 FOR 2), '')::INTEGER, 0);
                       zone_seconds := COALESCE(NULLIF(substring(zone FROM 6 FOR 2), '')::INTEGER, 0);
                       IF zone_hours > 25 OR zone_minutes > 59 OR zone_seconds > 59 THEN RETURN value; END IF;
                       delta := zone_hours * 3600 + zone_minutes * 60 + zone_seconds;
                       IF left(zone, 1) = '-' THEN delta := -delta; END IF;
                     END IF;
                     day := parts[1];
                     IF length(day) = 8 THEN
                       day := substring(day FROM 1 FOR 4) || '-' || substring(day FROM 5 FOR 2) || '-' || substring(day FROM 7 FOR 2);
                     END IF;
                     whole := (day || 'T' || lpad(hours::TEXT, 2, '0') || ':' ||
                               lpad(minutes::TEXT, 2, '0') || ':' || lpad(seconds::TEXT, 2, '0'))::TIMESTAMP
                              - delta * INTERVAL '1 second';
                     RETURN to_char(whole, 'YYYY-MM-DD"T"HH24:MI:SS') || '.' ||
                            rpad(COALESCE(parts[3], ''), 9, '0') || 'Z';
                   EXCEPTION WHEN invalid_datetime_format OR datetime_field_overflow THEN
                     RETURN value;
                   END; $$"#.to_string(),
                r#"CREATE OR REPLACE FUNCTION oxllm_spend_tag_names(tag_json TEXT, primary_tag TEXT)
                   RETURNS TABLE (tag_name TEXT) LANGUAGE plpgsql IMMUTABLE AS $$
                   DECLARE parsed JSONB;
                   BEGIN
                     BEGIN parsed := tag_json::JSONB;
                     EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN
                       parsed := NULL;
                     END;
                     IF jsonb_typeof(parsed) = 'array' THEN
                       IF NOT EXISTS (SELECT 1 FROM jsonb_array_elements(parsed) e(value)
                                      WHERE jsonb_typeof(e.value) <> 'string') THEN
                         RETURN QUERY SELECT DISTINCT (e.value #>> '{}') COLLATE "C"
                           FROM jsonb_array_elements(parsed) e(value)
                           WHERE e.value #>> '{}' <> '';
                         IF FOUND THEN RETURN; END IF;
                       END IF;
                     END IF;
                     RETURN QUERY SELECT primary_tag WHERE primary_tag IS NOT NULL AND primary_tag <> '';
                   END; $$"#.to_string(),
                "CREATE OR REPLACE FUNCTION oxllm_invalidate_spend_metadata() RETURNS trigger \
                 LANGUAGE plpgsql AS $$ BEGIN NEW.metadata_version := 0; RETURN NEW; END; $$".to_string(),
                "DROP TRIGGER IF EXISTS invalidate_spend_metadata ON spend_logs".to_string(),
                "CREATE TRIGGER invalidate_spend_metadata BEFORE UPDATE OF timestamp, tags, tag_name \
                 ON spend_logs FOR EACH ROW \
                 WHEN (OLD.timestamp IS DISTINCT FROM NEW.timestamp OR OLD.tags IS DISTINCT FROM NEW.tags \
                       OR OLD.tag_name IS DISTINCT FROM NEW.tag_name) \
                 EXECUTE FUNCTION oxllm_invalidate_spend_metadata()".to_string(),
                format!(
                    "CREATE OR REPLACE VIEW spend_logs_read AS \
                      SELECT {LEGACY_SPEND_LOG_SELECT_COLS} FROM spend_logs WHERE metadata_version = 1 \
                     UNION ALL SELECT {} FROM spend_logs WHERE metadata_version <> 1",
                    read_columns("oxllm_spend_timestamp(timestamp)", LEGACY_SPEND_LOG_SELECT_COLS)
                ),
                "CREATE OR REPLACE VIEW spend_log_tags_read AS \
                 SELECT st.spend_log_id, st.tag_name FROM spend_log_tags st \
                 JOIN spend_logs s ON s.id = st.spend_log_id WHERE s.metadata_version = 1 \
                 UNION ALL SELECT s.id, t.tag_name COLLATE \"C\" FROM spend_logs s \
                 CROSS JOIN LATERAL oxllm_spend_tag_names(s.tags, s.tag_name) t \
                 WHERE s.metadata_version <> 1".to_string(),
            ]);
        }
        PlaceholderStyle::Question => {
            statements.extend([
                "CREATE TRIGGER IF NOT EXISTS invalidate_spend_metadata \
                 AFTER UPDATE OF timestamp, tags, tag_name ON spend_logs \
                 WHEN OLD.timestamp IS NOT NEW.timestamp OR OLD.tags IS NOT NEW.tags \
                      OR OLD.tag_name IS NOT NEW.tag_name \
                 BEGIN UPDATE spend_logs SET metadata_version = 0 WHERE id = NEW.id; END"
                    .to_string(),
                sqlite_read_view(LEGACY_SPEND_LOG_SELECT_COLS),
                "CREATE VIEW IF NOT EXISTS spend_log_tags_read AS \
                 WITH legacy AS (SELECT id, tag_name, CASE WHEN json_valid(tags) THEN \
                   CASE WHEN json_type(tags) = 'array' AND NOT EXISTS \
                     (SELECT 1 FROM json_each(tags) e WHERE e.type <> 'text') \
                   THEN tags ELSE '[]' END ELSE '[]' END AS json_tags \
                   FROM spend_logs WHERE metadata_version <> 1), \
                 decoded AS (SELECT DISTINCT l.id, e.value COLLATE BINARY AS tag_name \
                   FROM legacy l JOIN json_each(l.json_tags) e WHERE e.value <> '') \
                 SELECT st.spend_log_id, st.tag_name FROM spend_log_tags st \
                   JOIN spend_logs s ON s.id = st.spend_log_id WHERE s.metadata_version = 1 \
                 UNION ALL SELECT id, tag_name FROM decoded \
                 UNION ALL SELECT l.id, l.tag_name COLLATE BINARY FROM legacy l \
                   WHERE l.tag_name IS NOT NULL AND l.tag_name <> '' AND NOT EXISTS \
                     (SELECT 1 FROM decoded d WHERE d.id = l.id)"
                    .to_string(),
            ]);
        }
    }
    statements
}

/// SQLite's date functions also discard submilliseconds. Strip the fraction
/// before UTC conversion and append its original digits padded to nine places.
/// Offset arithmetic uses integral seconds, including offsets across UTC days.
fn sqlite_read_view(columns: &str) -> String {
    // Keep already-canonical historical rows cheap while their index marker
    // awaits repair. The expanded parser runs only for noncanonical timestamps.
    let timestamp = r#"CASE WHEN length(timestamp) = 30
        AND timestamp GLOB '????-??-??T??:??:??.?????????Z' AND substr(timestamp, 18, 2) <> '60'
        THEN timestamp
        ELSE CASE WHEN date_len IN (8, 10) AND separator IN ('T', 't', ' ')
        AND day GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
        AND length(clock_digits) IN (2, 4, 6) AND clock_digits NOT GLOB '*[^0-9]*'
        AND (clock = clock_digits OR
          (length(clock_digits) = 4 AND clock = substr(clock_digits, 1, 2) || ':' || substr(clock_digits, 3, 2)) OR
          (length(clock_digits) = 6 AND clock = substr(clock_digits, 1, 2) || ':' || substr(clock_digits, 3, 2) || ':' || substr(clock_digits, 5, 2)))
        AND CAST(substr(clock_digits, 1, 2) AS INTEGER) <= 23
        AND CAST(substr(clock_digits, 3, 2) AS INTEGER) <= 59
        AND CAST(substr(clock_digits, 5, 2) AS INTEGER) <= 60
        AND ((decimal_pos = 0 AND fraction = '') OR
          (decimal_pos > 0 AND length(clock_digits) = 6 AND length(fraction) BETWEEN 1 AND 9))
        AND fraction NOT GLOB '*[^0-9]*'
        AND (zone IN ('Z', 'z') OR (length(zone_digits) IN (3, 5, 7)
          AND substr(zone_digits, 1, 1) IN ('+', '-') AND substr(zone_digits, 2) NOT GLOB '*[^0-9]*'
          AND (zone = zone_digits OR
            (length(zone_digits) = 5 AND zone = substr(zone_digits, 1, 3) || ':' || substr(zone_digits, 4, 2)) OR
            (length(zone_digits) = 7 AND zone = substr(zone_digits, 1, 3) || ':' || substr(zone_digits, 4, 2) || ':' || substr(zone_digits, 6, 2)))
          AND CAST(substr(zone_digits, 2, 2) AS INTEGER) <= 25
          AND CAST(substr(zone_digits, 4, 2) AS INTEGER) <= 59
          AND CAST(substr(zone_digits, 6, 2) AS INTEGER) <= 59))
        AND strftime('%Y-%m-%dT%H:%M:%S', whole_second) = whole_second
        THEN COALESCE(strftime('%Y-%m-%dT%H:%M:%S', whole_second, printf('%+d seconds', offset_seconds))
          || '.' || substr(fraction || '000000000', 1, 9) || 'Z', timestamp)
        ELSE timestamp END END"#;
    format!(
        r#"CREATE VIEW IF NOT EXISTS spend_logs_read AS
         WITH input AS (SELECT *, replace(CASE WHEN instr(timestamp, '[') > 0 AND substr(timestamp, -1) = ']'
           THEN substr(timestamp, 1, instr(timestamp, '[') - 1) ELSE timestamp END, '−', '-') AS iso
           FROM spend_logs WHERE metadata_version <> 1),
         calendar AS (SELECT *, CASE WHEN substr(iso, 5, 1) = '-' AND substr(iso, 8, 1) = '-' THEN 10
           WHEN substr(iso, 1, 8) NOT GLOB '*[^0-9]*' THEN 8 ELSE 0 END AS date_len FROM input),
         tails AS (SELECT *, CASE WHEN date_len = 10 THEN substr(iso, 1, 10)
           ELSE substr(iso, 1, 4) || '-' || substr(iso, 5, 2) || '-' || substr(iso, 7, 2) END AS day,
           substr(iso, date_len + 1, 1) AS separator, substr(iso, date_len + 2) AS tail FROM calendar),
         zones AS (SELECT *, CASE WHEN instr(tail, 'Z') > 0 THEN instr(tail, 'Z')
           WHEN instr(tail, 'z') > 0 THEN instr(tail, 'z') WHEN instr(tail, '+') > 0 THEN instr(tail, '+')
           WHEN instr(tail, '-') > 0 THEN instr(tail, '-') ELSE 0 END AS zone_pos FROM tails),
         parts AS (SELECT *, substr(tail, 1, zone_pos - 1) AS clock_fraction,
           CASE WHEN zone_pos > 0 THEN substr(tail, zone_pos) ELSE NULL END AS zone FROM zones),
         decimals AS (SELECT *, CASE WHEN instr(clock_fraction, '.') > 0 THEN instr(clock_fraction, '.')
           WHEN instr(clock_fraction, ',') > 0 THEN instr(clock_fraction, ',') ELSE 0 END AS decimal_pos FROM parts),
         fractions AS (SELECT *, CASE WHEN decimal_pos = 0 THEN clock_fraction
           ELSE substr(clock_fraction, 1, decimal_pos - 1) END AS clock,
           CASE WHEN decimal_pos = 0 THEN '' ELSE substr(clock_fraction, decimal_pos + 1) END AS fraction FROM decimals),
         digits AS (SELECT *, replace(clock, ':', '') AS clock_digits, replace(zone, ':', '') AS zone_digits FROM fractions),
         normalized AS (SELECT *, day || 'T' || substr(clock_digits, 1, 2) || ':' ||
           CASE WHEN length(clock_digits) >= 4 THEN substr(clock_digits, 3, 2) ELSE '00' END || ':' ||
           CASE WHEN length(clock_digits) >= 6 THEN printf('%02d', min(CAST(substr(clock_digits, 5, 2) AS INTEGER), 59)) ELSE '00' END AS whole_second,
           CASE WHEN zone IN ('Z', 'z') THEN 0 ELSE
             (CASE WHEN substr(zone_digits, 1, 1) = '-' THEN 1 ELSE -1 END) *
             (CAST(substr(zone_digits, 2, 2) AS INTEGER) * 3600 + CAST(substr(zone_digits, 4, 2) AS INTEGER) * 60 +
               CAST(substr(zone_digits, 6, 2) AS INTEGER)) END AS offset_seconds FROM digits)
          SELECT {columns} FROM spend_logs WHERE metadata_version = 1
         UNION ALL SELECT {} FROM normalized"#,
        read_columns(timestamp, columns)
    )
}

pub(crate) fn deployment_attribution_statements(style: PlaceholderStyle) -> Vec<String> {
    let add = match style {
        PlaceholderStyle::Dollar => "ADD COLUMN IF NOT EXISTS",
        PlaceholderStyle::Question => "ADD COLUMN",
    };
    let mut statements = vec![format!("ALTER TABLE spend_logs {add} deployment_id TEXT")];
    statements.extend(read_view_statements(
        style,
        DEPLOYMENT_SPEND_LOG_SELECT_COLS,
    ));
    statements
}

pub(crate) fn model_group_attribution_statements(style: PlaceholderStyle) -> Vec<String> {
    let add = match style {
        PlaceholderStyle::Dollar => "ADD COLUMN IF NOT EXISTS",
        PlaceholderStyle::Question => "ADD COLUMN",
    };
    let mut statements = vec![
        format!("ALTER TABLE spend_logs {add} requested_model TEXT"),
        format!("ALTER TABLE spend_logs {add} model_group TEXT"),
    ];
    statements.extend(read_view_statements(
        style,
        MODEL_GROUP_SPEND_LOG_SELECT_COLS,
    ));
    statements
}

pub(crate) fn session_attribution_statements(style: PlaceholderStyle) -> Vec<String> {
    let add = match style {
        PlaceholderStyle::Dollar => "ADD COLUMN IF NOT EXISTS",
        PlaceholderStyle::Question => "ADD COLUMN",
    };
    let mut statements = vec![
        format!("ALTER TABLE spend_logs {add} session_id TEXT"),
        "CREATE INDEX IF NOT EXISTS idx_spend_logs_session_timestamp ON spend_logs(session_id, timestamp)".into(),
    ];
    statements.extend(read_view_statements(style, SPEND_LOG_SELECT_COLS));
    statements
}

fn read_view_statements(style: PlaceholderStyle, columns: &str) -> Vec<String> {
    match style {
        PlaceholderStyle::Dollar => vec![format!(
            "CREATE OR REPLACE VIEW spend_logs_read AS SELECT {columns} FROM spend_logs WHERE metadata_version = 1 \
             UNION ALL SELECT {} FROM spend_logs WHERE metadata_version <> 1",
            read_columns("oxllm_spend_timestamp(timestamp)", columns)
        )],
        PlaceholderStyle::Question => vec![
            "DROP VIEW IF EXISTS spend_logs_read".into(),
            sqlite_read_view(columns),
        ],
    }
}
