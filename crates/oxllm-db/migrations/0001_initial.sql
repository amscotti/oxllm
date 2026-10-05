-- 0001_initial.sql — reference snapshot of the full v1–v14 schema for a
-- FRESH database (SQLite flavour).
--
-- NOTE: `src/migrate.rs` is the authoritative source of truth. This file is
-- a human-readable reference only; the application applies migrations via
-- `Database::migrate`, and fresh databases already include every column that
-- the additive `ALTER TABLE ... ADD COLUMN` batches exist to
-- retrofit onto older databases. Do not apply this file manually in place of
-- running the migrations.

CREATE TABLE IF NOT EXISTS keys (
  key_id TEXT PRIMARY KEY,
  key_hash TEXT UNIQUE NOT NULL,
  key_alias TEXT,
  team_id TEXT,
  user_id TEXT,
  org_id TEXT,
  project_id TEXT,
  max_budget REAL,
  budget_duration TEXT,
  rpm INTEGER,
  tpm INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  expires_at TEXT,
  is_active INTEGER NOT NULL DEFAULT 1,
  models TEXT,
  allowed_cache_controls TEXT
);

CREATE TABLE IF NOT EXISTS teams (
  team_id TEXT PRIMARY KEY,
  team_name TEXT NOT NULL,
  max_budget REAL,
  budget_duration TEXT,
  rpm INTEGER,
  tpm INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  models TEXT,
  blocked INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  user_name TEXT NOT NULL,
  max_budget REAL,
  budget_duration TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  models TEXT
);

CREATE TABLE IF NOT EXISTS organizations (
  org_id TEXT PRIMARY KEY,
  org_name TEXT NOT NULL,
  max_budget REAL,
  budget_duration TEXT,
  rpm INTEGER,
  tpm INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  models TEXT,
  metadata TEXT,
  soft_budget REAL,
  max_parallel_requests INTEGER,
  model_max_budget TEXT
);

CREATE TABLE IF NOT EXISTS projects (
  project_id TEXT PRIMARY KEY,
  project_name TEXT NOT NULL,
  org_id TEXT,
  team_id TEXT,
  max_budget REAL,
  budget_duration TEXT,
  rpm INTEGER,
  tpm INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spend_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  request_id TEXT,
  model TEXT NOT NULL,
  provider TEXT NOT NULL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  total_tokens INTEGER,
  cost REAL,
  key_id TEXT,
  team_id TEXT,
  user_id TEXT,
  status TEXT,
  latency_ms INTEGER,
  end_user_id TEXT,
  tag_name TEXT,
  tags TEXT,
  event_id TEXT,
  metadata_version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS spend_log_tags (
  spend_log_id BIGINT NOT NULL REFERENCES spend_logs(id) ON DELETE CASCADE,
  tag_name TEXT COLLATE BINARY NOT NULL,
  PRIMARY KEY (spend_log_id, tag_name)
);

CREATE TABLE IF NOT EXISTS _spend_metadata_repair (
  name TEXT PRIMARY KEY,
  last_id BIGINT
);
INSERT INTO _spend_metadata_repair (name, last_id) VALUES ('spend-v1', NULL)
  ON CONFLICT (name) DO NOTHING;

CREATE TRIGGER IF NOT EXISTS invalidate_spend_metadata
AFTER UPDATE OF timestamp, tags, tag_name ON spend_logs
WHEN OLD.timestamp IS NOT NEW.timestamp OR OLD.tags IS NOT NEW.tags
  OR OLD.tag_name IS NOT NEW.tag_name
BEGIN
  UPDATE spend_logs SET metadata_version = 0 WHERE id = NEW.id;
END;

-- Read fallbacks remain active for old-replica and imported writes. Repair
-- commits a bounded batch at a time and never gates schema readiness.
CREATE VIEW IF NOT EXISTS spend_logs_read AS
WITH input AS (
  SELECT *, replace(CASE WHEN instr(timestamp, '[') > 0 AND substr(timestamp, -1) = ']'
    THEN substr(timestamp, 1, instr(timestamp, '[') - 1) ELSE timestamp END, '−', '-') AS iso
  FROM spend_logs WHERE metadata_version <> 1
), calendar AS (
  SELECT *, CASE WHEN substr(iso, 5, 1) = '-' AND substr(iso, 8, 1) = '-' THEN 10
    WHEN substr(iso, 1, 8) NOT GLOB '*[^0-9]*' THEN 8 ELSE 0 END AS date_len FROM input
), tails AS (
  SELECT *, CASE WHEN date_len = 10 THEN substr(iso, 1, 10)
    ELSE substr(iso, 1, 4) || '-' || substr(iso, 5, 2) || '-' || substr(iso, 7, 2) END AS day,
    substr(iso, date_len + 1, 1) AS separator, substr(iso, date_len + 2) AS tail FROM calendar
), zones AS (
  SELECT *, CASE WHEN instr(tail, 'Z') > 0 THEN instr(tail, 'Z')
    WHEN instr(tail, 'z') > 0 THEN instr(tail, 'z') WHEN instr(tail, '+') > 0 THEN instr(tail, '+')
    WHEN instr(tail, '-') > 0 THEN instr(tail, '-') ELSE 0 END AS zone_pos FROM tails
), parts AS (
  SELECT *, substr(tail, 1, zone_pos - 1) AS clock_fraction,
    CASE WHEN zone_pos > 0 THEN substr(tail, zone_pos) ELSE NULL END AS zone FROM zones
), decimals AS (
  SELECT *, CASE WHEN instr(clock_fraction, '.') > 0 THEN instr(clock_fraction, '.')
    WHEN instr(clock_fraction, ',') > 0 THEN instr(clock_fraction, ',') ELSE 0 END AS decimal_pos FROM parts
), fractions AS (
  SELECT *, CASE WHEN decimal_pos = 0 THEN clock_fraction
    ELSE substr(clock_fraction, 1, decimal_pos - 1) END AS clock,
    CASE WHEN decimal_pos = 0 THEN '' ELSE substr(clock_fraction, decimal_pos + 1) END AS fraction FROM decimals
), digits AS (
  SELECT *, replace(clock, ':', '') AS clock_digits, replace(zone, ':', '') AS zone_digits FROM fractions
), normalized AS (
  SELECT *, day || 'T' || substr(clock_digits, 1, 2) || ':' ||
    CASE WHEN length(clock_digits) >= 4 THEN substr(clock_digits, 3, 2) ELSE '00' END || ':' ||
    CASE WHEN length(clock_digits) >= 6 THEN printf('%02d', min(CAST(substr(clock_digits, 5, 2) AS INTEGER), 59)) ELSE '00' END AS whole_second,
    CASE WHEN zone IN ('Z', 'z') THEN 0 ELSE
      (CASE WHEN substr(zone_digits, 1, 1) = '-' THEN 1 ELSE -1 END) *
      (CAST(substr(zone_digits, 2, 2) AS INTEGER) * 3600 + CAST(substr(zone_digits, 4, 2) AS INTEGER) * 60 +
        CAST(substr(zone_digits, 6, 2) AS INTEGER)) END AS offset_seconds FROM digits
)
SELECT id, timestamp, request_id, model, provider, prompt_tokens, completion_tokens,
  total_tokens, cost, key_id, team_id, user_id, end_user_id, tag_name, tags, status, latency_ms
FROM spend_logs WHERE metadata_version = 1
UNION ALL
SELECT id, CASE WHEN length(timestamp) = 30
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
  ELSE timestamp END END AS timestamp,
  request_id, model, provider, prompt_tokens, completion_tokens, total_tokens, cost,
  key_id, team_id, user_id, end_user_id, tag_name, tags, status, latency_ms
FROM normalized;

CREATE VIEW IF NOT EXISTS spend_log_tags_read AS
WITH legacy AS (
  SELECT id, tag_name, CASE WHEN json_valid(tags) THEN
    CASE WHEN json_type(tags) = 'array' AND NOT EXISTS
      (SELECT 1 FROM json_each(tags) e WHERE e.type <> 'text')
    THEN tags ELSE '[]' END ELSE '[]' END AS json_tags
  FROM spend_logs WHERE metadata_version <> 1
), decoded AS (
  SELECT DISTINCT l.id, e.value COLLATE BINARY AS tag_name
  FROM legacy l JOIN json_each(l.json_tags) e WHERE e.value <> ''
)
SELECT st.spend_log_id, st.tag_name FROM spend_log_tags st
  JOIN spend_logs s ON s.id = st.spend_log_id WHERE s.metadata_version = 1
UNION ALL SELECT id, tag_name FROM decoded
UNION ALL SELECT l.id, l.tag_name COLLATE BINARY FROM legacy l
  WHERE l.tag_name IS NOT NULL AND l.tag_name <> ''
    AND NOT EXISTS (SELECT 1 FROM decoded d WHERE d.id = l.id);

CREATE TABLE IF NOT EXISTS models (
  model_id TEXT PRIMARY KEY,
  model_name TEXT NOT NULL,
  oxllm_params TEXT NOT NULL,
  model_info TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Existing duplicate records are preserved; new inserts/renames claim aliases.
CREATE TABLE IF NOT EXISTS model_aliases (
  model_name TEXT COLLATE BINARY NOT NULL CONSTRAINT model_aliases_name PRIMARY KEY
);
INSERT INTO model_aliases (model_name)
  SELECT DISTINCT model_name COLLATE BINARY FROM models WHERE TRUE
  ON CONFLICT (model_name) DO NOTHING;

CREATE TRIGGER IF NOT EXISTS claim_model_alias_insert BEFORE INSERT ON models
BEGIN
  INSERT INTO model_aliases (model_name) VALUES (NEW.model_name);
END;
CREATE TRIGGER IF NOT EXISTS claim_model_alias_update BEFORE UPDATE OF model_name ON models
WHEN OLD.model_name COLLATE BINARY IS NOT NEW.model_name COLLATE BINARY
BEGIN
  INSERT INTO model_aliases (model_name) VALUES (NEW.model_name);
END;
CREATE TRIGGER IF NOT EXISTS release_model_alias_update AFTER UPDATE OF model_name ON models
WHEN OLD.model_name COLLATE BINARY IS NOT NEW.model_name COLLATE BINARY
BEGIN
  DELETE FROM model_aliases WHERE model_name = OLD.model_name
    AND NOT EXISTS (SELECT 1 FROM models WHERE model_name COLLATE BINARY = OLD.model_name COLLATE BINARY);
END;
CREATE TRIGGER IF NOT EXISTS release_model_alias_delete AFTER DELETE ON models
BEGIN
  DELETE FROM model_aliases WHERE model_name = OLD.model_name
    AND NOT EXISTS (SELECT 1 FROM models WHERE model_name COLLATE BINARY = OLD.model_name COLLATE BINARY);
END;

-- v3: end-user (customer), budget plan, and tag tables.
CREATE TABLE IF NOT EXISTS customers (
  end_user_id TEXT PRIMARY KEY,
  alias TEXT,
  email TEXT,
  max_budget REAL,
  spend REAL NOT NULL DEFAULT 0,
  budget_duration TEXT,
  blocked INTEGER NOT NULL DEFAULT 0,
  model_blocklist TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS budgets (
  budget_id TEXT PRIMARY KEY,
  max_budget REAL,
  soft_budget REAL,
  max_parallel_requests INTEGER,
  tpm_limit INTEGER,
  rpm_limit INTEGER,
  budget_duration TEXT,
  model_max_budget TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
  name TEXT PRIMARY KEY,
  description TEXT,
  models TEXT,
  model_info TEXT,
  spend REAL,
  max_budget REAL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- v5: skills table (versioned instruction files).
CREATE TABLE IF NOT EXISTS skills (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  content TEXT NOT NULL,
  version TEXT NOT NULL DEFAULT '1.0',
  model_pattern TEXT,
  team_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_spend_logs_timestamp ON spend_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_spend_logs_key_id ON spend_logs(key_id);
CREATE INDEX IF NOT EXISTS idx_spend_logs_model ON spend_logs(model);
CREATE INDEX IF NOT EXISTS idx_spend_logs_team_id ON spend_logs(team_id);
CREATE INDEX IF NOT EXISTS idx_spend_logs_key_timestamp ON spend_logs(key_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_spend_logs_request_id ON spend_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_spend_logs_end_user_id ON spend_logs(end_user_id);
CREATE INDEX IF NOT EXISTS idx_spend_logs_tag_name ON spend_logs(tag_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_spend_logs_event_id ON spend_logs(event_id);
CREATE INDEX IF NOT EXISTS idx_spend_log_tags_name ON spend_log_tags(tag_name, spend_log_id);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(model_name);
CREATE INDEX IF NOT EXISTS idx_keys_team_id ON keys(team_id);
CREATE INDEX IF NOT EXISTS idx_keys_user_id ON keys(user_id);
CREATE INDEX IF NOT EXISTS idx_keys_org_id ON keys(org_id);
CREATE INDEX IF NOT EXISTS idx_keys_project_id ON keys(project_id);
CREATE INDEX IF NOT EXISTS idx_projects_org_id ON projects(org_id);
CREATE INDEX IF NOT EXISTS idx_projects_team_id ON projects(team_id);
CREATE INDEX IF NOT EXISTS idx_customers_blocked ON customers(blocked);
CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
