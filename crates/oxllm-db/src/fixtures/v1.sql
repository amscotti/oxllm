-- Frozen v1 schema from 194039b:crates/oxllm-db/src/lib.rs,
-- v0_schema_version_statement + v1_statements_for(Question).
-- Keep this independent of current migration generators. In particular,
-- teams.blocked, organization v4 fields and spend v8+ fields do not exist,
-- and models uses the original litellm_params column name.
CREATE TABLE _schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE keys (
    key_id TEXT PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL, key_alias TEXT,
    team_id TEXT, user_id TEXT, org_id TEXT, project_id TEXT, max_budget REAL,
    budget_duration TEXT, rpm INTEGER, tpm INTEGER, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, expires_at TEXT, is_active INTEGER NOT NULL DEFAULT 1,
    models TEXT
);
CREATE TABLE teams (
    team_id TEXT PRIMARY KEY, team_name TEXT NOT NULL, max_budget REAL,
    budget_duration TEXT, rpm INTEGER, tpm INTEGER, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, models TEXT
);
CREATE TABLE users (
    user_id TEXT PRIMARY KEY, user_name TEXT NOT NULL, max_budget REAL,
    budget_duration TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, models TEXT
);
CREATE TABLE organizations (
    org_id TEXT PRIMARY KEY, org_name TEXT NOT NULL, max_budget REAL,
    budget_duration TEXT, rpm INTEGER, tpm INTEGER, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY, project_name TEXT NOT NULL, org_id TEXT, team_id TEXT,
    max_budget REAL, budget_duration TEXT, rpm INTEGER, tpm INTEGER,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE spend_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, request_id TEXT,
    model TEXT NOT NULL, provider TEXT NOT NULL, prompt_tokens INTEGER,
    completion_tokens INTEGER, total_tokens INTEGER, cost REAL, key_id TEXT,
    team_id TEXT, user_id TEXT, status TEXT, latency_ms INTEGER
);
CREATE TABLE models (
    model_id TEXT PRIMARY KEY, model_name TEXT NOT NULL, litellm_params TEXT NOT NULL,
    model_info TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX idx_spend_logs_timestamp ON spend_logs(timestamp);
CREATE INDEX idx_spend_logs_key_id ON spend_logs(key_id);
CREATE INDEX idx_spend_logs_model ON spend_logs(model);
CREATE INDEX idx_spend_logs_team_id ON spend_logs(team_id);
CREATE INDEX idx_spend_logs_key_timestamp ON spend_logs(key_id, timestamp);
INSERT INTO _schema_version VALUES (1, '2024-01-01T00:00:00Z');

-- Synthetic records written with that schema, including a disabled key,
-- model configuration and three differently rendered timestamp fractions.
INSERT INTO organizations VALUES ('v1-org', 'Original org', 100.5, '30d', 20, 300,
    '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z');
INSERT INTO teams VALUES ('v1-team', 'Original team', 50.25, '7d', 10, 200,
    '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', '["gpt-4o"]');
INSERT INTO users VALUES ('v1-user', 'Original user', 25.5, '1d',
    '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', '["gpt-4o"]');
INSERT INTO projects VALUES ('v1-project', 'Original project', 'v1-org', 'v1-team',
    75.5, '30d', 15, 250, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z');
INSERT INTO keys VALUES ('v1-key', 'fixture-hash', 'Original key', 'v1-team', 'v1-user',
    'v1-org', 'v1-project', 12.5, '1d', 5, 100, '2024-01-01T00:00:00Z',
    '2024-01-01T00:00:00Z', '2030-01-01T00:00:00Z', 0, '["gpt-4o"]');
INSERT INTO models VALUES ('v1-model', 'Original model', '{"model":"openai/gpt-4o"}',
    '{"fixture":true}', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z');
INSERT INTO spend_logs (timestamp, request_id, model, provider, prompt_tokens,
    completion_tokens, total_tokens, cost, key_id, team_id, user_id, status, latency_ms)
VALUES ('2026-07-31T23:59:59Z', 'v1-whole', 'gpt-4o', 'openai', 10, 5, 15, 1.25,
    'v1-key', 'v1-team', 'v1-user', 'success', 42),
    ('2026-07-31T23:59:59.5Z', 'v1-half', 'gpt-4o', 'openai', 20, 10, 30, 2.5,
    'v1-key', 'v1-team', 'v1-user', 'success', 84),
    ('2026-07-31T23:59:59.999999999Z', 'v1-last', 'gpt-4o', 'openai', 2, 1, 3, 0.25,
    'v1-key', 'v1-team', 'v1-user', 'success', 7);
