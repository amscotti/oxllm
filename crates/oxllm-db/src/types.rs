//! Record types persisted by the oxllm database layer.

use serde::{Deserialize, Serialize};

/// A virtual API key row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRecord {
    pub key_id: String,
    pub key_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_alias: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_duration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<i64>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    pub is_active: bool,
    /// JSON-encoded array of allowed model patterns (e.g. `["gpt-4o", "claude-*"]`).
    /// `None` or empty array means "all models allowed".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<String>,
    /// JSON-encoded array of cache controls permitted for this key.
    /// `None` preserves the deployment's default cache-control policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_cache_controls: Option<String>,
}

/// A team row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamRecord {
    pub team_id: String,
    pub team_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_duration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<i64>,
    pub created_at: String,
    pub updated_at: String,
    /// JSON-encoded array of allowed model patterns. `None`/empty = all allowed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<String>,
    /// Whether the team is administratively blocked. Mirrors the reference
    /// `team.blocked`: when `true`, every key in the team is rejected at auth
    /// time regardless of the key's own `is_active` flag, so `/team/unblock`
    /// never clobbers individually-blocked keys.
    #[serde(default, skip_serializing_if = "is_false")]
    pub blocked: bool,
}

fn is_false(b: &bool) -> bool {
    !b
}

/// A user row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRecord {
    pub user_id: String,
    pub user_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_duration: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    /// JSON-encoded array of allowed model patterns. `None`/empty = all allowed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<String>,
}

/// An organization row. An organization groups teams and caps total spend
/// across all keys whose `org_id` points here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationRecord {
    pub org_id: String,
    pub org_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_duration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<i64>,
    pub created_at: String,
    pub updated_at: String,
    /// JSON-encoded array of allowed model patterns. `None`/empty = all allowed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<String>,
    /// Arbitrary JSON metadata blob.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<String>,
    /// Non-blocking budget threshold (advisory; does not reject requests).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soft_budget: Option<f64>,
    /// Concurrent request limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_parallel_requests: Option<i64>,
    /// JSON object of per-model max budgets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_max_budget: Option<String>,
}

/// A project row. A project optionally belongs to an org and/or a team and
/// caps total spend across all keys whose `project_id` points here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectRecord {
    pub project_id: String,
    pub project_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub org_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_duration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<i64>,
    pub created_at: String,
    pub updated_at: String,
}

/// A model deployment persisted via the management API.
///
/// `oxllm_params` / `model_info` are stored as JSON strings in the DB and
/// deserialized on load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigRecord {
    pub model_id: String,
    pub model_name: String,
    /// Provider config (same shape as `oxllm_params` in the YAML config).
    #[serde(alias = "litellm_params")]
    pub oxllm_params: serde_json::Value,
    /// Optional metadata (same shape as `model_info` in the YAML config).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_info: Option<serde_json::Value>,
    pub created_at: String,
    pub updated_at: String,
}

/// One spend log row (a single request's cost record).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendLog {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub model: String,
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// OpenAI request-body end-user / customer id (`customers.end_user_id`).
    /// Distinct from [`SpendLog::user_id`] (the virtual-key owner).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_user_id: Option<String>,
    /// Request tag from the `x-oxllm-tag` header (`tags.name`). Distinct from
    /// routing metadata; used for per-tag spend analytics and attribution.
    /// The PRIMARY (first) request tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag_name: Option<String>,
    /// Every request tag as a JSON array string (`["app-a","env-prod"]`),
    /// carrying body `metadata.tags` plus header values. `None` when the
    /// request carried no tags. Exact, case-sensitive membership is indexed
    /// separately in `spend_log_tags` by the spend-log write transaction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<i64>,
}

/// One durable metering event. Generate once, then retain this identity across
/// retries (including a batch retry split into individual writes). Request IDs
/// are intentionally not deduplication keys: distinct billed calls can share one.
#[derive(Debug, Clone)]
pub struct SpendLogEvent {
    pub event_id: String,
    pub log: SpendLog,
}

impl SpendLogEvent {
    pub fn new(log: SpendLog) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            log,
        }
    }
}

/// Decode all tag memberships once for both new writes and legacy backfills.
/// Repeated tags count once per request. Malformed/empty legacy JSON falls back
/// to the v9 primary column. Names remain literal, including case and escapes.
#[cfg(any(feature = "postgres", feature = "sqlite"))]
pub(crate) fn spend_tag_names(tags: Option<&str>, primary: Option<&str>) -> Vec<String> {
    let mut tags = tags
        .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
        .unwrap_or_default();
    tags.retain(|tag| !tag.is_empty());
    if tags.is_empty() {
        tags.extend(primary.filter(|tag| !tag.is_empty()).map(str::to_string));
    }
    tags.sort();
    tags.dedup();
    tags
}

/// Filter parameters for spend-log queries.
///
/// Full ISO-8601 bounds are normalized to UTC nanosecond precision. The DB
/// expands a bare `YYYY-MM-DD` end bound through the final nanosecond of that
/// day; callers may pass bare dates directly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpendFilter {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<i64>,
    /// Match spend rows carrying this exact, case-sensitive tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag_name: Option<String>,
    /// Whitelisted sort column: id | timestamp | model | provider | cost | key_id | team_id | prompt_tokens | completion_tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort_by: Option<String>,
    /// asc | desc
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort_order: Option<String>,
}

/// One day's aggregated spend: `(date, total_cost)`. The date is the
/// `YYYY-MM-DD` truncation of `spend_logs.timestamp`.
pub type DailySpend = (String, f64);

/// Aggregated spend by key from `spend_logs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendByKey {
    /// Nullable: spend rows logged without a key aggregate into a group
    /// whose `key_id` is `null` (previously collapsed to `""`).
    pub key_id: Option<String>,
    pub key_alias: Option<String>,
    pub total_spend: f64,
    pub total_requests: i64,
    pub last_used: Option<String>,
}

/// Aggregated spend by model/tag from `spend_logs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendByModel {
    pub model: Option<String>,
    pub provider: Option<String>,
    pub total_spend: f64,
    pub total_requests: i64,
}

/// Aggregated spend by tag from `spend_logs`. A multi-tag request counts
/// toward every tag it carries, so per-tag sums overlap by design.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendByTag {
    /// `None` groups rows that carried no tags at all.
    pub tag_name: Option<String>,
    pub total_spend: f64,
    pub total_requests: i64,
}

/// A bounded page of complete aggregates. Each group's totals cover the full
/// requested date range; pagination selects groups, not input spend logs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendPage<T> {
    pub data: Vec<T>,
    pub limit: i64,
    pub offset: i64,
    #[serde(default)]
    pub has_more: bool,
    pub next_offset: Option<i64>,
}

pub type SpendByKeyPage = SpendPage<SpendByKey>;
pub type SpendByModelPage = SpendPage<SpendByModel>;
pub type SpendByTagPage = SpendPage<SpendByTag>;
pub type SpendByUserPage = SpendPage<SpendByUser>;
pub type SpendByTeamPage = SpendPage<SpendByTeam>;
pub type SpendByOrgPage = SpendPage<SpendByOrg>;

/// Progress from one committed metadata-maintenance batch. The persisted
/// keyset cursor advances across restarts and wraps after the end of a sweep.
/// A contending Postgres worker can return zero rows with `has_more = true`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpendMetadataRepair {
    pub scanned_rows: usize,
    pub repaired_rows: usize,
    pub has_more: bool,
}

/// References detached by an atomic scope deletion. Returned only after the
/// transaction commits so callers can invalidate the affected auth caches.
#[derive(Debug, Default)]
pub struct ScopeDeletion {
    pub deleted: bool,
    pub key_hashes: Vec<String>,
    pub project_ids: Vec<String>,
}

/// Aggregated spend by user from `spend_logs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendByUser {
    /// Nullable: spend rows logged without a user aggregate into a group
    /// whose `user_id` is `null`.
    pub user_id: Option<String>,
    pub user_name: Option<String>,
    pub total_spend: f64,
    pub total_requests: i64,
}

/// Aggregated spend by team from `spend_logs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendByTeam {
    /// Nullable: spend rows logged without a team aggregate into a group
    /// whose `team_id` is `null`.
    pub team_id: Option<String>,
    pub team_name: Option<String>,
    pub total_spend: f64,
    pub total_requests: i64,
}

/// Aggregated spend by organization from `spend_logs`. The `org_id` is
/// resolved through the key each spend row was logged against (the `keys`
/// table carries `org_id`); rows without a key, or whose key has no org,
/// aggregate into a `null` group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendByOrg {
    pub org_id: Option<String>,
    pub total_spend: f64,
    pub total_requests: i64,
}

/// An end-user / customer row.
///
/// Tracked by the OpenAI request body `"user"` field (legacy `end_user_id`).
/// Hot-path enforcement (when a DB is configured):
/// - `blocked` → 403 at pipeline entry
/// - `model_blocklist` → 403 when the requested model matches a pattern
/// - `max_budget` → reserved/settled via `spend:customer:{end_user_id}` in
///   `BudgetHook` (same claim-based model as key/team/user scopes)
/// - `spend` → durable total advanced atomically with each spend-log insert
///
/// Requests whose body `user` does not match a customer row pass through
/// without customer-scope enforcement (no auto-create).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerRecord {
    pub end_user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_budget: Option<f64>,
    /// Running total spent by this end-user (durable DB counter). Hot-path
    /// budget enforcement uses the cache counter; this column is what
    /// `/customer/info` reports and is advanced by the spend writer.
    pub spend: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_duration: Option<String>,
    pub blocked: bool,
    /// JSON-encoded array of blocked model patterns (`*` wildcards ok).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_blocklist: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A reusable budget plan row (attachable to keys/teams in future work).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetRecord {
    pub budget_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soft_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_parallel_requests: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm_limit: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm_limit: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_duration: Option<String>,
    /// JSON object of per-model max budgets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_max_budget: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A tag row for request tagging / spend grouping.
///
/// Request traffic stamped with `x-oxllm-tag: {name}` (when a DB is configured):
/// - `max_budget` → reserved/settled via `spend:tag:{name}` in `BudgetHook`
/// - `spend` → durable total advanced atomically with each spend-log insert
///
/// Unknown tag names (header present, no row) pass through without tag-scope
/// enforcement. The `models` column is storage/metadata only today.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRecord {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON-encoded array of model names (metadata; not hot-path ACL today).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<String>,
    /// Arbitrary JSON metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_info: Option<String>,
    /// Running total spent under this tag (durable DB counter).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spend: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_budget: Option<f64>,
    pub created_at: String,
    pub updated_at: String,
}

/// A skill row — a versioned instruction file (SKILL.md content) that
/// extends a model's capabilities. The `content` holds the SKILL.md body
/// that gets injected into the system prompt when an associated model is
/// used. `model_pattern` is an optional glob (e.g. `"claude-*"`) scoping
/// the skill to matching models; `team_id` optionally scopes it to a team.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRecord {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub content: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_pattern: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}
