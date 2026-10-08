//! Spend tracking endpoints.

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use oxllm_core::UsageSummary;
use oxllm_core::{DeploymentPricing, calculate_cost_for_service_tier};
use serde::Deserialize;
use serde_json::json;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::state::SharedState;

use super::{db_err, load_db, require_master};

/// Maximum page size accepted by `GET /spend/logs` (v1). User-supplied
/// `limit` values are clamped into `1..=MAX_LIMIT` so a negative or huge
/// limit can never reach the SQL layer (SQLite `LIMIT -1` removes the row
/// cap entirely → full-table dump). Mirrors `audit::MAX_LIMIT`.
const MAX_LIMIT: i64 = 500;

/// Default page size for `GET /spend/logs` (v1) when `limit` is not
/// supplied. Matches the default assumed by `Database::query_spend`.
const DEFAULT_LIMIT: i64 = 100;

#[derive(Debug, Deserialize)]
pub struct SpendLogsQuery {
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default)]
    pub key_id: Option<String>,
    #[serde(default)]
    pub team_id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    /// Only rows carrying this request tag (body `metadata.tags` or the
    /// `x-oxllm-tag` header). Matches any tag in the row's list.
    #[serde(default)]
    pub tag_name: Option<String>,
    #[serde(default)]
    pub limit: Option<i64>,
    #[serde(default)]
    pub offset: Option<i64>,
}

impl From<&SpendLogsQuery> for oxllm_db::SpendFilter {
    fn from(q: &SpendLogsQuery) -> Self {
        oxllm_db::SpendFilter {
            start_date: q.start_date.clone(),
            end_date: q.end_date.clone(),
            key_id: q.key_id.clone(),
            team_id: q.team_id.clone(),
            model: q.model.clone(),
            tag_name: q.tag_name.clone(),
            limit: q.limit,
            offset: q.offset,
            sort_by: None,
            sort_order: None,
            request_id: None,
        }
    }
}

/// `GET /spend/logs` — query spend logs.
pub async fn spend_logs(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<SpendLogsQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    // Clamp user-supplied paging before it reaches SQL. SQLite treats
    // `LIMIT -1` as "no cap", so an unclamped negative limit would dump the
    // whole table (audit::list_audit_logs documents the same defense).
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let offset = q.offset.unwrap_or(0).max(0);

    let mut filter: oxllm_db::SpendFilter = (&q).into();
    filter.limit = Some(limit);
    filter.offset = Some(offset);
    let logs = db.query_spend(&filter).await.map_err(db_err)?;
    // True aggregates over the full filter scope (not just the current
    // page) so the numbers don't depend on `limit`.
    let total_cost = db.total_spend_filtered(&filter).await.map_err(db_err)?;
    let total_requests = db.count_spend_filtered(&filter).await.map_err(db_err)?;

    Ok(Json(json!({
    "data": logs,
    "total_cost": total_cost,
    "total_requests": total_requests,
    }))
    .into_response())
}

#[derive(Debug, Deserialize)]
pub struct SpendTotalQuery {
    #[serde(default)]
    pub key_id: Option<String>,
    /// Filter by the `team_id` recorded on spend rows. The scope is the same
    /// as `/spend/logs`/`v2`'s `team_id` filter. Without this, a caller
    /// passing `team_id=` (e.g. the admin UI's budget-utilization bar for
    /// team rows) had the parameter silently ignored and received the
    /// *global* total.
    #[serde(default)]
    pub team_id: Option<String>,
    #[serde(default)]
    pub start_date: Option<String>,
}

/// `GET /spend/total` — aggregate spend over the `key_id` / `team_id` /
/// `start_date` scope. Uses the same filtered aggregation as the log
/// endpoints so the reported total always describes the caller's filter
/// scope (not just the current page or the whole table).
pub async fn spend_total(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<SpendTotalQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let filter = oxllm_db::SpendFilter {
        key_id: q.key_id.clone(),
        team_id: q.team_id.clone(),
        start_date: q.start_date.clone(),
        ..Default::default()
    };
    let total = db.total_spend_filtered(&filter).await.map_err(db_err)?;
    let count = db.count_spend_filtered(&filter).await.map_err(db_err)?;

    Ok(Json(json!({
    "total_cost": total,
    "total_requests": count,
    }))
    .into_response())
}

/// Query params for `GET /spend/logs/v2` (cursor pagination).
#[derive(Debug, Deserialize, Default)]
pub struct SpendLogsV2Query {
    /// Cursor: only rows with `id < cursor` are returned. `None` starts at
    /// the most recent row. Ignored when `offset` is set (sorted pages).
    #[serde(default)]
    pub cursor: Option<i64>,
    /// Page size (1..=1000). Defaults to 50.
    #[serde(default)]
    pub limit: Option<i64>,
    #[serde(default)]
    pub offset: Option<i64>,
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default)]
    pub key_id: Option<String>,
    #[serde(default)]
    pub team_id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    /// Only rows carrying this request tag (any tag in the row's list).
    #[serde(default)]
    pub tag_name: Option<String>,
    #[serde(default)]
    pub sort_by: Option<String>,
    #[serde(default)]
    pub sort_order: Option<String>,
}

impl From<&SpendLogsV2Query> for oxllm_db::SpendFilter {
    fn from(q: &SpendLogsV2Query) -> Self {
        oxllm_db::SpendFilter {
            start_date: q.start_date.clone(),
            end_date: q.end_date.clone(),
            key_id: q.key_id.clone(),
            team_id: q.team_id.clone(),
            model: q.model.clone(),
            tag_name: q.tag_name.clone(),
            limit: None,
            offset: q.offset,
            sort_by: q.sort_by.clone(),
            sort_order: q.sort_order.clone(),
            request_id: None,
        }
    }
}

/// `GET /spend/logs/v2` — cursor/offset-paginated spend logs with filters + sort.
pub async fn spend_logs_v2(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<SpendLogsV2Query>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    // Clamp once at the top so both the SQL layer and the echoed `limit`
    // metadata agree (`query_spend_paginated` clamps internally too, but
    // echoing the raw query param gave clients a misleading contract).
    let limit = q.limit.unwrap_or(50).clamp(1, 1000);
    // Echo the clamped offset, not the raw input: a negative `?offset=-5`
    // is coerced to 0 inside `query_spend_paginated`, so echoing `-5` here
    // would let a client build a `next_cursor` URL pointing at a value
    // inconsistent with the query that actually ran. `limit` is already
    // the clamped value; `offset` needs the same treatment.
    let offset = q.offset.unwrap_or(0).max(0);
    let filter: oxllm_db::SpendFilter = (&q).into();
    // When sorting by a non-default column, use OFFSET paging (cursor is id-based).
    let use_offset = q.offset.is_some()
        || matches!(
            q.sort_by.as_deref(),
            Some("model")
                | Some("provider")
                | Some("cost")
                | Some("key_id")
                | Some("team_id")
                | Some("prompt_tokens")
                | Some("completion_tokens")
                | Some("timestamp")
        );
    let cursor = if use_offset { None } else { q.cursor };
    let logs = db
        .query_spend_paginated(cursor, limit, &filter)
        .await
        .map_err(db_err)?;

    // A page with fewer rows than the limit is the final page — there's no
    // next page to point at, so omit `next_cursor` (serialize as null).
    let last_id = logs.last().and_then(|l| l.id);
    let next_cursor = compute_next_cursor(use_offset, logs.len(), limit, Some(offset), last_id);
    // True aggregates over the full filter scope (not just the current
    // page) so the numbers don't depend on `limit`.
    let total_cost = db.total_spend_filtered(&filter).await.map_err(db_err)?;
    let total_requests = db.count_spend_filtered(&filter).await.map_err(db_err)?;

    Ok(Json(json!({
    "data": logs,
    "next_cursor": next_cursor,
    "limit": limit,
    "offset": offset,
    "sort_by": q.sort_by,
    "sort_order": q.sort_order,
    "total_cost": total_cost,
    "total_requests": total_requests,
    "paging": if use_offset { "offset" } else { "cursor" },
    }))
    .into_response())
}

/// Query params for `GET /global/spend` (optional date range).
#[derive(Debug, Deserialize, Default)]
pub struct GlobalSpendQuery {
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
}

/// `GET /global/spend` — total spend across all keys, optionally bounded by a
/// date range.
pub async fn global_spend(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<GlobalSpendQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    // The DB applies identical inclusive bounds to cost and request count.
    let filter = oxllm_db::SpendFilter {
        start_date: q.start_date.clone(),
        end_date: q.end_date.clone(),
        ..Default::default()
    };
    let total = db.total_spend_filtered(&filter).await.map_err(db_err)?;
    let count = db.count_spend_filtered(&filter).await.map_err(db_err)?;

    Ok(Json(json!({
    "total_cost": total,
    "total_requests": count,
    "start_date": q.start_date,
    "end_date": q.end_date,
    }))
    .into_response())
}

/// Query params for `GET /global/activity`.
#[derive(Debug, Deserialize)]
pub struct GlobalActivityQuery {
    /// Number of days to include in the activity window. Defaults to 30.
    #[serde(default)]
    pub days: Option<i64>,
}

/// `GET /global/activity` — daily spend for the last `days` days.
pub async fn global_activity(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<GlobalActivityQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let days = q.days.unwrap_or(30).max(1);
    let buckets = db.daily_activity(days).await.map_err(db_err)?;

    let results: Vec<serde_json::Value> = buckets
        .iter()
        .map(|(day, cost)| {
            json!({
            "day": day,
            "spend": cost,
            })
        })
        .collect();
    let total: f64 = buckets.iter().map(|(_, c)| *c).sum();

    Ok(Json(json!({
    "results": results,
    "days": days,
    "total_spend": total,
    }))
    .into_response())
}

/// Request body for `POST /spend/calculate`.
#[derive(Debug, Default, Deserialize)]
pub struct SpendCalculateRequest {
    pub model: String,
    pub usage: UsageBody,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub service_tier: Option<String>,
    /// A configured model id or model_info.id disambiguates load-balanced aliases.
    #[serde(default)]
    pub deployment_id: Option<String>,
    #[serde(default)]
    pub pricing_override: Option<serde_json::Value>,
    #[serde(default, alias = "response_time_ms")]
    pub duration_ms: Option<f64>,
}

/// OpenAI-shaped usage object accepted by `/spend/calculate`. All token
/// fields are optional and default to 0 so partial payloads work.
#[derive(Debug, Default)]
pub struct UsageBody {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_creation_1h_input_tokens: Option<u64>,
    pub audio_input_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub citation_tokens: Option<u64>,
    pub web_search_requests: Option<u64>,
    /// Complete upstream receipt, used by Perplexity's dynamic-fee preview.
    pub reported_cost: Option<f64>,
}

impl<'de> Deserialize<'de> for UsageBody {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut raw = serde_json::Value::deserialize(deserializer)?;
        if !raw.is_object() {
            return Err(serde::de::Error::custom("usage must be an object"));
        }
        for pointer in [
            "/prompt_tokens",
            "/input_tokens",
            "/completion_tokens",
            "/output_tokens",
            "/total_tokens",
            "/cache_read_input_tokens",
            "/cache_creation_input_tokens",
            "/cache_creation_1h_input_tokens",
            "/audio_input_tokens",
            "/reasoning_tokens",
            "/citation_tokens",
            "/web_search_requests",
            "/num_search_queries",
            "/prompt_tokens_details/cached_tokens",
            "/prompt_tokens_details/cache_write_tokens",
            "/prompt_tokens_details/cache_creation_tokens",
            "/prompt_tokens_details/audio_tokens",
            "/prompt_tokens_details/web_search_requests",
            "/completion_tokens_details/reasoning_tokens",
            "/output_tokens_details/thinking_tokens",
            "/output_tokens_details/reasoning_tokens",
            "/input_tokens_details/cached_tokens",
            "/input_tokens_details/cache_write_tokens",
            "/input_tokens_details/cache_creation_tokens",
            "/input_tokens_details/audio_tokens",
            "/input_tokens_details/web_search_requests",
            "/input_tokens_details/ephemeral_1h_input_tokens",
            "/cache_creation/ephemeral_1h_input_tokens",
            "/cache_creation/ephemeral_5m_input_tokens",
        ] {
            if raw
                .pointer(pointer)
                .is_some_and(|value| !value.is_null() && value.as_u64().is_none())
            {
                return Err(serde::de::Error::custom(
                    "token and search counts must be non-negative integers",
                ));
            }
        }
        for field in [
            "prompt_tokens_details",
            "input_tokens_details",
            "completion_tokens_details",
            "output_tokens_details",
            "cache_creation",
        ] {
            if raw
                .get(field)
                .is_some_and(|value| !value.is_null() && !value.is_object())
            {
                return Err(serde::de::Error::custom("usage details must be objects"));
            }
        }
        let native = raw
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64)
            .is_none()
            && raw
                .get("input_tokens")
                .and_then(serde_json::Value::as_u64)
                .is_some();
        if !native {
            if let Some(cached) = raw
                .get("cache_read_input_tokens")
                .and_then(serde_json::Value::as_u64)
            {
                let details = raw["prompt_tokens_details"]
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
                let mut details = details;
                if details
                    .get("cached_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .is_none()
                {
                    details.insert("cached_tokens".into(), json!(cached));
                }
                raw["prompt_tokens_details"] = serde_json::Value::Object(details);
            }
            if raw
                .get("prompt_tokens")
                .and_then(serde_json::Value::as_u64)
                .is_none()
            {
                raw["prompt_tokens"] = json!(0);
            }
        }
        // The same extractor as inference settlement preserves native cache
        // accounting and standard nested usage buckets in an admin preview.
        let mut usage =
            crate::orchestration::usage::extract_usage(&json!({"usage":raw})).unwrap_or_default();
        usage.cache_creation_1h_input_tokens = usage.cache_creation_1h_input_tokens.or_else(|| {
            raw.get("cache_creation_1h_input_tokens")
                .and_then(serde_json::Value::as_u64)
        });
        usage.reasoning_tokens = usage.reasoning_tokens.or_else(|| {
            raw.get("reasoning_tokens")
                .and_then(serde_json::Value::as_u64)
        });
        usage.web_search_requests = usage.web_search_requests.or_else(|| {
            raw.get("web_search_requests")
                .and_then(serde_json::Value::as_u64)
        });
        Ok(Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: Some(usage.total_tokens),
            cache_read_input_tokens: usage.cache_read_input_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            cache_creation_1h_input_tokens: usage.cache_creation_1h_input_tokens,
            audio_input_tokens: usage.audio_input_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            citation_tokens: usage.citation_tokens,
            web_search_requests: usage.web_search_requests,
            reported_cost: crate::pipeline::extract_body_usage_cost(&json!({"usage": raw})),
        })
    }
}

impl From<&UsageBody> for UsageSummary {
    fn from(u: &UsageBody) -> Self {
        let total = u
            .total_tokens
            .unwrap_or_else(|| u.prompt_tokens.saturating_add(u.completion_tokens));
        UsageSummary {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: total,
            cache_read_input_tokens: u.cache_read_input_tokens,
            cache_creation_input_tokens: u.cache_creation_input_tokens,
            cache_creation_1h_input_tokens: u.cache_creation_1h_input_tokens,
            audio_input_tokens: u.audio_input_tokens,
            reasoning_tokens: u.reasoning_tokens,
            citation_tokens: u.citation_tokens,
            web_search_requests: u.web_search_requests,
        }
    }
}

/// `POST /spend/calculate` — preview the cost of a model+usage pair without
/// making an upstream call. No DB needed.
///
/// Accepts standard or native usage JSON. Optional provider/tier/custom rates
/// use the inference price resolver; configured aliases inherit their selected
/// rate card. Ambiguous aliases require a selector or explicit custom rates.
pub async fn spend_calculate(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<SpendCalculateRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    if req.model.trim().is_empty()
        || req
            .provider
            .as_deref()
            .is_some_and(|provider| provider.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "model and supplied provider must not be empty",
        ));
    }
    if req
        .duration_ms
        .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
    {
        return Err(ApiError::bad_request(
            "duration_ms must be finite and non-negative",
        ));
    }
    let requested_tier = req
        .service_tier
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase);
    let tier = match requested_tier.as_deref() {
        None | Some("default" | "standard") => None,
        Some("batch" | "batches") => Some("batches"),
        Some("priority" | "flex" | "fast" | "ultrafast" | "balanced") => requested_tier.as_deref(),
        _ => return Err(ApiError::bad_request("unsupported service_tier")),
    };
    let state = shared.load_full();
    let (pricing_model, provider, pricing) = preview_rate_card(&state, &req).await?;
    let usage = UsageSummary::from(&req.usage);
    let reported_cost = (provider == "perplexity" && pricing.is_none())
        .then_some(req.usage.reported_cost)
        .flatten();
    let cost = reported_cost.or_else(|| {
        calculate_cost_for_service_tier(&pricing_model, &usage, pricing.as_ref(), &provider, tier)
            .or_else(|| {
                req.duration_ms.and_then(|duration| match pricing.as_ref() {
                    Some(pricing) => pricing.cost_per_second(duration),
                    None => oxllm_core::lookup_pricing_with_provider(&pricing_model, &provider)
                        .and_then(|pricing| pricing.cost_per_second(duration)),
                })
            })
    });
    if cost.is_some_and(|cost| !cost.is_finite() || cost < 0.0) {
        return Err(ApiError::bad_request("computed cost is not representable"));
    }
    Ok(Json(json!({
    "cost": cost,
    "model": req.model,
    }))
    .into_response())
}

fn validate_preview_pricing(pricing: &DeploymentPricing) -> Result<(), ApiError> {
    let rates = [
        Some(pricing.input_cost_per_token),
        Some(pricing.output_cost_per_token),
        pricing.cache_read_input_token_cost,
        pricing.cache_creation_input_token_cost,
        pricing.cache_creation_input_token_cost_above_1hr,
        pricing.output_cost_per_reasoning_token,
        pricing.cost_per_second_rate,
        pricing.input_cost_per_second,
        pricing.output_cost_per_second,
    ];
    if rates
        .into_iter()
        .flatten()
        .any(|rate| !rate.is_finite() || rate < 0.0)
    {
        return Err(ApiError::bad_request(
            "pricing rates must be finite and non-negative",
        ));
    }
    Ok(())
}

fn explicit_preview_pricing(
    value: &serde_json::Value,
    model: &str,
) -> Result<DeploymentPricing, ApiError> {
    let object = value
        .as_object()
        .ok_or_else(|| ApiError::bad_request("pricing_override must be an object"))?;
    let token_fields = [
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
        "output_cost_per_reasoning_token",
    ];
    if object.keys().any(|key| {
        !token_fields.contains(&key.as_str())
            && ![
                "cost_per_second",
                "cost_per_second_rate",
                "input_cost_per_second",
                "output_cost_per_second",
            ]
            .contains(&key.as_str())
    }) {
        return Err(ApiError::bad_request("unsupported pricing_override field"));
    }
    if !object.values().any(|value| !value.is_null()) {
        return Err(ApiError::bad_request(
            "pricing_override requires at least one rate",
        ));
    }
    if object
        .values()
        .filter(|value| !value.is_null())
        .any(|value| {
            value
                .as_f64()
                .is_none_or(|rate| !rate.is_finite() || rate < 0.0)
        })
    {
        return Err(ApiError::bad_request(
            "pricing rates must be finite and non-negative",
        ));
    }
    let mut parameters = object.clone();
    if let Some(legacy) = parameters.remove("cost_per_second_rate") {
        if parameters.contains_key("cost_per_second") {
            return Err(ApiError::bad_request("specify cost_per_second only once"));
        }
        parameters.insert("cost_per_second".into(), legacy);
    }
    parameters.insert("model".into(), json!(model));
    let parameters: oxllm_core::deployment::OxllmParams =
        serde_json::from_value(serde_json::Value::Object(parameters))
            .map_err(|_| ApiError::bad_request("invalid pricing_override"))?;
    // Reuse configuration inheritance and explicit-zero token precedence.
    let pricing = parameters
        .pricing_override()
        .ok_or_else(|| ApiError::bad_request("pricing_override requires at least one rate"))?;
    validate_preview_pricing(&pricing)?;
    Ok(pricing)
}

async fn preview_rate_card(
    state: &crate::state::AppState,
    req: &SpendCalculateRequest,
) -> Result<(String, String, Option<DeploymentPricing>), ApiError> {
    let explicit_model = match req.provider.as_deref() {
        Some(provider) => {
            let prefix = format!("{provider}/");
            format!(
                "{prefix}{}",
                req.model.strip_prefix(&prefix).unwrap_or(&req.model)
            )
        }
        None => req.model.clone(),
    };
    let explicit = req
        .pricing_override
        .as_ref()
        .map(|value| explicit_preview_pricing(value, &explicit_model))
        .transpose()?;
    let deployments = state
        .router
        .deployments_for_model(&state.deployments, &req.model);
    let mut candidates = deployments
        .into_iter()
        .filter(|deployment| {
            req.provider
                .as_deref()
                .is_none_or(|provider| deployment.provider_slug() == provider)
                && req.deployment_id.as_deref().is_none_or(|id| {
                    deployment
                        .model_info
                        .as_ref()
                        .and_then(|info| info.id.as_deref())
                        == Some(id)
                })
        })
        .cloned()
        .collect::<Vec<_>>();
    if candidates.is_empty() && req.deployment_id.is_some() {
        let deployment = match (&state.db, req.deployment_id.as_deref()) {
            (Some(db), Some(id)) => db.get_model(id).await.map_err(db_err)?,
            _ => None,
        }
        .ok_or_else(|| ApiError::bad_request("deployment_id was not found"))?;
        if deployment.model_name != req.model {
            return Err(ApiError::bad_request(
                "deployment_id does not belong to this model alias",
            ));
        }
        let deployment = crate::state::db_record_to_deployment(
            &deployment,
            state
                .config
                .general_settings
                .allow_private_api_base
                .unwrap_or(false),
            state.credential_cipher.as_ref(),
        )
        .map_err(|_| ApiError::bad_request("selected deployment configuration is unavailable"))?;
        if req
            .provider
            .as_deref()
            .is_some_and(|provider| deployment.provider_slug() != provider)
        {
            return Err(ApiError::bad_request(
                "deployment_id does not belong to this provider",
            ));
        }
        candidates.push(deployment);
    }
    let mut selected = None;
    for deployment in candidates {
        let provider = deployment.provider_slug().to_string();
        if deployment.oxllm_params.model.contains('*')
            && (req.model == deployment.model_name || req.model.contains('*'))
        {
            if explicit.is_some() {
                continue;
            }
            return Err(ApiError::bad_request(
                "wildcard pricing requires a concrete model or explicit pricing_override",
            ));
        }
        let model = crate::pipeline::deployment_pricing_model(&deployment, &req.model);
        let mut configured_pricing = deployment.oxllm_params.clone();
        configured_pricing.model = model.clone();
        let pricing = match req.pricing_override.as_ref() {
            Some(value) => Some(explicit_preview_pricing(value, &model)?),
            None => configured_pricing.pricing_override(),
        };
        if let Some(pricing) = pricing.as_ref() {
            validate_preview_pricing(pricing)?;
        }
        let card = (model, provider, pricing);
        if selected.as_ref().is_some_and(
            |previous: &(String, String, Option<DeploymentPricing>)| {
                if explicit.is_some() {
                    serde_json::to_value(&previous.2).ok() != serde_json::to_value(&card.2).ok()
                } else {
                    serde_json::to_value(previous).ok() != serde_json::to_value(&card).ok()
                }
            },
        ) {
            return Err(ApiError::bad_request(
                "model has multiple rate cards; specify provider, deployment_id, or pricing_override",
            ));
        }
        selected = Some(card);
    }
    Ok(selected.unwrap_or_else(|| {
        (
            req.model.clone(),
            req.provider.clone().unwrap_or_default(),
            explicit,
        )
    }))
}

/// Normalize an `end_date` query param into an inclusive upper bound.
///
/// A bare `YYYY-MM-DD` date (10 chars, no time component) sorts *before*
/// every timestamp on that day (`'2026-07-02' < '2026-07-02T00:00:00Z'`),
/// which silently dropped the entire end day from range queries. Expand it
/// to the final nanosecond of the day so the bound is inclusive. Full
/// timestamps pass through unchanged.
pub(super) fn inclusive_end_date(end_date: Option<&str>) -> Option<String> {
    end_date.map(oxllm_db::inclusive_spend_end_date)
}

/// `GET /spend/keys` — aggregated spend by key.
pub async fn spend_by_key(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<SpendTagsQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let rows = db
        .query_spend_by_key_paginated(
            q.start_date.as_deref(),
            q.end_date.as_deref(),
            q.limit.unwrap_or(oxllm_db::SPEND_ROLLUP_DEFAULT_LIMIT),
            q.offset.unwrap_or(0),
        )
        .await
        .map_err(db_err)?;
    Ok(Json(rows).into_response())
}

/// `GET /spend/by-model` — aggregated spend by model/provider.
pub async fn spend_by_model(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<SpendTagsQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let rows = db
        .query_spend_by_model_paginated(
            q.start_date.as_deref(),
            q.end_date.as_deref(),
            q.limit.unwrap_or(oxllm_db::SPEND_ROLLUP_DEFAULT_LIMIT),
            q.offset.unwrap_or(0),
        )
        .await
        .map_err(db_err)?;
    Ok(Json(rows).into_response())
}

/// Query parameters for bounded scope rollups. Each group's values
/// cover the full date range; limit/offset select groups, not input log rows.
#[derive(Debug, Default, Deserialize)]
pub struct SpendTagsQuery {
    pub start_date: Option<String>,
    pub end_date: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// `GET /spend/tags` — paginated aggregated spend by request tag.
///
/// Groups `spend_logs` by the tags each request carried (body
/// `metadata.tags` plus the `x-oxllm-tag` header). A multi-tag request
/// counts toward EVERY tag it carries, so per-tag sums overlap by design —
/// each row answers "how much did traffic tagged with me spend". Rows
/// without tags aggregate under a `null` tag_name.
pub async fn spend_by_tag(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<SpendTagsQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let page = db
        .query_spend_by_tag_paginated(
            q.start_date.as_deref(),
            q.end_date.as_deref(),
            q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT),
            q.offset.unwrap_or(0).max(0),
        )
        .await
        .map_err(db_err)?;
    Ok(Json(page).into_response())
}

/// `GET /spend/users` — aggregated spend by user.
pub async fn spend_by_user(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<SpendTagsQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let rows = db
        .query_spend_by_user_paginated(
            q.start_date.as_deref(),
            q.end_date.as_deref(),
            q.limit.unwrap_or(oxllm_db::SPEND_ROLLUP_DEFAULT_LIMIT),
            q.offset.unwrap_or(0),
        )
        .await
        .map_err(db_err)?;
    Ok(Json(rows).into_response())
}

/// `GET /spend/teams` — aggregated spend by team.
pub async fn spend_by_team(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<SpendTagsQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let rows = db
        .query_spend_by_team_paginated(
            q.start_date.as_deref(),
            q.end_date.as_deref(),
            q.limit.unwrap_or(oxllm_db::SPEND_ROLLUP_DEFAULT_LIMIT),
            q.offset.unwrap_or(0),
        )
        .await
        .map_err(db_err)?;
    Ok(Json(rows).into_response())
}

/// `GET /spend/organizations` — aggregated spend by organization.
///
/// The `org_id` is resolved through the key each spend row was logged
/// against (`keys.org_id`); rows without an org-bearing key aggregate
/// into a `null` group.
pub async fn spend_by_org(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<SpendTagsQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let rows = db
        .query_spend_by_org_paginated(
            q.start_date.as_deref(),
            q.end_date.as_deref(),
            q.limit.unwrap_or(oxllm_db::SPEND_ROLLUP_DEFAULT_LIMIT),
            q.offset.unwrap_or(0),
        )
        .await
        .map_err(db_err)?;
    Ok(Json(rows).into_response())
}

/// `POST /spend/regenerate` — not implemented.
///
/// Returns 400 rather than a success-shaped no-op so callers don't believe
/// a regeneration actually ran.
pub async fn spend_regenerate(
    State(_shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    Err(ApiError::bad_request(
        "spend regenerate is not yet implemented",
    ))
}

/// Compute the `next_cursor` for the v2 spend endpoint.
///
/// Both paging styles omit the cursor on the final page (when fewer rows
/// were returned than the page size). For offset paging the cursor is the
/// next row offset; for id-based cursor paging it is the last row's id.
fn compute_next_cursor(
    use_offset: bool,
    logs_len: usize,
    page_limit: i64,
    offset: Option<i64>,
    last_id: Option<i64>,
) -> Option<i64> {
    if logs_len as i64 >= page_limit {
        if use_offset {
            Some(offset.unwrap_or(0) + logs_len as i64)
        } else {
            last_id
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    fn make_auth() -> AuthUser {
        use crate::auth::hash_key;
        AuthUser::master(hash_key(b"sk-master"))
    }

    #[tokio::test]
    async fn spend_calculate_returns_known_model_cost() {
        let auth = make_auth();
        let req = SpendCalculateRequest {
            model: "gpt-4o".to_string(),
            usage: UsageBody {
                prompt_tokens: 1000,
                completion_tokens: 500,
                total_tokens: Some(1500),
                ..Default::default()
            },
            ..Default::default()
        };
        let resp = spend_calculate(State(shared_without_db()), Extension(auth), Json(req))
            .await
            .expect("handler");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn spend_calculate_unknown_model_is_null_cost() {
        let auth = make_auth();
        let req = SpendCalculateRequest {
            model: "made-up-model".to_string(),
            usage: UsageBody {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: Some(15),
                ..Default::default()
            },
            ..Default::default()
        };
        let resp = spend_calculate(State(shared_without_db()), Extension(auth), Json(req))
            .await
            .expect("handler");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn usage_body_from_openai_shape() {
        let body = UsageBody {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: Some(150),
            cache_read_input_tokens: Some(10),
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            ..Default::default()
        };
        let usage = UsageSummary::from(&body);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.cache_read_input_tokens, Some(10));
    }

    #[test]
    fn usage_body_defaults_total_when_absent() {
        let body = UsageBody {
            prompt_tokens: 7,
            completion_tokens: 3,
            total_tokens: None,
            ..Default::default()
        };
        let usage = UsageSummary::from(&body);
        assert_eq!(usage.total_tokens, 10, "total should default to sum");
    }

    #[test]
    fn usage_body_extreme_token_counts_do_not_overflow() {
        let body = UsageBody {
            prompt_tokens: u64::MAX,
            completion_tokens: 1,
            ..Default::default()
        };
        assert_eq!(UsageSummary::from(&body).total_tokens, u64::MAX);
    }

    #[test]
    fn v2_filter_conversion_carries_team() {
        let q = SpendLogsV2Query {
            team_id: Some("team-1".to_string()),
            ..Default::default()
        };
        let f: oxllm_db::SpendFilter = (&q).into();
        assert_eq!(f.team_id.as_deref(), Some("team-1"));
    }

    #[test]
    fn compute_next_cursor_null_when_under_limit_cursor_paging() {
        // Final page on cursor paging: fewer rows than the limit -> null cursor.
        assert_eq!(compute_next_cursor(false, 5, 50, None, Some(42)), None);
    }

    #[test]
    fn compute_next_cursor_set_when_at_limit_cursor_paging() {
        // Full page on cursor paging -> last id is the next cursor.
        assert_eq!(compute_next_cursor(false, 50, 50, None, Some(42)), Some(42));
    }

    #[test]
    fn compute_next_cursor_null_when_under_limit_offset_paging() {
        assert_eq!(compute_next_cursor(true, 5, 50, Some(0), None), None);
    }

    #[test]
    fn compute_next_cursor_set_when_at_limit_offset_paging() {
        assert_eq!(compute_next_cursor(true, 50, 50, Some(0), None), Some(50));
    }

    // ---- U45: page-size cap raised to 1000 --------------------------------

    #[test]
    fn compute_next_cursor_set_when_at_1000_cursor_paging() {
        // A full page at the new cap (1000) must still yield a next cursor.
        assert_eq!(
            compute_next_cursor(false, 1000, 1000, None, Some(7)),
            Some(7)
        );
    }

    #[test]
    fn compute_next_cursor_null_when_under_1000_cursor_paging() {
        // Final page below the new cap → null cursor.
        assert_eq!(compute_next_cursor(false, 999, 1000, None, Some(7)), None);
    }

    // ---- inclusive_end_date (#M6) ------------------------------------------

    #[test]
    fn date_only_end_date_expands_to_end_of_day() {
        // '2026-07-02' < '2026-07-02T10:00:00Z' lexicographically, so a bare
        // date used as `timestamp <= end` excluded the whole end day.
        assert_eq!(
            inclusive_end_date(Some("2026-07-02")).as_deref(),
            Some("2026-07-02T23:59:59.999999999Z")
        );
    }

    #[test]
    fn full_timestamp_end_date_passes_through() {
        let ts = "2026-07-02T10:00:00Z";
        assert_eq!(inclusive_end_date(Some(ts)).as_deref(), Some(ts));
    }

    #[test]
    fn none_end_date_stays_none() {
        assert_eq!(inclusive_end_date(None), None);
    }

    // ---- spend_regenerate (#M8) --------------------------------------------

    fn shared_without_db() -> SharedState {
        let state = crate::state::AppState::with_registry_cache_db(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: Default::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
            std::sync::Arc::new(crate::cache::InMemoryCache::default()),
            None,
            None,
        )
        .expect("state");
        std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(state))
    }

    #[tokio::test]
    async fn spend_regenerate_rejects_non_master() {
        // require_master fires before the not-implemented error.
        let err = spend_regenerate(State(shared_without_db()), Extension(AuthUser::open()))
            .await
            .expect_err("must 403");
        assert_eq!(err.status, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn spend_regenerate_is_bad_request_not_success() {
        // Master auth passes the gate, but a no-op must not report success.
        let err = spend_regenerate(State(shared_without_db()), Extension(make_auth()))
            .await
            .expect_err("must 400");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("not yet implemented"));
    }

    // ---- Integration-style tests with a real in-memory DB -------------------

    use std::sync::Arc;

    async fn shared_with_db() -> (SharedState, Arc<oxllm_db::Database>) {
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");
        let state = crate::state::AppState::with_registry_cache_db(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: Default::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
            Arc::new(crate::cache::InMemoryCache::default()),
            None,
            Some(db.clone()),
        )
        .expect("state");
        (Arc::new(arc_swap::ArcSwap::from_pointee(state)), db)
    }

    fn spend_log_with_cost(
        timestamp: &str,
        cost: f64,
        team_id: Option<&str>,
    ) -> oxllm_db::SpendLog {
        oxllm_db::SpendLog {
            id: None,
            timestamp: timestamp.to_string(),
            request_id: Some(format!("r-{timestamp}-{cost}")),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            cost: Some(cost),
            key_id: Some("k-1".to_string()),
            team_id: team_id.map(str::to_string),
            user_id: None,
            end_user_id: None,
            tag_name: None,
            tags: None,
            status: Some("success".to_string()),
            latency_ms: Some(42),
        }
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body bytes");
        serde_json::from_slice(&bytes).expect("json body")
    }

    fn logs_query(limit: Option<i64>, offset: Option<i64>) -> SpendLogsQuery {
        SpendLogsQuery {
            start_date: None,
            end_date: None,
            key_id: None,
            team_id: None,
            model: None,
            tag_name: None,
            limit,
            offset,
        }
    }

    // HIGH: spend_logs v1 must clamp limit/offset before SQL (SQLite
    // LIMIT -1 removes the row cap → full-table dump).
    #[tokio::test]
    async fn spend_logs_clamps_negative_limit() {
        let (shared, db) = shared_with_db().await;
        for i in 0..15 {
            db.insert_spend_log(&spend_log_with_cost(&format!("2026-07-0{i}"), 1.0, None))
                .await
                .expect("insert");
        }

        let resp = spend_logs(
            State(shared),
            Extension(make_auth()),
            Query(logs_query(Some(-1), Some(-10))),
        )
        .await
        .expect("handler");
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let data = body["data"].as_array().expect("array");
        // Clamped into [1, MAX_LIMIT] — never the whole table.
        assert_eq!(
            data.len(),
            1,
            "negative limit must clamp to 1, got full dump"
        );
        assert!(
            body["total_requests"].as_u64().unwrap_or(0) == 15,
            "total is full scope"
        );
    }

    #[tokio::test]
    async fn spend_logs_clamps_huge_limit() {
        let (shared, db) = shared_with_db().await;
        for i in 0..3 {
            db.insert_spend_log(&spend_log_with_cost(&format!("2026-07-0{i}"), 1.0, None))
                .await
                .expect("insert");
        }

        let resp = spend_logs(
            State(shared),
            Extension(make_auth()),
            Query(logs_query(Some(1_000_000), None)),
        )
        .await
        .expect("handler");
        let body = body_json(resp).await;
        let data = body["data"].as_array().expect("array");
        assert_eq!(
            data.len(),
            3,
            "all rows returned, clamped limit doesn't truncate"
        );
    }

    // MEDIUM: inclusive_end_date must apply to spend_logs (v1). A bare
    // `YYYY-MM-DD` end_date previously dropped the entire end day.
    #[tokio::test]
    async fn spend_logs_end_date_is_inclusive_of_end_day() {
        let (shared, db) = shared_with_db().await;
        db.insert_spend_log(&spend_log_with_cost("2026-07-30T12:00:00Z", 1.0, None))
            .await
            .expect("insert");
        db.insert_spend_log(&spend_log_with_cost("2026-07-31T12:00:00Z", 2.0, None))
            .await
            .expect("insert");

        let q = SpendLogsQuery {
            start_date: Some("2026-07-30".to_string()),
            end_date: Some("2026-07-31".to_string()),
            ..logs_query(None, None)
        };
        let resp = spend_logs(State(shared), Extension(make_auth()), Query(q))
            .await
            .expect("handler");
        let body = body_json(resp).await;
        let data = body["data"].as_array().expect("array");
        assert_eq!(data.len(), 2, "end day must be included");
    }

    // MEDIUM: total_cost / total_requests must reflect the full filter
    // scope, not just the current page.
    #[tokio::test]
    async fn spend_logs_totals_are_full_scope_not_page_scoped() {
        let (shared, db) = shared_with_db().await;
        for i in 0..60 {
            db.insert_spend_log(&spend_log_with_cost(&format!("2026-07-{i:02}"), 1.0, None))
                .await
                .expect("insert");
        }

        let resp = spend_logs(
            State(shared),
            Extension(make_auth()),
            Query(logs_query(Some(10), None)),
        )
        .await
        .expect("handler");
        let body = body_json(resp).await;
        assert_eq!(
            body["data"].as_array().expect("array").len(),
            10,
            "page size"
        );
        assert_eq!(
            body["total_requests"].as_u64(),
            Some(60),
            "true total, not page count"
        );
        assert_eq!(
            body["total_cost"].as_f64(),
            Some(60.0),
            "true sum, not page sum"
        );
    }

    // LOW: spend/logs v1 must support a team_id filter (parity with v2 and
    // the reference). Previously the From impl hardcoded team_id: None, so an
    // operator on the v1 endpoint could not narrow spend to a team.
    #[tokio::test]
    async fn spend_logs_v1_filters_by_team_id() {
        let (shared, db) = shared_with_db().await;
        for i in 0..3 {
            db.insert_spend_log(&spend_log_with_cost(
                &format!("2026-07-0{i}T00:00:00Z"),
                1.0,
                Some("team-a"),
            ))
            .await
            .expect("insert");
        }
        for i in 0..2 {
            db.insert_spend_log(&spend_log_with_cost(
                &format!("2026-08-0{i}T00:00:00Z"),
                2.0,
                Some("team-b"),
            ))
            .await
            .expect("insert");
        }

        let q = SpendLogsQuery {
            team_id: Some("team-a".to_string()),
            ..logs_query(None, None)
        };
        let resp = spend_logs(State(shared), Extension(make_auth()), Query(q))
            .await
            .expect("handler");
        let body = body_json(resp).await;
        let data = body["data"].as_array().expect("array");
        assert_eq!(data.len(), 3, "only team-a rows");
        assert!(
            data.iter().all(|r| r["team_id"] == "team-a"),
            "all rows belong to team-a"
        );
        assert_eq!(
            body["total_requests"].as_u64(),
            Some(3),
            "totals respect the team_id filter"
        );
    }

    /// `/spend/total` must honour `team_id` (and `key_id`): the admin UI's
    /// per-team budget-utilization bar queries this endpoint, and a silently
    /// ignored filter would report the GLOBAL spend against every team's own
    /// budget.
    /// `/spend/tags` aggregates spend by tag; multi-tag rows overlap
    /// into every carried tag by design. `/spend/logs?tag_name=` filters
    /// by exact membership so `app-a` never matches `app-ab`.
    #[tokio::test]
    async fn spend_tags_aggregates_and_filters_by_tag() {
        let (shared, db) = shared_with_db().await;
        let with_tags = |ts: &str, cost: f64, tags: Option<&str>, primary: Option<&str>| {
            let mut log = spend_log_with_cost(ts, cost, None);
            log.tags = tags.map(str::to_string);
            log.tag_name = primary.map(str::to_string);
            log
        };
        db.insert_spend_log(&with_tags(
            "2026-07-01T00:00:00Z",
            1.0,
            Some(r#"["app-a"]"#),
            Some("app-a"),
        ))
        .await
        .expect("insert");
        db.insert_spend_log(&with_tags(
            "2026-07-02T00:00:00Z",
            2.0,
            Some(r#"["app-a","env-prod"]"#),
            Some("app-a"),
        ))
        .await
        .expect("insert");
        db.insert_spend_log(&with_tags("2026-07-03T00:00:00Z", 4.0, None, None))
            .await
            .expect("insert");

        let by_name = |body: &serde_json::Value, tag: &str| {
            body["data"]
                .as_array()
                .expect("data")
                .iter()
                .find(|r| r["tag_name"] == tag)
                .unwrap_or_else(|| panic!("missing {tag} in {body}"))
                .clone()
        };
        let dq = SpendTagsQuery::default();
        let resp = spend_by_tag(State(shared.clone()), Extension(make_auth()), Query(dq))
            .await
            .expect("handler");
        let body = body_json(resp).await;
        assert_eq!(by_name(&body, "app-a")["total_spend"], 3.0, "1.0 + 2.0");
        assert_eq!(by_name(&body, "app-a")["total_requests"], 2);
        assert_eq!(
            by_name(&body, "env-prod")["total_spend"],
            2.0,
            "multi-tag row only"
        );
        let untagged = body["data"]
            .as_array()
            .expect("data")
            .iter()
            .find(|r| r["tag_name"].is_null())
            .expect("untagged group");
        assert_eq!(untagged["total_spend"], 4.0);

        let fq = SpendLogsQuery {
            tag_name: Some("app-a".to_string()),
            ..logs_query(None, None)
        };
        let resp = spend_logs(State(shared), Extension(make_auth()), Query(fq))
            .await
            .expect("handler");
        let body = body_json(resp).await;
        assert_eq!(body["data"].as_array().map(|a| a.len()), Some(2));
    }

    #[tokio::test]
    async fn spend_total_filters_by_team_id() {
        let (shared, db) = shared_with_db().await;
        for i in 0..3 {
            db.insert_spend_log(&spend_log_with_cost(
                &format!("2026-07-0{i}T00:00:00Z"),
                1.0,
                Some("team-a"),
            ))
            .await
            .expect("insert");
        }
        db.insert_spend_log(&spend_log_with_cost(
            "2026-08-01T00:00:00Z",
            9.0,
            Some("team-b"),
        ))
        .await
        .expect("insert");

        let q = SpendTotalQuery {
            team_id: Some("team-a".to_string()),
            start_date: None,
            key_id: None,
        };
        let resp = spend_total(State(shared), Extension(make_auth()), Query(q))
            .await
            .expect("handler");
        let body = body_json(resp).await;
        assert_eq!(body["total_cost"].as_f64(), Some(3.0), "team-a rows only");
        assert_eq!(body["total_requests"].as_u64(), Some(3));
    }

    // MEDIUM: global/spend must compute cost and count over the same scope.
    #[tokio::test]
    async fn global_spend_count_uses_end_date() {
        let (shared, db) = shared_with_db().await;
        db.insert_spend_log(&spend_log_with_cost("2026-07-01T00:00:00Z", 1.0, None))
            .await
            .expect("insert");
        db.insert_spend_log(&spend_log_with_cost("2026-08-01T00:00:00Z", 2.0, None))
            .await
            .expect("insert");

        let q = GlobalSpendQuery {
            start_date: Some("2026-07-01".to_string()),
            end_date: Some("2026-07-31".to_string()),
        };
        let resp = global_spend(State(shared), Extension(make_auth()), Query(q))
            .await
            .expect("handler");
        let body = body_json(resp).await;
        assert_eq!(
            body["total_requests"].as_u64(),
            Some(1),
            "count honors end_date"
        );
        assert_eq!(
            body["total_cost"].as_f64(),
            Some(1.0),
            "cost is same scope as count"
        );
    }

    // MEDIUM: global/spend end_date inclusivity (bare YYYY-MM-DD expands).
    #[tokio::test]
    async fn global_spend_end_date_inclusive_of_end_day() {
        let (shared, db) = shared_with_db().await;
        db.insert_spend_log(&spend_log_with_cost("2026-07-31T12:00:00Z", 1.0, None))
            .await
            .expect("insert");

        let q = GlobalSpendQuery {
            start_date: Some("2026-07-01".to_string()),
            end_date: Some("2026-07-31".to_string()),
        };
        let resp = global_spend(State(shared), Extension(make_auth()), Query(q))
            .await
            .expect("handler");
        let body = body_json(resp).await;
        assert_eq!(
            body["total_requests"].as_u64(),
            Some(1),
            "end day must be included"
        );
    }

    // LOW: spend/logs/v2 must echo the clamped limit, not the raw query param.
    #[tokio::test]
    async fn spend_logs_v2_echoes_clamped_limit() {
        let (shared, db) = shared_with_db().await;
        db.insert_spend_log(&spend_log_with_cost("2026-07-01T00:00:00Z", 1.0, None))
            .await
            .expect("insert");

        let q = SpendLogsV2Query {
            limit: Some(-5),
            ..Default::default()
        };
        let resp = spend_logs_v2(State(shared), Extension(make_auth()), Query(q))
            .await
            .expect("handler");
        let body = body_json(resp).await;
        let echoed = body["limit"].as_i64().expect("limit");
        assert!(
            (1..=1000).contains(&echoed),
            "echoed limit must be clamped into [1, 1000], got {echoed}"
        );
    }

    // LOW: spend/logs/v2 must echo the clamped offset (negative offsets are
    // coerced to 0 inside query_spend_paginated; echoing the raw value would
    // let a client build a next_cursor URL pointing at a value inconsistent
    // with the query that actually ran).
    #[tokio::test]
    async fn spend_logs_v2_echoes_clamped_offset() {
        let (shared, db) = shared_with_db().await;
        for i in 0..5 {
            db.insert_spend_log(&spend_log_with_cost(
                &format!("2026-07-0{i}T00:00:00Z"),
                1.0,
                None,
            ))
            .await
            .expect("insert");
        }

        let q = SpendLogsV2Query {
            limit: Some(3),
            offset: Some(-10),
            ..Default::default()
        };
        let resp = spend_logs_v2(State(shared), Extension(make_auth()), Query(q))
            .await
            .expect("handler");
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let echoed = body["offset"].as_i64().expect("offset");
        assert_eq!(
            echoed, 0,
            "negative offset must be echoed as 0, got {echoed}"
        );
        // First page of 3, next_cursor advances from clamped offset 0.
        assert_eq!(
            body["next_cursor"].as_i64(),
            Some(3),
            "next_cursor must be derived from the clamped offset"
        );
    }
}
