//! Spend logs page, CSV export, and spend/budget data-shaping shared with
//! the keys/teams/users tables.

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::{AppState, is_htmx, proxy_error_response, render_page_or_partial};
use crate::csv::spend_logs_to_csv;
use crate::dates::{days_before_iso, parse_iso_date_to_unix_days, today_iso_date};
use crate::error::AdminError;
use crate::proxy_client::{ListResponse, ProxyClient};
use crate::urlencode::urlencoding_minimal;

#[derive(Debug, Deserialize, Default, Clone)]
pub(crate) struct SpendQuery {
    #[serde(default)]
    pub(crate) cursor: Option<i64>,
    /// Offset for sorted paging (server-side).
    #[serde(default)]
    pub(crate) offset: Option<i64>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) key_id: Option<String>,
    #[serde(default)]
    pub(crate) team_id: Option<String>,
    #[serde(default)]
    pub(crate) start_date: Option<String>,
    #[serde(default)]
    pub(crate) end_date: Option<String>,
    /// Column to sort: timestamp | model | provider | cost | key_id | team_id | prompt_tokens | completion_tokens
    #[serde(default)]
    pub(crate) sort_by: Option<String>,
    /// asc | desc (default desc)
    #[serde(default)]
    pub(crate) sort_order: Option<String>,
}

impl SpendQuery {
    fn sort_by(&self) -> &str {
        match self.sort_by.as_deref().unwrap_or("timestamp") {
            "model" | "provider" | "cost" | "key_id" | "team_id" | "prompt_tokens"
            | "completion_tokens" | "timestamp" => self.sort_by.as_deref().unwrap_or("timestamp"),
            _ => "timestamp",
        }
    }

    fn sort_order(&self) -> &str {
        match self.sort_order.as_deref().unwrap_or("desc") {
            "asc" => "asc",
            _ => "desc",
        }
    }

    /// URL-encoded filter-only query string (no sort/paging), used by the
    /// sort header and "Load more" links so values containing `&`, `=`, or
    /// spaces round-trip correctly. Mirrors `to_proxy_query_string`'s
    /// encoding but excludes `sort_by`/`sort_order`/`offset`/`cursor`, which
    /// the templates interpolate separately.
    fn to_filters_query_string(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (k, v) in [
            ("model", self.model.as_deref()),
            ("key_id", self.key_id.as_deref()),
            ("team_id", self.team_id.as_deref()),
            ("start_date", self.start_date.as_deref()),
            ("end_date", self.end_date.as_deref()),
        ] {
            if let Some(val) = v {
                if !val.is_empty() {
                    parts.push(format!("{k}={}", urlencoding_minimal(val)));
                }
            }
        }
        parts.join("&")
    }

    fn to_filter_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        for (k, v) in [
            ("model", self.model.as_deref()),
            ("key_id", self.key_id.as_deref()),
            ("team_id", self.team_id.as_deref()),
            ("start_date", self.start_date.as_deref()),
            ("end_date", self.end_date.as_deref()),
        ] {
            if let Some(val) = v {
                if !val.is_empty() {
                    map.insert(k.to_string(), json!(val));
                }
            }
        }
        Value::Object(map)
    }

    /// Proxy spend/logs/v2 query (filters + sort + offset/cursor).
    fn to_proxy_query_string(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        // Prefer offset paging when sorting so ORDER BY is consistent across
        // pages. Negative values (e.g. `?offset=-5`) are clamped to 0 so they
        // are never forwarded verbatim to the proxy.
        if let Some(o) = self.offset {
            parts.push(format!("offset={}", o.max(0)));
        } else if let Some(c) = self.cursor {
            // legacy cursor id paging when no offset
            parts.push(format!("cursor={}", c.max(0)));
        }
        for (k, v) in [
            ("model", self.model.as_deref()),
            ("key_id", self.key_id.as_deref()),
            ("team_id", self.team_id.as_deref()),
            ("start_date", self.start_date.as_deref()),
            ("end_date", self.end_date.as_deref()),
        ] {
            if let Some(val) = v {
                if !val.is_empty() {
                    parts.push(format!("{k}={}", urlencoding_minimal(val)));
                }
            }
        }
        parts.push(format!("sort_by={}", self.sort_by()));
        parts.push(format!("sort_order={}", self.sort_order()));
        parts.join("&")
    }

    /// Query string for admin list/export (filters + sort, no cursor).
    fn to_list_query_string(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (k, v) in [
            ("model", self.model.as_deref()),
            ("key_id", self.key_id.as_deref()),
            ("team_id", self.team_id.as_deref()),
            ("start_date", self.start_date.as_deref()),
            ("end_date", self.end_date.as_deref()),
        ] {
            if let Some(val) = v {
                if !val.is_empty() {
                    parts.push(format!("{k}={}", urlencoding_minimal(val)));
                }
            }
        }
        parts.push(format!("sort_by={}", self.sort_by()));
        parts.push(format!("sort_order={}", self.sort_order()));
        parts.join("&")
    }

    pub(crate) fn has_date_range(&self) -> bool {
        non_empty(self.start_date.as_deref()).is_some()
            && non_empty(self.end_date.as_deref()).is_some()
    }

    /// Fill missing start/end with a default window (last `default_days` days,
    /// inclusive). Used so the UI always has an explicit export range.
    pub(crate) fn with_default_date_range(mut self, default_days: i64) -> Self {
        let end = non_empty(self.end_date.as_deref())
            .map(|s| s.to_string())
            .unwrap_or_else(today_iso_date);
        let start = non_empty(self.start_date.as_deref())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                days_before_iso(&end, default_days.saturating_sub(1).max(0))
                    .unwrap_or_else(|| end.clone())
            });
        self.start_date = Some(start);
        self.end_date = Some(end);
        self
    }
}

fn non_empty(s: Option<&str>) -> Option<&str> {
    s.map(str::trim).filter(|v| !v.is_empty())
}

/// Validate export date range. Returns `(start, end)` or an error message.
pub(crate) fn validate_export_date_range(q: &SpendQuery) -> Result<(String, String), String> {
    let start = non_empty(q.start_date.as_deref())
        .ok_or_else(|| "start_date is required (YYYY-MM-DD)".to_string())?
        .to_string();
    let end = non_empty(q.end_date.as_deref())
        .ok_or_else(|| "end_date is required (YYYY-MM-DD)".to_string())?
        .to_string();
    let start_days = parse_iso_date_to_unix_days(&start)
        .ok_or_else(|| format!("invalid start_date '{start}' (use YYYY-MM-DD)"))?;
    let end_days = parse_iso_date_to_unix_days(&end)
        .ok_or_else(|| format!("invalid end_date '{end}' (use YYYY-MM-DD)"))?;
    if start_days > end_days {
        return Err("start_date must be on or before end_date".into());
    }
    const MAX_RANGE_DAYS: i64 = 366;
    if end_days - start_days > MAX_RANGE_DAYS {
        return Err(format!(
            "date range too large (max {MAX_RANGE_DAYS} days); narrow Start/End in the UI"
        ));
    }
    Ok((start, end))
}

fn value_as_str(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

fn value_as_f64(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
        .unwrap_or(0.0)
}

pub(crate) fn sort_spend_logs(logs: &mut [Value], sort_by: &str, sort_order: &str) {
    let desc = sort_order != "asc";
    logs.sort_by(|a, b| {
        let cmp = match sort_by {
            "model" => value_as_str(a.get("model").unwrap_or(&Value::Null))
                .cmp(value_as_str(b.get("model").unwrap_or(&Value::Null))),
            "provider" => value_as_str(a.get("provider").unwrap_or(&Value::Null))
                .cmp(value_as_str(b.get("provider").unwrap_or(&Value::Null))),
            "key_id" => value_as_str(a.get("key_id").unwrap_or(&Value::Null))
                .cmp(value_as_str(b.get("key_id").unwrap_or(&Value::Null))),
            "team_id" => value_as_str(a.get("team_id").unwrap_or(&Value::Null))
                .cmp(value_as_str(b.get("team_id").unwrap_or(&Value::Null))),
            "cost" => value_as_f64(a.get("cost").unwrap_or(&Value::Null))
                .partial_cmp(&value_as_f64(b.get("cost").unwrap_or(&Value::Null)))
                .unwrap_or(std::cmp::Ordering::Equal),
            "prompt_tokens" => value_as_f64(a.get("prompt_tokens").unwrap_or(&Value::Null))
                .partial_cmp(&value_as_f64(
                    b.get("prompt_tokens").unwrap_or(&Value::Null),
                ))
                .unwrap_or(std::cmp::Ordering::Equal),
            "completion_tokens" => value_as_f64(a.get("completion_tokens").unwrap_or(&Value::Null))
                .partial_cmp(&value_as_f64(
                    b.get("completion_tokens").unwrap_or(&Value::Null),
                ))
                .unwrap_or(std::cmp::Ordering::Equal),
            _ => cmp_timestamp(
                a.get("timestamp").unwrap_or(&Value::Null),
                b.get("timestamp").unwrap_or(&Value::Null),
            ),
        };
        if desc { cmp.reverse() } else { cmp }
    });
}

/// Compare two `timestamp` values chronologically. Prefer a numeric
/// comparison (Unix epoch seconds/millis, whether JSON number or numeric
/// string) so a non-ISO format still sorts in chronological order; fall back
/// to lexicographic comparison, which is correct for ISO 8601 / RFC 3339
/// strings (the proxy's actual format). This guards the CSV-export re-sort
/// against a future proxy revision that returns epoch integers.
fn cmp_timestamp(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (timestamp_as_number(a), timestamp_as_number(b)) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        _ => value_as_str(a).cmp(value_as_str(b)),
    }
}

/// Best-effort numeric coercion of a timestamp value: JSON numbers and
/// numeric strings (e.g. epoch seconds) parse; ISO 8601 strings do not.
fn timestamp_as_number(v: &Value) -> Option<f64> {
    if v.is_number() {
        return value_as_f64(v).into();
    }
    value_as_str(v).parse::<f64>().ok()
}

pub(crate) fn enrich_spend_logs(logs: Vec<Value>, keys: &[Value], teams: &[Value]) -> Vec<Value> {
    let mut key_alias: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    key_alias.insert("master".into(), "Master key".into());
    for k in keys {
        if let Some(id) = k.get("key_id").and_then(|v| v.as_str()) {
            let alias = k
                .get("key_alias")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(id)
                .to_string();
            key_alias.insert(id.to_string(), alias);
        }
    }
    let mut team_name: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for t in teams {
        if let Some(id) = t.get("team_id").and_then(|v| v.as_str()) {
            let name = t
                .get("team_name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(id)
                .to_string();
            team_name.insert(id.to_string(), name);
        }
    }

    logs.into_iter()
        .map(|mut row| {
            if let Some(obj) = row.as_object_mut() {
                if let Some(kid) = obj.get("key_id").and_then(|v| v.as_str()) {
                    if let Some(alias) = key_alias.get(kid) {
                        obj.insert("key_alias".into(), json!(alias));
                    }
                }
                if let Some(tid) = obj.get("team_id").and_then(|v| v.as_str()) {
                    if let Some(name) = team_name.get(tid) {
                        obj.insert("team_name".into(), json!(name));
                    }
                }
            }
            row
        })
        .collect()
}

/// Budget utilization tier derived from `spend / max_budget`.
pub(crate) fn budget_spend_tier(pct: f64) -> &'static str {
    if pct >= 80.0 {
        "danger"
    } else if pct >= 50.0 {
        "warn"
    } else {
        "ok"
    }
}

/// Pull the total spend from a `/spend/total` payload. LiteLLM's responses
/// vary by version: most return `{"total_spend": <f>}` but some return a
/// nested `results` array. We pick the first numeric value we find.
pub(crate) fn spend_total_from_response(v: &Value) -> f64 {
    if let Some(s) = v.get("total_spend").and_then(|x| x.as_f64()) {
        return s;
    }
    if let Some(arr) = v.get("results").and_then(|x| x.as_array()) {
        if let Some(first) = arr.first() {
            if let Some(s) = first.get("total_spend").and_then(|x| x.as_f64()) {
                return s;
            }
            if let Some(s) = first.get("spend").and_then(|x| x.as_f64()) {
                return s;
            }
            if let Some(s) = first.get("total_cost").and_then(|x| x.as_f64()) {
                return s;
            }
        }
    }
    if let Some(s) = v.get("total_cost").and_then(|x| x.as_f64()) {
        return s;
    }
    0.0
}

/// For each entity with `max_budget`, fetch the in-window spend via
/// `/spend/total` and attach `budget_spend`, `budget_spend_pct`, and
/// `budget_spend_tier` for template rendering. Failures are logged and
/// treated as zero spend so a proxy hiccup never breaks the table.
///
/// Only scopes the endpoint actually filters on are enriched: `key_id` and
/// `team_id`. Any other `id_key` (notably `user_id` — there is no
/// user-scoped total endpoint) returns immediately, because an ignored
/// filter makes `/spend/total` answer with the **global** total and every
/// row would show the same worldwide spend against its own budget.
///
/// The query window is derived from each entity's `budget_duration`
/// (LiteLLM-style: `1d`, `7d`, `30d`, `1mo`, `1w`, ...) so the bar reflects
/// the same window the proxy enforces. Rows with a missing/unparseable
/// duration fall back to a 30-day window. Calls are fanned out with bounded
/// concurrency (`CONCURRENCY`); rows past `ENRICHMENT_CAP` are skipped to
/// keep page renders bounded (matches `enrich_models`). The aggregate fan-out
/// is bounded by `PAGE_ENRICHMENT_DEADLINE` so a degraded proxy cannot tie up
/// a worker for `ceil(CAP / CONCURRENCY) * PROXY_REQUEST_TIMEOUT` (~6 min);
/// rows not enriched before the deadline render without a budget bar.
pub(crate) async fn enrich_with_budget_spend(
    proxy: &ProxyClient,
    rows: &mut [Value],
    id_key: &str,
) {
    enrich_with_budget_spend_bounded(proxy, rows, id_key, PAGE_ENRICHMENT_DEADLINE).await;
}

/// Overall wall-clock budget for a single interactive page-render enrichment
/// pass. Each individual proxy call is bounded by `PROXY_REQUEST_TIMEOUT`
/// (15 s); this bounds the *aggregate* so a slow proxy can't tie up an admin
/// worker during a page render. Mirrors `EXPORT_DEADLINE` for the CSV path;
/// the page-render budget is tighter because the operator is waiting on the
/// response interactively.
pub(crate) const PAGE_ENRICHMENT_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

pub(crate) async fn enrich_with_budget_spend_bounded(
    proxy: &ProxyClient,
    rows: &mut [Value],
    id_key: &str,
    deadline: std::time::Duration,
) {
    use futures_util::stream::{self, StreamExt};
    use std::time::Instant;

    // `/spend/total` filters on `key_id` and `team_id` only. Anything else
    // (user rows) cannot be scoped here; rendering a global figure against a
    // per-entity budget would be worse than rendering no figure at all.
    if !matches!(id_key, "key_id" | "team_id") {
        tracing::debug!(
            entity = %id_key,
            "skipping budget-spend enrichment: /spend/total has no such filter"
        );
        return;
    }

    const CONCURRENCY: usize = 8;
    const ENRICHMENT_CAP: usize = 200;
    let today = today_iso_date();
    // Collect (row index, id, budget, start_date) for every row worth enriching.
    let tasks: Vec<(usize, String, f64, String)> = rows
        .iter()
        .enumerate()
        .take(ENRICHMENT_CAP)
        .filter_map(|(i, row)| {
            let max_budget = match row.get("max_budget") {
                Some(Value::Null) | None => None,
                Some(v) => v.as_f64().or_else(|| v.as_i64().map(|n| n as f64)),
            };
            let budget = max_budget.filter(|b| *b > 0.0)?;
            let id = row
                .get(id_key)
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())?
                .to_string();
            let duration = row.get("budget_duration").and_then(|v| v.as_str());
            let start_date = budget_window_start_date(duration, &today);
            Some((i, id, budget, start_date))
        })
        .collect();
    if tasks.is_empty() {
        return;
    }

    let started = Instant::now();
    let mut stream = stream::iter(tasks)
        .map(|(i, id, budget, start_date)| {
            let params = format!(
                "{}={}&start_date={}",
                id_key,
                urlencoding_minimal(&id),
                start_date
            );
            async move {
                let spend = match proxy.spend_total(&params).await {
                    Ok(v) => spend_total_from_response(&v),
                    Err(e) => {
                        tracing::warn!(error = %e, entity = %id_key, id = %id, "spend/total failed");
                        0.0
                    }
                };
                let pct = ((spend / budget) * 100.0).clamp(0.0, 100.0);
                (i, spend, pct)
            }
        })
        .buffer_unordered(CONCURRENCY);

    // Drive the fan-out with an overall deadline. `buffer_unordered` only
    // polls `CONCURRENCY` futures at once, so breaking the loop drops the
    // remaining un-started tasks (and cancels the in-flight ones); rows that
    // never produced a result simply render without `budget_spend`. Worst
    // case the handler returns within `deadline` plus one in-flight proxy
    // call (matching `fetch_spend_pages`'s documented bound).
    let mut results: Vec<(usize, f64, f64)> = Vec::new();
    loop {
        if started.elapsed() >= deadline {
            break;
        }
        match stream.next().await {
            Some(item) => results.push(item),
            None => break,
        }
    }
    drop(stream);

    for (i, spend, pct) in results {
        if let Some(obj) = rows[i].as_object_mut() {
            obj.insert("budget_spend".into(), json!(spend));
            obj.insert("budget_spend_pct".into(), json!(pct));
            obj.insert("budget_spend_tier".into(), json!(budget_spend_tier(pct)));
        }
    }
}

/// Compute the `/spend/total` `start_date` for a given `budget_duration`,
/// aligned to LiteLLM's semantics. Units mirror LiteLLM's `duration_parser`:
/// `s`/`m`/`h` collapse to today (sub-day window), `d`/`w`/`mo` map to N-1
/// days back so the window covers N calendar days inclusive of today
/// (`1d` -> today, `7d` -> today-6, `30d` -> today-29, matching the previous
/// hard-coded 30-day behaviour). Word aliases (`daily`, `weekly`, `monthly`,
/// `hourly`) are normalised first. Missing/unparseable durations fall back
/// to the historical 30-day window so existing rows render unchanged.
fn budget_window_start_date(duration: Option<&str>, today: &str) -> String {
    let fallback = || days_before_iso(today, 29).unwrap_or_else(|| today.to_string());
    let raw = match duration.map(str::trim).filter(|s| !s.is_empty()) {
        Some(d) => d,
        None => return fallback(),
    };
    let canonical = match raw.to_lowercase().as_str() {
        "hourly" => "1h",
        "daily" => "24h",
        "weekly" => "7d",
        "monthly" => "30d",
        _ => raw,
    };
    let split = canonical.bytes().position(|b| !b.is_ascii_digit());
    let Some(split) = split else {
        return fallback();
    };
    let (num_str, unit) = canonical.split_at(split);
    let Ok(n) = num_str.parse::<i64>() else {
        return fallback();
    };
    if n <= 0 {
        return fallback();
    }
    let days_back = match unit {
        "s" | "m" | "h" => 0,
        "d" => n - 1,
        "w" => match n.checked_mul(7) {
            Some(days) => days - 1,
            None => return fallback(),
        },
        "mo" => match n.checked_mul(30) {
            Some(days) => days - 1,
            None => return fallback(),
        },
        _ => return fallback(),
    };
    days_before_iso(today, days_back.max(0)).unwrap_or_else(fallback)
}

pub(crate) async fn spend_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<SpendQuery>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    // Always materialize Start/End so list + export share an explicit window.
    // Fill whenever *either* side is missing: `with_default_date_range` fills
    // sides individually, and leaving one absent would produce an export link
    // missing a bound (which `validate_export_date_range` rejects with a 400).
    let q = if q.has_date_range() {
        q
    } else {
        q.with_default_date_range(30)
    };
    let query_str = q.to_proxy_query_string();

    // Fan spend_logs and the two dropdown feeds out concurrently so worst-case
    // latency is a single request-timeout window rather than 3x (matching the
    // dashboard / keys handlers). `spend_logs` surfaces its error; the key and
    // team lists degrade gracefully to empty on failure.
    let (spend_res, keys, teams) = tokio::join!(
        state.proxy.spend_logs(&query_str),
        async {
            state.proxy.list_keys().await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "list_keys for spend page failed");
                ListResponse::unavailable()
            })
        },
        async {
            state.proxy.list_teams().await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "list_teams for spend page failed");
                ListResponse::unavailable()
            })
        },
    );
    let value = match spend_res {
        Ok(v) => v,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };

    let mut logs = value
        .get("data")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let next_cursor = value.get("next_cursor").cloned().unwrap_or(Value::Null);
    let page_total_cost = value
        .get("total_cost")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let page_total_requests = value
        .get("total_requests")
        .and_then(|v| v.as_u64())
        .unwrap_or(logs.len() as u64);
    let filters = q.to_filter_json();
    let sort_by = q.sort_by().to_string();
    let sort_order = q.sort_order().to_string();

    logs = enrich_spend_logs(logs, &keys.items, &teams.items);
    // Sorting is done server-side via /spend/logs/v2 sort_by/sort_order.

    if is_htmx && (q.cursor.is_some() || q.offset.is_some()) {
        let filters_qs = q.to_filters_query_string();
        let ctx = json!({
            "rows": logs,
            "next_cursor": next_cursor,
            "filters": filters,
            "filters_qs": filters_qs,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "is_oob": true,
        });
        let html = state
            .templates
            .render("partials/spend_pagination.html", &ctx)?;
        return Ok(Html(html).into_response());
    }

    let list_query = q.to_list_query_string();
    let filters_qs = q.to_filters_query_string();
    let ctx = json!({
        "session_active": true,
        "logs": logs,
        "rows": logs,
        "next_cursor": next_cursor,
        "filters": filters,
        "filters_qs": filters_qs,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "keys": keys.items,
        "keys_truncated": keys.truncated,
        "keys_unavailable": keys.unavailable,
        "teams": teams.items,
        "teams_truncated": teams.truncated,
        "teams_unavailable": teams.unavailable,
        "list_query": list_query,
        "export_query": list_query,
        "export_start": q.start_date,
        "export_end": q.end_date,
        "page_total_cost": page_total_cost,
        "page_total_requests": page_total_requests,
    });
    render_page_or_partial(
        &state,
        &headers,
        is_htmx,
        "pages/spend.html",
        "partials/spend_table.html",
        &ctx,
    )
}

/// Export spend logs as CSV for the UI date range (`start_date` + `end_date`
/// required). Optional model/key/team filters still apply.
pub(crate) async fn spend_export_csv(
    State(state): State<AppState>,
    Query(q): Query<SpendQuery>,
) -> Result<Response, AdminError> {
    const PAGE: i64 = 200;
    // Safety cap within a date window (not a substitute for Start/End).
    const MAX_ROWS: usize = 10_000;

    let (start, end) = match validate_export_date_range(&q) {
        Ok(range) => range,
        Err(msg) => {
            return Ok((
                StatusCode::BAD_REQUEST,
                [("content-type", "text/plain; charset=utf-8")],
                format!(
                    "Export requires a date range from the Spend filters.\n{msg}\n\nSet Start and End, then click Export CSV."
                ),
            )
                .into_response());
        }
    };

    // Force proxy filters to the validated range (ignore empty/partial query).
    let mut q = q;
    q.start_date = Some(start.clone());
    q.end_date = Some(end.clone());

    // Snapshot cutoff: capture the current UTC date once before iterating so
    // every paginated request shares the same upper bound. Without this, rows
    // inserted mid-export can be duplicated (cursor boundary shifts) or
    // skipped. We cap at today because the proxy's `end_date` is a DATE
    // filter; historical windows (end_date in the past) are already stable.
    let snapshot_end = {
        let today = today_iso_date();
        if end <= today { end.clone() } else { today }
    };
    q.end_date = Some(snapshot_end.clone());
    // Also pass the cutoff as an explicit `created_at_max` filter so the
    // intent is self-documenting even if a future proxy revision adds support.
    let created_at_max = snapshot_end.clone();

    let keys = state.proxy.list_keys().await.unwrap_or_default();
    let teams = state.proxy.list_teams().await.unwrap_or_default();

    let fetch = match fetch_spend_pages(
        &state.proxy,
        &q,
        &created_at_max,
        PAGE,
        MAX_ROWS,
        EXPORT_DEADLINE,
    )
    .await
    {
        Ok(f) => f,
        Err(e) => {
            // CSV export is triggered via `window.location = ...` (a full
            // navigation, not an HTMX swap), so an `HX-Redirect` would be
            // silently ignored and leave the user on a blank page. On 401
            // return a real HTTP redirect so the browser lands on /login.
            if crate::app::is_proxy_unauthorized(&e) {
                return Ok((
                    StatusCode::SEE_OTHER,
                    [("location", "/login")],
                    Html(String::new()),
                )
                    .into_response());
            }
            tracing::warn!(error = %e, "spend CSV export fetch failed");
            return Ok((
                StatusCode::BAD_GATEWAY,
                [("content-type", "text/plain; charset=utf-8")],
                format!(
                    "Failed to fetch spend logs from the proxy.\n{e}\n\nNarrow the date range or retry."
                ),
            )
                .into_response());
        }
    };

    let mut all = enrich_spend_logs(fetch.rows, &keys.items, &teams.items);
    sort_spend_logs(&mut all, q.sort_by(), q.sort_order());
    let mut csv = spend_logs_to_csv(&all);
    if fetch.truncated_by_deadline {
        csv.push_str(
            "\n# Export truncated: reached the 60s overall deadline. \
             Narrow the date range for a complete export.\n",
        );
        tracing::warn!(
            rows = all.len(),
            "spend CSV export truncated by overall deadline"
        );
    }
    if fetch.truncated_by_row_cap {
        csv.push_str(&format!(
            "\n# Export truncated: reached the {}-row cap. \
             Narrow the date range for a complete export.\n",
            MAX_ROWS
        ));
        tracing::warn!(
            rows = all.len(),
            cap = MAX_ROWS,
            "spend CSV export truncated by row cap"
        );
    }
    if fetch.truncated_by_missing_cursor {
        csv.push_str(
            "\n# Export truncated: a full page arrived without a usable \
             pagination cursor. Narrow the date range for a complete export.\n",
        );
        tracing::warn!(
            rows = all.len(),
            "spend CSV export truncated: full page without a usable cursor"
        );
    }
    let filename = format!("oxllm-spend-{start}_to_{end}.csv");
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/csv; charset=utf-8"),
    );
    if let Ok(v) =
        axum::http::HeaderValue::from_str(&format!("attachment; filename=\"{filename}\""))
    {
        headers.insert(axum::http::header::CONTENT_DISPOSITION, v);
    }
    Ok((StatusCode::OK, headers, csv).into_response())
}

/// Overall deadline for a single CSV export. A wide date range (up to 366
/// days) with small pages could otherwise drive ~`MAX_ROWS / 200` sequential
/// proxy round-trips and tie up a worker for many minutes. Each page is
/// individually bounded by `PROXY_REQUEST_TIMEOUT` (15 s); this bounds the
/// *sum* across pages so the handler always returns within `EXPORT_DEADLINE`
/// plus one in-flight proxy call.
pub(crate) const EXPORT_DEADLINE: std::time::Duration = std::time::Duration::from_secs(60);

/// Outcome of the paginated spend-log fetch backing a CSV export.
#[derive(Debug)]
pub(crate) struct ExportFetch {
    pub rows: Vec<Value>,
    pub truncated_by_deadline: bool,
    /// Set when the `max_rows` cap stopped the loop while more pages remained.
    pub truncated_by_row_cap: bool,
    /// Set when a full batch arrived without a usable cursor (no
    /// `next_cursor` and no integer `id` on the rows), so more pages may
    /// exist but pagination cannot continue.
    pub truncated_by_missing_cursor: bool,
}

/// Walk the spend-log pages once, honouring both `MAX_ROWS` and an overall
/// wall-clock `deadline`. Returns the collected rows and a `truncated` flag
/// set when the deadline stopped the loop mid-stream (the caller surfaces a
/// truncation note so the operator knows to narrow the range). Proxy errors
/// bubble up as `Err` for the handler to render.
pub(crate) async fn fetch_spend_pages(
    proxy: &ProxyClient,
    q: &SpendQuery,
    created_at_max: &str,
    page: i64,
    max_rows: usize,
    deadline: std::time::Duration,
) -> Result<ExportFetch, anyhow::Error> {
    use std::time::Instant;
    let mut rows: Vec<Value> = Vec::new();
    let mut cursor: Option<i64> = None;
    let started = Instant::now();
    let mut truncated_by_deadline = false;
    let mut truncated_by_row_cap = false;
    let mut truncated_by_missing_cursor = false;
    loop {
        if started.elapsed() >= deadline {
            truncated_by_deadline = true;
            break;
        }
        let mut parts = vec![format!("limit={page}")];
        if let Some(c) = cursor {
            parts.push(format!("cursor={c}"));
        }
        for (k, v) in [
            ("model", q.model.as_deref()),
            ("key_id", q.key_id.as_deref()),
            ("team_id", q.team_id.as_deref()),
            ("start_date", q.start_date.as_deref()),
            ("end_date", q.end_date.as_deref()),
            ("created_at_max", Some(created_at_max)),
        ] {
            if let Some(val) = v {
                if !val.is_empty() {
                    parts.push(format!("{k}={}", urlencoding_minimal(val)));
                }
            }
        }
        let value = proxy.spend_logs(&parts.join("&")).await?;
        let batch = value
            .get("data")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if batch.is_empty() {
            break;
        }
        let next = value
            .get("next_cursor")
            .and_then(|v| v.as_i64())
            .or_else(|| {
                batch
                    .last()
                    .and_then(|r| r.get("id").and_then(|i| i.as_i64()))
            });
        let batch_len = batch.len();
        rows.extend(batch);
        if rows.len() >= max_rows {
            rows.truncate(max_rows);
            // Only a truncation when another page was still reachable;
            // otherwise the cap coincided with the natural end of the data.
            truncated_by_row_cap = matches!(next, Some(c) if cursor != Some(c));
            break;
        }
        match next {
            Some(c) if cursor != Some(c) => cursor = Some(c),
            _ => {
                // A full batch with no usable cursor (proxy omitted
                // `next_cursor` and rows lack an integer `id`) plausibly
                // has more pages; flag truncation instead of silently
                // exporting a partial CSV.
                if next.is_none() && batch_len == page as usize {
                    truncated_by_missing_cursor = true;
                }
                break;
            }
        }
    }
    Ok(ExportFetch {
        rows,
        truncated_by_deadline,
        truncated_by_row_cap,
        truncated_by_missing_cursor,
    })
}
