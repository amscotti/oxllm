//! Dashboard page: global spend totals, proxy health, and daily activity.

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use serde_json::{Value, json};

use crate::app::{AppState, is_htmx, render_page_or_partial};
use crate::error::AdminError;

pub(crate) async fn dashboard(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);

    // Fan the four independent proxy calls out concurrently so worst-case
    // latency is a single request-timeout window rather than 4x.
    let (total_spend, live, ready, activity) = tokio::join!(
        async {
            state.proxy.global_spend().await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "global_spend failed");
                json!({})
            })
        },
        state.proxy.health(),
        state.proxy.readiness(),
        async {
            state.proxy.global_activity(30).await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "global_activity failed");
                json!({})
            })
        }
    );

    let total_cost = total_spend
        .get("total_cost")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let total_requests = total_spend
        .get("total_requests")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let activity_raw = activity
        .get("results")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    // Compute bar % in Rust — MiniJinja loop `{% set %}` does not update outer
    // scope, so max-in-loop templates always produced 0%-wide activity bars.
    let max_activity_spend = activity_raw
        .iter()
        .filter_map(|r| r.get("spend").and_then(|v| v.as_f64()))
        .fold(0.0_f64, f64::max);
    let activity_rows: Vec<Value> = activity_raw
        .into_iter()
        .map(|r| {
            let spend = r.get("spend").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let bar_pct = if max_activity_spend > 0.0 {
                ((spend / max_activity_spend) * 100.0).round()
            } else {
                0.0
            };
            json!({
                "day": r.get("day").cloned().unwrap_or(Value::Null),
                "spend": spend,
                "bar_pct": bar_pct,
            })
        })
        .collect();

    let health_view = match (&live, &ready) {
        (Ok(_), Ok(r)) => {
            let status = r
                .get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("healthy");
            let ready_ok = status == "ok" || status == "healthy" || status == "ready";
            json!({
                "reachable": true,
                "healthy": ready_ok,
                "degraded": !ready_ok,
                "status": status,
                "checks": r.get("checks").cloned().unwrap_or(json!({})),
                "error": null,
            })
        }
        (Ok(_), Err(e)) => json!({
            "reachable": true,
            "healthy": false,
            "degraded": true,
            "status": "degraded",
            "checks": {},
            "error": e.to_string(),
        }),
        (Err(e), _) => json!({
            "reachable": false,
            "healthy": false,
            "degraded": false,
            "status": "unreachable",
            "checks": {},
            "error": e.to_string(),
        }),
    };

    let ctx = json!({
        "session_active": true,
        "total_spend": total_cost,
        "total_requests": total_requests,
        "health": health_view,
        "activity": activity_rows,
        "activity_days": 30,
        "proxy_url": state.config.proxy_url,
    });

    render_page_or_partial(
        &state,
        &headers,
        is_htmx,
        "pages/dashboard.html",
        "partials/dashboard_summary.html",
        &ctx,
    )
}
