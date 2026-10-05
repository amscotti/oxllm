//! Models page and model create/delete handlers.

use axum::Form;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::{AppState, error_response, is_htmx, proxy_error_response, render_page_or_partial};
use crate::error::AdminError;
use crate::proxy_client::ProxyClient;

pub(crate) async fn models_page(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let list = match state.proxy.list_models().await {
        Ok(l) => l,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    let enriched = enrich_models(&state.proxy, list.items).await;
    let ctx =
        json!({ "session_active": true, "models": enriched, "models_truncated": list.truncated });
    render_page_or_partial(
        &state,
        &headers,
        is_htmx,
        "pages/models.html",
        "partials/models_table.html",
        &ctx,
    )
}

/// Form payload for `POST /models/create`. `model_name` is the client-facing
/// alias (e.g. `gpt-4o`); `provider_model` is the upstream provider/model
/// (e.g. `openai/gpt-4o`). The optional fields pass through verbatim into
/// `oxllm_params.*`.
///
/// The HTML form field is still named `litellm_model` (see
/// `templates/pages/models.html`); the serde rename keeps the parser in sync
/// with the template until the template is renamed too.
#[derive(Debug, Deserialize, Default)]
pub(crate) struct ModelCreateForm {
    #[serde(default)]
    model_name: String,
    #[serde(default, rename = "litellm_model")]
    provider_model: String,
    /// Optional API key (or `os.environ/...` reference). When blank the field
    /// is omitted and the proxy falls back to its own env-resolved key.
    #[serde(default)]
    api_key: String,
    /// Optional upstream base URL (e.g. for Azure / self-hosted gateways).
    #[serde(default)]
    api_base: String,
}

impl ModelCreateForm {
    fn to_payload(&self) -> Value {
        let mut params = serde_json::Map::new();
        params.insert("model".into(), self.provider_model.clone().into());
        if !self.api_key.is_empty() {
            params.insert("api_key".into(), self.api_key.clone().into());
        }
        if !self.api_base.is_empty() {
            params.insert("api_base".into(), self.api_base.clone().into());
        }
        json!({
            "model_name": self.model_name,
            "oxllm_params": Value::Object(params),
        })
    }
}

pub(crate) async fn models_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<ModelCreateForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    if form.model_name.trim().is_empty() || form.provider_model.trim().is_empty() {
        return Ok(error_response(
            &state,
            is_htmx,
            "Model name and upstream model are required.",
        ));
    }
    let payload = form.to_payload();
    if let Err(e) = state.proxy.create_model(payload).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    // Refresh the table so the new row shows up alongside the existing ones.
    let list = match state.proxy.list_models().await {
        Ok(l) => l,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    let enriched = enrich_models(&state.proxy, list.items).await;
    let ctx = json!({
        "session_active": true,
        "models": enriched,
        "models_truncated": list.truncated,
        "created": format!("Model {} added.", form.model_name),
    });
    let html = state.templates.render("partials/models_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

/// `POST /models/{model_id}/delete`. The path parameter is the proxy's DB
/// `model_id` (see `enrich_models`), percent-encoded by the table template —
/// a raw `model_name` with `/` (e.g. `openai/gpt-4o`) would split into
/// multiple path segments and 404 against this route, and the proxy's
/// `/model/delete` rejects names anyway.
pub(crate) async fn models_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(model_id): Path<String>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    if let Err(e) = state.proxy.delete_model(&model_id).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let list = match state.proxy.list_models().await {
        Ok(l) => l,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    let enriched = enrich_models(&state.proxy, list.items).await;
    let ctx = json!({
        "session_active": true,
        "models": enriched,
        "models_truncated": list.truncated,
        "created": format!("Model {} removed.", model_id),
    });
    let html = state.templates.render("partials/models_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

/// Build the enriched model list used by `models_page`, `models_create`,
/// and `models_delete`. Each entry from `/model/list` is augmented with the
/// matching `/model/info` payload (provider, pricing, token ceilings).
///
/// Calls are fanned out with bounded concurrency (`CONCURRENCY`) so a proxy
/// with many deployments does not pay one serial round-trip per model.
/// Enrichment is also capped at `ENRICHMENT_CAP` models so a very large list
/// cannot saturate the proxy client's connection pool; entries past the cap
/// are rendered without pricing. The aggregate fan-out is bounded by
/// `PAGE_ENRICHMENT_DEADLINE` so a degraded proxy cannot tie up a worker;
/// entries not enriched before the deadline render without pricing info.
async fn enrich_models(proxy: &ProxyClient, list: Vec<Value>) -> Vec<Value> {
    enrich_models_bounded(proxy, list, crate::pages::spend::PAGE_ENRICHMENT_DEADLINE).await
}

async fn enrich_models_bounded(
    proxy: &ProxyClient,
    list: Vec<Value>,
    deadline: std::time::Duration,
) -> Vec<Value> {
    use futures_util::stream::{self, StreamExt};
    use std::time::Instant;

    const CONCURRENCY: usize = 8;
    const ENRICHMENT_CAP: usize = 200;

    let n = list.len();
    let mut enriched: Vec<Value> = Vec::with_capacity(n);

    let tasks: Vec<(usize, String)> = list
        .iter()
        .enumerate()
        .take(ENRICHMENT_CAP)
        .filter_map(|(i, m)| {
            let id = m
                .get("id")
                .or_else(|| m.get("model_name"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())?
                .to_string();
            Some((i, id))
        })
        .collect();

    let infos: std::collections::HashMap<usize, Value> = if tasks.is_empty() {
        std::collections::HashMap::new()
    } else {
        let started = Instant::now();
        let mut stream = stream::iter(tasks)
            .map(|(i, id)| async move {
                let info = proxy.model_info(&id).await.unwrap_or_else(|e| {
                    tracing::warn!(error = %e, model = %id, "model_info failed");
                    json!({})
                });
                (i, info)
            })
            .buffer_unordered(CONCURRENCY);
        // Drive the fan-out with an overall deadline (matches
        // `enrich_with_budget_spend_bounded`). Breaking the loop drops the
        // remaining un-started tasks and cancels in-flight ones; models that
        // never produced a result render without pricing info.
        let mut collected: std::collections::HashMap<usize, Value> =
            std::collections::HashMap::new();
        loop {
            if started.elapsed() >= deadline {
                break;
            }
            match stream.next().await {
                Some((i, info)) => {
                    collected.insert(i, info);
                }
                None => break,
            }
        }
        drop(stream);
        collected
    };

    for (i, m) in list.into_iter().enumerate() {
        let Some(info) = infos.get(&i) else {
            enriched.push(m);
            continue;
        };
        let id = m
            .get("id")
            .or_else(|| m.get("model_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let num_or_null = |key: &str| -> Value {
            info.get(key)
                .and_then(|v| {
                    if v.is_null() {
                        None
                    } else if v.is_number() {
                        Some(v.clone())
                    } else {
                        v.as_str()
                            .and_then(|s| s.parse::<f64>().ok())
                            .map(|n| json!(n))
                    }
                })
                .unwrap_or(Value::Null)
        };
        let provider = info
            .get("provider")
            .or_else(|| m.get("provider"))
            .cloned()
            .unwrap_or(json!("unknown"));
        let provider_model = info
            .get("provider_model")
            .or_else(|| info.get("litellm_model"))
            .or_else(|| m.get("provider_model"))
            .cloned()
            .unwrap_or(Value::Null);
        let price_provider = info
            .get("price_provider")
            .or_else(|| info.get("litellm_provider"))
            .cloned()
            .unwrap_or(Value::Null);
        // The DB primary key, present only for DB-backed (dynamically added)
        // models. `POST /model/delete` addresses deployments by this id, so
        // the table template only offers a delete button when it is set.
        let db_model_id = m.get("model_id").cloned().unwrap_or(Value::Null);
        let merged = json!({
            "id": id,
            "model_name": id,
            "model_id": db_model_id,
            "provider": provider,
            "price_provider": price_provider,
            "provider_model": provider_model,
            "max_input_tokens": num_or_null("max_input_tokens"),
            "max_output_tokens": num_or_null("max_output_tokens"),
            "input_cost_per_token": num_or_null("input_cost_per_token"),
            "output_cost_per_token": num_or_null("output_cost_per_token"),
        });
        enriched.push(merged);
    }
    enriched
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy_client::ProxyClient;
    use httpmock::MockServer;
    use serde_json::json;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn enrich_models_uses_bounded_concurrency_not_serial() {
        let server = MockServer::start();
        let n: usize = 16;
        let per_call = Duration::from_millis(60);
        let mut list = Vec::with_capacity(n);
        for i in 0..n {
            let id = format!("m{i}");
            list.push(json!({ "id": id, "provider": "p" }));
            server.mock(move |when, then| {
                when.method(httpmock::Method::GET)
                    .path("/model/info")
                    .query_param("model", format!("m{i}"));
                then.status(200).delay(per_call).body(r#"{"provider":"p"}"#);
            });
        }

        let proxy = ProxyClient::new(&server.base_url(), "sk-test");
        let start = Instant::now();
        let out = enrich_models(&proxy, list).await;
        let elapsed = start.elapsed();

        assert_eq!(out.len(), n);
        for (i, row) in out.iter().enumerate() {
            assert_eq!(row["id"], json!(format!("m{i}")));
        }
        // With concurrency 8 and 16 calls of 60ms, ~2 batches = ~120ms.
        // Serial would be ~960ms. Allow generous slack for CI jitter but
        // assert well clear of the serial lower bound.
        let serial_bound = per_call * n as u32;
        assert!(
            elapsed < serial_bound * 3 / 4,
            "enrich_models took {elapsed:?} — looks serial (serial bound {serial_bound:?}, n={n})"
        );
    }

    /// MEDIUM #1: `enrich_models` must bound the aggregate wall-clock time of
    /// its `/model/info` fan-out so a degraded proxy cannot tie up a worker.
    /// With a short deadline and a slow proxy, the loop must stop early: only
    /// a subset of models get pricing info, and the call returns in well under
    /// the unbounded worst case.
    #[tokio::test]
    async fn enrich_models_stops_at_overall_deadline() {
        let server = MockServer::start();
        let n: usize = 32;
        let per_call = Duration::from_millis(500);
        let mut list = Vec::with_capacity(n);
        for i in 0..n {
            let id = format!("m{i}");
            list.push(json!({ "id": id, "provider": "orig" }));
            server.mock(move |when, then| {
                when.method(httpmock::Method::GET)
                    .path("/model/info")
                    .query_param("model", format!("m{i}"));
                then.status(200)
                    .delay(per_call)
                    .body(r#"{"provider":"p","litellm_provider":"p"}"#);
            });
        }

        let proxy = ProxyClient::new(&server.base_url(), "sk-test");
        let started = Instant::now();
        let out = enrich_models_bounded(&proxy, list, Duration::from_millis(150)).await;
        let elapsed = started.elapsed();

        assert_eq!(out.len(), n);
        // An enriched model carries the pricing provider from /model/info;
        // an un-enriched one falls back to the original "orig" provider (no
        // price_provider). The deadline (150 ms) is well under one batch
        // delay (500 ms), so only a subset are enriched.
        let enriched = out
            .iter()
            .filter(|r| {
                r.get("price_provider")
                    .map(|v| !v.is_null())
                    .unwrap_or(false)
            })
            .count();
        assert!(
            enriched >= 1 && enriched < n,
            "expected some-but-not-all models enriched, got {enriched}/{n}"
        );
        // Unbounded worst case: ceil(32/8) * 500 ms = 2000 ms.
        assert!(
            elapsed < Duration::from_millis(1500),
            "enrich_models ran {elapsed:?} — deadline did not bound the fan-out"
        );
    }
}
