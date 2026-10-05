//! Cost calculator page and run handler.

use axum::Form;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::{AppState, is_htmx, proxy_error_response};
use crate::auth::csrf_token_from_headers;
use crate::error::AdminError;
use crate::proxy_client::ListResponse;

pub(crate) async fn calculator_page(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, AdminError> {
    let models = state
        .proxy
        .list_models()
        .await
        .unwrap_or_else(|_| ListResponse::unavailable());
    let csrf = csrf_token_from_headers(&headers, &state.config.session_secret).unwrap_or_default();
    let html = state.templates.render(
        "pages/calculator.html",
        &json!({
            "session_active": true,
            "csrf_token": csrf,
            "models": models.items,
            "models_truncated": models.truncated,
            "models_unavailable": models.unavailable,
            "result": null,
        }),
    )?;
    Ok(Html(html).into_response())
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct CalculatorForm {
    #[serde(default)]
    model: String,
    #[serde(default)]
    prompt_tokens: String,
    #[serde(default)]
    completion_tokens: String,
}

pub(crate) async fn calculator_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<CalculatorForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    // Reject non-numeric / negative token input inline rather than silently
    // coercing to 0 (which would render a misleading "$0.00000000" estimate).
    // `u64::from_str` rejects signs and non-digit characters, so "abc", "1,000",
    // and "-5" all surface as parse errors.
    let prompt: Option<u64> = form.prompt_tokens.trim().parse().ok();
    let completion: Option<u64> = form.completion_tokens.trim().parse().ok();
    let (prompt, completion, parse_error) = match (prompt, completion) {
        (Some(p), Some(c)) => (p, c, None),
        _ => (
            0u64,
            0u64,
            Some("Enter non-negative whole numbers for both token counts."),
        ),
    };
    let models = state
        .proxy
        .list_models()
        .await
        .unwrap_or_else(|_| ListResponse::unavailable());
    if let Some(err_msg) = parse_error {
        let ctx = json!({
            "session_active": true,
            "models": models.items,
            "models_truncated": models.truncated,
            "models_unavailable": models.unavailable,
            "result": null,
            "error": err_msg,
            "form_model": form.model,
            "form_prompt": form.prompt_tokens,
            "form_completion": form.completion_tokens,
        });
        if is_htmx {
            let html = state
                .templates
                .render("partials/calculator_result.html", &ctx)?;
            return Ok(Html(html).into_response());
        }
        let csrf =
            csrf_token_from_headers(&headers, &state.config.session_secret).unwrap_or_default();
        let mut map = match ctx {
            Value::Object(m) => m,
            _ => serde_json::Map::new(),
        };
        map.insert("csrf_token".into(), json!(csrf));
        let html = state
            .templates
            .render("pages/calculator.html", &Value::Object(map))?;
        return Ok(Html(html).into_response());
    }
    let body = json!({
        "model": form.model,
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt.saturating_add(completion),
        }
    });
    let result = match state.proxy.spend_calculate(body).await {
        Ok(v) => v,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    let ctx = json!({
        "session_active": true,
        "models": models.items,
        "models_truncated": models.truncated,
        "models_unavailable": models.unavailable,
        "result": result,
        "form_model": form.model,
        "form_prompt": form.prompt_tokens,
        "form_completion": form.completion_tokens,
    });
    if is_htmx {
        let html = state
            .templates
            .render("partials/calculator_result.html", &ctx)?;
        return Ok(Html(html).into_response());
    }
    let csrf = csrf_token_from_headers(&headers, &state.config.session_secret).unwrap_or_default();
    let mut map = match ctx {
        Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    map.insert("csrf_token".into(), json!(csrf));
    let html = state
        .templates
        .render("pages/calculator.html", &Value::Object(map))?;
    Ok(Html(html).into_response())
}
