//! MiniJinja template environment, with templates embedded at compile time via
//! `include_str!`.

use minijinja::{Environment, value::Value as MjValue};
use serde::Serialize;

/// Holds a compile-time-populated MiniJinja environment.
#[derive(Clone)]
pub struct Templates {
    env: Environment<'static>,
}

impl Templates {
    /// Build the environment with all known templates registered.
    pub fn new() -> Self {
        let mut env = Environment::new();
        env.set_auto_escape_callback(|name| {
            if name.ends_with(".html") {
                minijinja::AutoEscape::Html
            } else {
                minijinja::AutoEscape::None
            }
        });
        // Percent-encode a value for use as a single URL **path segment**.
        // MiniJinja's built-in `urlencode` is query-string oriented (it keeps
        // `/` literal, so `openai/gpt-4o` would still split the path) and is
        // not compiled in by default; this filter escapes everything outside
        // the RFC 3986 unreserved set so ids interpolated into `hx-post`
        // URLs (`/models/{id}/delete`, …) survive `/`, `+`, `=`, and spaces.
        // Axum's `Path` extractor percent-decodes the segment back verbatim.
        env.add_filter("urlencode_path", |value: String| {
            crate::urlencode::urlencoding_minimal(&value)
        });
        register(
            &mut env,
            "base.html",
            include_str!("../templates/base.html"),
        );
        register(
            &mut env,
            "login.html",
            include_str!("../templates/login.html"),
        );
        register(
            &mut env,
            "partials/flash.html",
            include_str!("../templates/partials/flash.html"),
        );
        register(
            &mut env,
            "partials/error_alert.html",
            include_str!("../templates/partials/error_alert.html"),
        );
        register(
            &mut env,
            "partials/empty.html",
            include_str!("../templates/partials/empty.html"),
        );
        register(
            &mut env,
            "partials/list_status.html",
            include_str!("../templates/partials/list_status.html"),
        );
        register(
            &mut env,
            "partials/keys_table.html",
            include_str!("../templates/partials/keys_table.html"),
        );
        register(
            &mut env,
            "partials/key_form.html",
            include_str!("../templates/partials/key_form.html"),
        );
        register(
            &mut env,
            "partials/key_created.html",
            include_str!("../templates/partials/key_created.html"),
        );
        register(
            &mut env,
            "partials/teams_table.html",
            include_str!("../templates/partials/teams_table.html"),
        );
        register(
            &mut env,
            "partials/team_form.html",
            include_str!("../templates/partials/team_form.html"),
        );
        register(
            &mut env,
            "partials/users_table.html",
            include_str!("../templates/partials/users_table.html"),
        );
        register(
            &mut env,
            "partials/models_table.html",
            include_str!("../templates/partials/models_table.html"),
        );
        register(
            &mut env,
            "partials/user_form.html",
            include_str!("../templates/partials/user_form.html"),
        );
        register(
            &mut env,
            "partials/spend_table.html",
            include_str!("../templates/partials/spend_table.html"),
        );
        register(
            &mut env,
            "partials/spend_rows.html",
            include_str!("../templates/partials/spend_rows.html"),
        );
        register(
            &mut env,
            "partials/spend_load_more.html",
            include_str!("../templates/partials/spend_load_more.html"),
        );
        register(
            &mut env,
            "partials/spend_pagination.html",
            include_str!("../templates/partials/spend_pagination.html"),
        );
        register(
            &mut env,
            "partials/dashboard_summary.html",
            include_str!("../templates/partials/dashboard_summary.html"),
        );
        register(
            &mut env,
            "pages/dashboard.html",
            include_str!("../templates/pages/dashboard.html"),
        );
        register(
            &mut env,
            "pages/keys.html",
            include_str!("../templates/pages/keys.html"),
        );
        register(
            &mut env,
            "pages/teams.html",
            include_str!("../templates/pages/teams.html"),
        );
        register(
            &mut env,
            "pages/users.html",
            include_str!("../templates/pages/users.html"),
        );
        register(
            &mut env,
            "pages/organizations.html",
            include_str!("../templates/pages/organizations.html"),
        );
        register(
            &mut env,
            "pages/projects.html",
            include_str!("../templates/pages/projects.html"),
        );
        register(
            &mut env,
            "partials/organizations_table.html",
            include_str!("../templates/partials/organizations_table.html"),
        );
        register(
            &mut env,
            "partials/projects_table.html",
            include_str!("../templates/partials/projects_table.html"),
        );
        register(
            &mut env,
            "partials/organization_form.html",
            include_str!("../templates/partials/organization_form.html"),
        );
        register(
            &mut env,
            "partials/project_form.html",
            include_str!("../templates/partials/project_form.html"),
        );
        register(
            &mut env,
            "pages/spend.html",
            include_str!("../templates/pages/spend.html"),
        );
        register(
            &mut env,
            "pages/models.html",
            include_str!("../templates/pages/models.html"),
        );
        register(
            &mut env,
            "pages/calculator.html",
            include_str!("../templates/pages/calculator.html"),
        );
        register(
            &mut env,
            "partials/calculator_result.html",
            include_str!("../templates/partials/calculator_result.html"),
        );
        Self { env }
    }

    /// Render a template with a serializable context.
    pub fn render<S: Serialize>(&self, name: &str, ctx: &S) -> Result<String, minijinja::Error> {
        let template = self.env.get_template(name)?;
        let value = MjValue::from_serialize(ctx);
        template.render(&value)
    }
}

impl Default for Templates {
    fn default() -> Self {
        Self::new()
    }
}

fn register(env: &mut Environment<'static>, name: &'static str, src: &'static str) {
    if let Err(e) = env.add_template(name, src) {
        panic!("template registration failed for {name}: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_login_with_error() {
        let t = Templates::new();
        let html = t
            .render("login.html", &serde_json::json!({"error": "bad password"}))
            .expect("render");
        assert!(html.contains("oxllm Admin"));
        assert!(html.contains("bad password"));
    }

    #[test]
    fn renders_login_without_error() {
        let t = Templates::new();
        let html = t
            .render("login.html", &serde_json::json!({"error": null}))
            .expect("render");
        assert!(html.contains("<form"));
        assert!(!html.contains("bad password"));
    }

    #[test]
    fn renders_dashboard_with_summary() {
        let t = Templates::new();
        let html = t
            .render(
                "pages/dashboard.html",
                &serde_json::json!({
                "session_active": true,
                "total_spend": 1.23,
                "total_requests": 5,
                "health": {"healthy": true, "status": "ok", "reachable": true},
                "activity": [
                    {"day": "2024-01-01", "spend": 0.5, "bar_pct": 50.0},
                    {"day": "2024-01-02", "spend": 1.0, "bar_pct": 100.0}
                ],
                "activity_days": 30
                }),
            )
            .expect("render");
        assert!(html.contains("Dashboard"));
        assert!(html.contains("/logout"));
        assert!(html.contains("$1.2300"));
        assert!(html.contains("Total spend") || html.contains("Total Spend"));
        assert!(
            html.contains("width: 100%") || html.contains("width: 100.0%"),
            "activity bar should use precomputed bar_pct"
        );
        assert!(html.contains("width: 50%") || html.contains("width: 50.0%"));
    }

    #[test]
    fn spend_rows_render_missing_and_present_cost() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/spend_rows.html",
                &serde_json::json!({
                    "rows": [
                        {"timestamp": "t1", "model": "gpt-4o", "cost": 0.00125, "provider": "openai"},
                        {"timestamp": "t2", "model": "mystery", "provider": "xai"}
                    ]
                }),
            )
            .expect("render");
        assert!(html.contains("ox-cost"));
        assert!(html.contains("0.00125000") || html.contains("0.00125"));
        assert!(html.contains("ox-cost-missing") || html.contains("—"));
        assert!(html.contains("xAI") || html.contains("xai"));
    }

    #[test]
    fn base_includes_csrf_and_alpine_htmx_hooks() {
        let t = Templates::new();
        let html = t
            .render(
                "pages/keys.html",
                &serde_json::json!({
                    "session_active": true,
                    "csrf_token": "test-csrf-token",
                    "keys": []
                }),
            )
            .expect("render");
        assert!(html.contains("csrf-token"));
        assert!(html.contains("test-csrf-token"));
        assert!(html.contains("document.addEventListener('htmx:configRequest'"));
        assert!(html.contains("Alpine.initTree") || html.contains("oxInitAlpine"));
        assert!(!html.contains("document.body.addEventListener('htmx:configRequest'"));
    }

    #[test]
    fn renders_keys_table_with_rows() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/keys_table.html",
                &serde_json::json!({
                "keys": [
                {"key_id": "key-1", "key_alias": "alpha", "is_active": true}
                ]
                }),
            )
            .expect("render");
        assert!(html.contains("key-1"));
        assert!(html.contains("alpha"));
        assert!(html.contains("/keys/key-1/delete"));
        assert!(html.contains("Disable") || html.contains("Delete"));
    }

    #[test]
    fn renders_spend_table_with_rows() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/spend_table.html",
                &serde_json::json!({
                "logs": [{"timestamp": "2024-01-01", "model": "gpt-4o", "cost": 0.001, "key_id": "key-1", "key_alias": "prod"}],
                "rows": [{"timestamp": "2024-01-01", "model": "gpt-4o", "cost": 0.001, "key_id": "key-1", "key_alias": "prod"}],
                "filters": {},
                "next_cursor": null,
                "sort_by": "timestamp",
                "sort_order": "desc",
                "keys": [{"key_id": "key-1", "key_alias": "prod"}],
                "teams": [],
                "list_query": "sort_by=timestamp&sort_order=desc&start_date=2024-01-01&end_date=2024-01-31",
                "export_query": "sort_by=timestamp&sort_order=desc&start_date=2024-01-01&end_date=2024-01-31",
                "export_start": "2024-01-01",
                "export_end": "2024-01-31"
                }),
            )
            .expect("render");
        assert!(html.contains("gpt-4o"));
        assert!(html.contains("spend-rows"));
        assert!(html.contains("prod"));
        assert!(html.contains("Export CSV"));
        assert!(html.contains("2024-01-01") && html.contains("2024-01-31"));
        assert!(html.contains("sort_by=model") || html.contains("Model"));
        assert!(html.contains("name=\"start_date\""));
        assert!(html.contains("name=\"end_date\""));
        // The totals strip aggregates the whole filter scope, not the loaded
        // page — the label must say so (an operator reconciling page rows
        // against a "This page" sum sees mismatched numbers).
        assert!(html.contains("Filtered total"), "strip label: {html}");
        assert!(!html.contains("This page:"), "strip label: {html}");
    }

    #[test]
    fn renders_all_pages() {
        let t = Templates::new();
        for name in [
            "pages/keys.html",
            "pages/teams.html",
            "pages/users.html",
            "pages/organizations.html",
            "pages/projects.html",
            "pages/spend.html",
            "pages/models.html",
            "pages/calculator.html",
        ] {
            let html = t
                .render(
                    name,
                    &serde_json::json!({
                        "session_active": true,
                        "models": [],
                        "result": null,
                        "organizations": [],
                        "projects": [],
                        "teams": [],
                        "users": [],
                        "keys": [],
                    }),
                )
                .unwrap_or_else(|e| panic!("render {name}: {e}"));
            assert!(!html.is_empty(), "{name} produced empty output");
        }
    }

    #[test]
    fn renders_models_table_with_real_and_empty_costs() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/models_table.html",
                &serde_json::json!({
                    "models": [
                        {
                            "id": "gpt-4o",
                            "provider": "openai",
                            "input_cost_per_token": 0.000005,
                            "output_cost_per_token": 0.000015,
                            "max_input_tokens": 128000,
                            "max_output_tokens": 16384
                        },
                        {
                            "id": "mystery",
                            "provider": "unknown",
                            "input_cost_per_token": "",
                            "output_cost_per_token": "",
                            "max_input_tokens": "",
                            "max_output_tokens": ""
                        },
                        {
                            "id": "bare"
                        }
                    ]
                }),
            )
            .expect("models_table should render");
        assert!(html.contains("gpt-4o"));
        assert!(html.contains("mystery"));
        assert!(html.contains("bare"));
    }

    /// The delete button's hx-post URL must percent-encode the model id: a
    /// raw id containing `/` would split into multiple path segments and
    /// 404 against the `/models/{model_id}/delete` route. Only DB-backed
    /// models (with a `model_id`) get the button at all.
    #[test]
    fn renders_models_table_with_encoded_delete_url() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/models_table.html",
                &serde_json::json!({
                    "models": [
                        {"id": "db-model", "model_id": "model-a/b+c"},
                        {"id": "yaml-model"}
                    ]
                }),
            )
            .expect("models_table should render");
        assert!(
            html.contains("/models/model-a%2Fb%2Bc/delete"),
            "delete URL must be path-encoded: {html}"
        );
        // The config-managed row offers no delete action at all.
        assert!(
            !html.contains("/models/yaml-model"),
            "config-managed model must not be deletable: {html}"
        );
    }

    #[test]
    fn renders_models_page_with_models() {
        let t = Templates::new();
        let html = t
            .render(
                "pages/models.html",
                &serde_json::json!({
                    "session_active": true,
                    "models": [{"id": "gpt-4o", "provider": "openai", "input_cost_per_token": 0.000005, "output_cost_per_token": 0.000015}]
                }),
            )
            .expect("models page");
        assert!(html.contains("gpt-4o"));
        assert!(html.contains("Configured Models"));
    }

    #[test]
    fn flash_partial_has_status_role_and_autodismiss() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/flash.html",
                &serde_json::json!({"flash": "saved"}),
            )
            .expect("render flash");
        assert!(html.contains("ox-flash-success"));
        assert!(html.contains("role=\"status\""));
        assert!(html.contains("aria-live=\"polite\""));
        // Auto-dismiss timer + dismiss button.
        assert!(html.contains("setTimeout"));
        assert!(html.contains("Dismiss notification"));
        // Hidden when no flash.
        let empty = t
            .render("partials/flash.html", &serde_json::json!({}))
            .expect("render empty flash");
        assert!(!empty.contains("ox-flash"));
    }

    #[test]
    fn error_alert_has_alert_role_and_dismiss() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/error_alert.html",
                &serde_json::json!({"message": "boom"}),
            )
            .expect("render error");
        assert!(html.contains("role=\"alert\""));
        assert!(html.contains("aria-live=\"assertive\""));
        assert!(html.contains("Dismiss alert"));
        assert!(html.contains("setTimeout"));
    }

    #[test]
    fn base_has_skip_link_and_main_content_anchor() {
        let t = Templates::new();
        let html = t
            .render(
                "pages/keys.html",
                &serde_json::json!({
                    "session_active": true,
                    "csrf_token": "tok",
                    "keys": []
                }),
            )
            .expect("render keys page");
        assert!(html.contains("ox-skip-link"));
        assert!(html.contains("Skip to content"));
        assert!(html.contains("href=\"#main-content\""));
        assert!(html.contains("id=\"main-content\""));
    }

    #[test]
    fn keys_table_renders_budget_bar_when_max_budget_set() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/keys_table.html",
                &serde_json::json!({
                    "keys": [{
                        "key_id": "key-1",
                        "key_alias": "alpha",
                        "is_active": true,
                        "max_budget": 100.0,
                        "budget_spend": 85.0,
                        "budget_spend_pct": 85.0,
                        "budget_spend_tier": "danger"
                    }, {
                        "key_id": "key-2",
                        "key_alias": "beta",
                        "is_active": true
                    }]
                }),
            )
            .expect("render keys table");
        // Bar with progress fill.
        assert!(html.contains("ox-budget-bar"));
        assert!(html.contains("ox-budget-fill"));
        assert!(html.contains("ox-budget-fill-danger"));
        assert!(html.contains("width: 85"));
        // Dollar formatting.
        assert!(html.contains("$85.00"));
        assert!(html.contains("$100.00"));
        // Empty budget cell renders em-dash.
        assert!(html.contains(">—<"));
    }

    #[test]
    fn teams_table_renders_budget_bar_when_max_budget_set() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/teams_table.html",
                &serde_json::json!({
                    "teams": [{
                        "team_id": "team-1",
                        "team_name": "Eng",
                        "max_budget": 50.0,
                        "budget_spend": 10.0,
                        "budget_spend_pct": 20.0,
                        "budget_spend_tier": "ok"
                    }]
                }),
            )
            .expect("render teams table");
        assert!(html.contains("ox-budget-fill-ok"));
        assert!(html.contains("$10.00"));
        assert!(html.contains("$50.00"));
    }

    #[test]
    fn users_table_renders_budget_bar_when_max_budget_set() {
        let t = Templates::new();
        let html = t
            .render(
                "partials/users_table.html",
                &serde_json::json!({
                    "users": [{
                        "user_id": "u1",
                        "user_name": "Ada",
                        "max_budget": 25.0,
                        "budget_spend": 15.0,
                        "budget_spend_pct": 60.0,
                        "budget_spend_tier": "warn"
                    }]
                }),
            )
            .expect("render users table");
        assert!(html.contains("ox-budget-fill-warn"));
        assert!(html.contains("$15.00"));
        assert!(html.contains("$25.00"));
    }
}
