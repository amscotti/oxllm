//! Page-domain handlers, one module per admin UI section.

pub(crate) mod auth;
pub(crate) mod calculator;
pub(crate) mod dashboard;
pub(crate) mod keys;
pub(crate) mod models;
pub(crate) mod orgs;
pub(crate) mod projects;
pub(crate) mod spend;
pub(crate) mod teams;
pub(crate) mod users;

use serde_json::Value;

/// Wire marker carried by management model-policy replies. This HTTP adapter
/// stays independent of gateway implementation crates.
const MODELS_ACL_DENY_ALL: &str = "\0oxllm-acl-deny-all";

/// Parse an optional numeric form field (e.g. `max_budget`).
///
/// Empty / whitespace-only input is `Ok(None)` (field omitted on create,
/// cleared on update). Any other input that does not parse as a finite,
/// non-negative number is `Err(message)` — the caller must surface the error
/// to the user instead of silently dropping the value (which previously made
/// e.g. `keys_update` report success while keeping the old budget).
pub(crate) fn parse_optional_f64(raw: &str, field: &str) -> Result<Option<f64>, String> {
    let t = raw.trim();
    if t.is_empty() {
        return Ok(None);
    }
    match t.parse::<f64>() {
        Ok(v) if v.is_finite() && v >= 0.0 => Ok(Some(v)),
        Ok(_) => Err(format!("{field} must be a non-negative number")),
        Err(_) => Err(format!("{field} must be a number")),
    }
}

/// Integer variant of [`parse_optional_f64`] for `rpm` / `tpm` fields.
pub(crate) fn parse_optional_i64(raw: &str, field: &str) -> Result<Option<i64>, String> {
    let t = raw.trim();
    if t.is_empty() {
        return Ok(None);
    }
    match t.parse::<i64>() {
        Ok(v) if v >= 0 => Ok(Some(v)),
        Ok(_) => Err(format!("{field} must be a non-negative integer")),
        Err(_) => Err(format!("{field} must be an integer")),
    }
}

/// Parse comma-separated model patterns or a JSON string array. Returns
/// `None` when the input is empty/whitespace so the field is omitted from
/// the payload (preserving the "no restriction" default).
pub(crate) fn parse_csv_models(raw: &str) -> Option<Vec<String>> {
    if let Ok(models) = serde_json::from_str::<Vec<String>>(raw.trim()) {
        return (!models.is_empty()).then_some(models);
    }
    let out: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if out.is_empty() { None } else { Some(out) }
}

fn format_models_for_display(models: &[&str]) -> String {
    if !models.is_empty() && models.iter().all(|model| model.trim().is_empty()) {
        return serde_json::json!([MODELS_ACL_DENY_ALL]).to_string();
    }
    if models.iter().any(|model| {
        model.contains(',')
            || model.starts_with('[')
            || model.trim() != *model
            || model.chars().any(char::is_control)
    }) {
        serde_json::json!(models).to_string()
    } else {
        models.join(", ")
    }
}

/// Convert a `models` array or legacy JSON-string column on
/// each record into editable text. Use a JSON array when comma separation would
/// lose a model pattern. Mutates values in place; values without a `models` field or
/// with an invalid nonempty policy receive the same deny-all marker as the
/// gateway. Saving unrelated fields must not reinterpret stored data as CSV.
pub(crate) fn normalize_models_for_display(rows: &mut [Value]) {
    for row in rows.iter_mut() {
        if let Some(obj) = row.as_object_mut() {
            let csv = match obj.get("models") {
                Some(Value::String(models)) => Some(
                    serde_json::from_str::<Vec<String>>(models)
                        .map(|models| {
                            format_models_for_display(
                                &models.iter().map(String::as_str).collect::<Vec<_>>(),
                            )
                        })
                        .unwrap_or_else(|_| {
                            if models.trim().is_empty() {
                                String::new()
                            } else {
                                format_models_for_display(&[MODELS_ACL_DENY_ALL])
                            }
                        }),
                ),
                Some(Value::Array(models)) => models
                    .iter()
                    .map(Value::as_str)
                    .collect::<Option<Vec<_>>>()
                    .map(|models| format_models_for_display(&models))
                    .or_else(|| Some(format_models_for_display(&[MODELS_ACL_DENY_ALL]))),
                _ => None,
            };
            if let Some(csv) = csv {
                obj.insert("models".to_string(), Value::String(csv));
            }
        }
    }
}

#[cfg(test)]
mod model_display_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_arrays_and_legacy_columns_preserve_editable_model_policies() {
        let mut rows = [
            json!({"models":["gpt-4o","claude-*"]}),
            json!({"models":"[\"gpt-4o\",\"claude-*\"]"}),
            json!({"models":[]}),
            json!({"models":null}),
            json!({}),
        ];
        normalize_models_for_display(&mut rows);
        assert_eq!(rows[0]["models"], "gpt-4o, claude-*");
        assert_eq!(rows[1]["models"], rows[0]["models"]);
        assert_eq!(rows[2]["models"], "");
        assert!(rows[3]["models"].is_null() && rows[4].get("models").is_none());
    }

    #[test]
    fn model_policies_with_commas_or_json_like_names_round_trip_without_splitting() {
        let patterns = ["model,version", "[\"literal-model\"]", "claude-*"];
        let mut rows = [
            json!({"models":patterns}),
            json!({"models":json!(patterns).to_string()}),
        ];
        normalize_models_for_display(&mut rows);
        for row in rows {
            assert_eq!(
                parse_csv_models(row["models"].as_str().unwrap()).unwrap(),
                patterns
            );
        }
        assert_eq!(
            parse_csv_models("gpt-4o, claude-*").unwrap(),
            ["gpt-4o", "claude-*"]
        );
        assert!(parse_csv_models("[]").is_none());
    }

    #[test]
    fn control_characters_are_escaped_before_entering_html_text_inputs() {
        for pattern in [
            "\0oxllm-acl-deny-all",
            "model\nversion",
            "model\rversion",
            "model\tversion",
        ] {
            let mut rows = [json!({"models":[pattern]})];
            normalize_models_for_display(&mut rows);
            let field = rows[0]["models"].as_str().unwrap();
            assert!(
                !field.chars().any(char::is_control),
                "HTML inputs must receive escaped model text"
            );
            assert_eq!(parse_csv_models(field).unwrap(), [pattern]);
        }
    }

    #[test]
    fn corrupt_stored_policies_are_not_reinterpreted_as_csv_on_unrelated_edits() {
        for raw in [
            "not-json", "[1, *]", "[1,2]", "\"*\"", "*", "[\"\"]", "[\"  \"]",
        ] {
            let mut rows = [json!({"models":raw})];
            normalize_models_for_display(&mut rows);
            let field = rows[0]["models"].as_str().unwrap();
            assert_eq!(parse_csv_models(field).unwrap(), ["\0oxllm-acl-deny-all"]);
            assert!(!field.chars().any(char::is_control));
        }
        let mut blank = [json!({"models":"  "})];
        normalize_models_for_display(&mut blank);
        assert!(parse_csv_models(blank[0]["models"].as_str().unwrap()).is_none());
    }
}
