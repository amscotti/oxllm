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

/// Parse a comma-separated list of model patterns into a vec. Returns
/// `None` when the input is empty/whitespace so the field is omitted from
/// the payload (preserving the "no restriction" default).
pub(crate) fn parse_csv_models(raw: &str) -> Option<Vec<String>> {
    let out: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if out.is_empty() { None } else { Some(out) }
}

/// Convert a `models` JSON-string column (e.g. `["gpt-4o", "claude-*"]`) on
/// each record of a list response into a CSV string for display in the admin
/// UI forms. Mutates the values in place; values without a `models` field or
/// with an unparseable value are left untouched.
pub(crate) fn normalize_models_for_display(rows: &mut [Value]) {
    for row in rows.iter_mut() {
        if let Some(obj) = row.as_object_mut() {
            if let Some(models) = obj
                .get("models")
                .and_then(|v| v.as_str())
                .map(str::to_string)
            {
                let csv = match serde_json::from_str::<Vec<String>>(&models) {
                    Ok(v) => v.join(", "),
                    Err(_) => models,
                };
                obj.insert("models".to_string(), Value::String(csv));
            }
        }
    }
}
