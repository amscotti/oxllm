//! CSV escaping and spend-log serialization for the export endpoint.

use serde_json::Value;

pub(crate) fn csv_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    let needs_quote = s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r');
    if needs_quote {
        out.push('"');
        out.push_str(&s.replace('"', "\"\""));
        out.push('"');
    } else {
        out.push_str(s);
    }
    out
}

/// Prefix a leading formula-trigger character with `'` so spreadsheet apps
/// treat the cell as a literal string (OWASP CSV injection mitigation).
/// The prefix is not displayed by Excel/LibreOffice/Google Sheets.
///
/// `-` is intentionally NOT a trigger: Excel treats a leading `-` as a
/// formula, but legitimate negative numeric cells (e.g. a cost adjustment
/// of `-0.5`) are common in the exported data, and quoting them turns the
/// column into text. A cell that parses as a number is safe anyway.
fn neutralize_formula(cell: &str) -> String {
    let trimmed = cell.trim_start_matches(' ');
    match trimmed.chars().next() {
        Some('=' | '+' | '@' | '\t' | '\r') => {
            let mut out = String::with_capacity(cell.len() + 1);
            out.push('\'');
            out.push_str(cell);
            out
        }
        _ => cell.to_string(),
    }
}

pub(crate) fn csv_cell(s: &str) -> String {
    csv_escape(&neutralize_formula(s))
}

pub(crate) fn spend_logs_to_csv(logs: &[Value]) -> String {
    let mut out = String::from(
        "timestamp,request_id,model,provider,prompt_tokens,completion_tokens,total_tokens,cost,key_id,key_alias,team_id,team_name,status\n",
    );
    for row in logs {
        let get = |k: &str| -> String {
            match row.get(k) {
                Some(Value::Null) | None => String::new(),
                Some(Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
            }
        };
        let line = [
            get("timestamp"),
            get("request_id"),
            get("model"),
            get("provider"),
            get("prompt_tokens"),
            get("completion_tokens"),
            get("total_tokens"),
            get("cost"),
            get("key_id"),
            get("key_alias"),
            get("team_id"),
            get("team_name"),
            get("status"),
        ]
        .into_iter()
        .map(|s| csv_cell(&s))
        .collect::<Vec<_>>()
        .join(",");
        out.push_str(&line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_escape_plain_value_unchanged() {
        assert_eq!(csv_cell("prod-key"), "prod-key");
    }

    #[test]
    fn csv_escape_quotes_commas_and_quotes() {
        assert_eq!(csv_cell("a,b"), "\"a,b\"");
        assert_eq!(csv_cell("a\"b"), "\"a\"\"b\"");
        assert_eq!(csv_cell("line\nbreak"), "\"line\nbreak\"");
    }

    #[test]
    fn formula_injection_is_neutralized() {
        for trigger in ["=cmd|'/c calc'!A1", "+1+1", "@SUM(A1)", "\tfoo"] {
            let escaped = csv_cell(trigger);
            assert!(
                escaped.starts_with('\'') || escaped.starts_with("\"'"),
                "trigger {trigger:?} not neutralized: got {escaped:?}"
            );
        }
    }

    #[test]
    fn negative_numeric_cell_is_not_quoted() {
        // A leading `-` is treated as a plain (negative) number, not a
        // formula trigger, so cost columns stay numeric on import.
        assert_eq!(csv_cell("-0.5"), "-0.5");
        assert_eq!(csv_cell("-2+3"), "-2+3");
    }

    #[test]
    fn formula_prefix_applied_even_when_quoted() {
        let escaped = csv_cell("=HYPERLINK(\"https://evil/x\",\"click\")");
        assert!(escaped.starts_with("\"'="));
    }

    #[test]
    fn benign_alias_unchanged() {
        assert_eq!(csv_cell("prod-key"), "prod-key");
        assert_eq!(
            csv_cell("anthropic.claude-3.5-sonnet"),
            "anthropic.claude-3.5-sonnet"
        );
    }

    #[test]
    fn spend_logs_to_csv_neutralizes_formula_cells() {
        let row = serde_json::json!({
            "timestamp": "2026-01-01T00:00:00Z",
            "request_id": "=cmd|'/c calc'!A1",
            "model": "+1+1",
            "provider": "openai",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cost": 0.001,
            "key_id": "k1",
            "key_alias": "prod-key",
            "team_id": null,
            "team_name": "",
            "status": "success",
        });
        let csv = spend_logs_to_csv(&[row]);
        let data_line = csv.lines().nth(1).unwrap();
        for cell in [
            "'=cmd|'/c calc'!A1",
            "'+1+1",
            "prod-key",
            "openai",
            "success",
        ] {
            assert!(
                data_line.contains(cell),
                "expected cell {cell:?} in line: {data_line}"
            );
        }
    }
}
