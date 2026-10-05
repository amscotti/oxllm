//! Runtime skill injection — prepend matching skill content into the chat
//! system prompt before the request is metered and forwarded upstream.
//!
//! Skills are versioned instruction files stored via `/skill/*` management
//! endpoints. At request time, every skill whose `model_pattern` matches the
//! requested model (or has no pattern) and whose optional `team_id` is either
//! unset or equal to the caller's team is selected. Contents are concatenated
//! and either prepended to an existing system message or inserted as a new
//! leading system message.
//!
//! The matching lookup is served by a short-TTL cache in `oxllm-db`
//! (`list_skills_by_model`), so the hot path does not hit the database on
//! every request; skill writes invalidate it immediately in-process.
//!
//! No-ops when the `db` feature is off, no database is configured, or no
//! skills match. Fail-open on DB errors (log + continue without injection)
//! so a skills-table outage cannot take down chat.

use serde_json::{Value, json};

use crate::auth::AuthUser;
use crate::state::AppState;

/// Marker wrapping injected skill bodies so operators can spot them in
/// upstream request logs and so unit tests can assert on the boundary.
const SKILL_BLOCK_START: &str = "<!-- oxllm-skills -->";
const SKILL_BLOCK_END: &str = "<!-- /oxllm-skills -->";

/// Load matching skills for `model` / `auth` and inject their content into
/// `request_body.messages`. Mutates the body in place. Safe to call on
/// non-chat bodies (no `messages` array) — returns without change.
#[cfg(feature = "db")]
pub async fn inject_skills_into_request(
    state: &AppState,
    auth: &AuthUser,
    model: &str,
    request_body: &mut Value,
) {
    let Some(db) = state.db.as_ref() else {
        return;
    };
    let skills = match db.list_skills_by_model(model).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                error = %e,
                model,
                "skills lookup failed; continuing without skill injection"
            );
            return;
        }
    };
    if skills.is_empty() {
        return;
    }

    // Team filter: a skill with `team_id` only applies when the caller
    // belongs to that team. Skills without a team apply to every principal
    // (including master / open mode). Master keys without a team therefore
    // receive only unscoped skills — intentional, so team-private skills
    // never leak onto master-key traffic.
    let team_id = auth.team_id.as_deref();
    let mut contents: Vec<String> = Vec::new();
    for skill in skills {
        if let Some(ref skill_team) = skill.team_id {
            if skill_team.is_empty() {
                // empty string treated as unrestricted
            } else if team_id != Some(skill_team.as_str()) {
                continue;
            }
        }
        let body = skill.content.trim();
        if body.is_empty() {
            continue;
        }
        // Label each skill so multi-skill injection is auditable.
        contents.push(format!("# Skill: {}\n\n{}", skill.name, body));
    }
    if contents.is_empty() {
        return;
    }

    let block = format!(
        "{SKILL_BLOCK_START}\n{}\n{SKILL_BLOCK_END}",
        contents.join("\n\n---\n\n")
    );
    inject_system_block(request_body, &block);
    tracing::debug!(
        model,
        skill_count = contents.len(),
        "injected skill content into system prompt"
    );
}

/// Prepend `block` to the first system message's text content, or insert a
/// new leading system message when none exists. Only mutates chat-shaped
/// bodies with a `messages` array.
pub fn inject_system_block(request_body: &mut Value, block: &str) {
    let Some(messages) = request_body
        .get_mut("messages")
        .and_then(|m| m.as_array_mut())
    else {
        return;
    };
    if messages.is_empty() {
        return;
    }

    // Find the first system message.
    let system_idx = messages.iter().position(|m| {
        m.get("role")
            .and_then(|r| r.as_str())
            .is_some_and(|r| r == "system")
    });

    match system_idx {
        Some(idx) => {
            let msg = &mut messages[idx];
            match msg.get_mut("content") {
                Some(Value::String(s)) => {
                    if s.is_empty() {
                        *s = block.to_string();
                    } else {
                        *s = format!("{block}\n\n{s}");
                    }
                }
                Some(Value::Array(parts)) => {
                    // Multimodal system content: prepend a text part.
                    parts.insert(0, json!({"type": "text", "text": block}));
                }
                Some(_) | None => {
                    // Non-string / missing content — replace with the block.
                    if let Some(obj) = msg.as_object_mut() {
                        obj.insert("content".to_string(), Value::String(block.to_string()));
                    }
                }
            }
        }
        None => {
            messages.insert(
                0,
                json!({
                    "role": "system",
                    "content": block,
                }),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn inject_prepends_new_system_message_when_absent() {
        let mut body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        });
        inject_system_block(&mut body, "SKILL BODY");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "SKILL BODY");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn inject_prepends_to_existing_system_string() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hi"}
            ]
        });
        inject_system_block(&mut body, "SKILL");
        assert_eq!(
            body["messages"][0]["content"].as_str().unwrap(),
            "SKILL\n\nyou are helpful"
        );
    }

    #[test]
    fn inject_prepends_text_part_to_array_system_content() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "base"}]},
                {"role": "user", "content": "hi"}
            ]
        });
        inject_system_block(&mut body, "SKILL");
        let parts = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "SKILL");
        assert_eq!(parts[1]["text"], "base");
    }

    #[test]
    fn inject_noops_without_messages() {
        let mut body = json!({"model": "gpt-4o", "prompt": "hi"});
        inject_system_block(&mut body, "SKILL");
        assert!(body.get("messages").is_none());
        assert_eq!(body["prompt"], "hi");
    }

    #[test]
    fn inject_noops_on_empty_messages() {
        let mut body = json!({"messages": []});
        inject_system_block(&mut body, "SKILL");
        assert!(body["messages"].as_array().unwrap().is_empty());
    }

    #[test]
    fn skill_block_markers_are_stable() {
        // The HTML-comment markers are a cross-test contract; keep them stable.
        assert!(SKILL_BLOCK_START.contains("oxllm-skills"));
        assert!(SKILL_BLOCK_END.contains("/oxllm-skills"));
    }
}
