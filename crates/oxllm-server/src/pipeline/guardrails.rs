//! Guardrail adapters for raw inference requests. Prompt text not already
//! covered by chat guardrails is projected into messages; mutations are copied
//! back to their original locations.
//! Native content blocks, cache controls, tools, and binary inputs keep their
//! wire shape. Unsupported formats on recognized inference endpoints fail
//! before going upstream when request guardrails are active; endpoints the
//! classifier does not recognize are forwarded unscanned with a one-time
//! warning (platform APIs and configured pass-through targets are not
//! inference formats and must keep working).

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use axum::http::Method;
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::state::AppState;

const MAX_PROJECTED_TEXT_FIELDS: usize = 4096;

/// Upper bound on the distinct unscanned paths remembered for warn-once
/// suppression. Catch-all routes accept arbitrary client paths, so an
/// unbounded set would be a slow memory-growth vector; past the cap requests
/// are still allowed, just without a fresh warning.
const MAX_UNSCANNED_PATH_WARNINGS: usize = 256;

/// Paths already reported as unscanned by [`warn_unscanned_once`].
static UNSCANNED_PATHS_WARNED: LazyLock<Mutex<HashSet<String>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Report — once per path — that an endpoint the classifier does not
/// recognize is being forwarded without request-guardrail scanning. Returns
/// whether this call emitted the warning.
///
/// Platform endpoints (assistants, batches, fine-tuning, threads) and
/// operator-configured `pass_through_endpoints` are not inference formats
/// the classifier understands; rejecting every one of them whenever any
/// guardrail is configured made guardrails and those features mutually
/// exclusive. They are allowed through with a warning instead.
fn warn_unscanned_once(path: &str) -> bool {
    let first = {
        let mut warned = UNSCANNED_PATHS_WARNED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if warned.contains(path) || warned.len() >= MAX_UNSCANNED_PATH_WARNINGS {
            false
        } else {
            warned.insert(path.to_string());
            true
        }
    };
    if first {
        tracing::warn!(
            path = %path,
            "request guardrails do not scan this endpoint; forwarding unscanned"
        );
    }
    first
}

pub(crate) fn request_guardrails_enabled(state: &AppState) -> bool {
    state.pii_guardrail.is_some()
        || state.presidio_guardrail.is_some()
        || state.injection_guardrail.is_some()
        || state.aim_guardrail.is_some()
        || state.lakera_guardrail.is_some()
        || state.bedrock_guardrail.is_some()
        || state.model_armor.is_some()
        || state.custom_guardrail_chain.is_some()
}

fn unsupported() -> ApiError {
    ApiError::bad_request(
        "request guardrails cannot safely inspect this raw inference format; use the translated /v1/chat/completions, /v1/messages, or /v1/responses endpoint",
    )
}

/// Apply the same configured guardrail chain as chat, returning whether the
/// raw JSON body changed. The caller must enforce model ACLs before this call.
pub(crate) async fn apply_raw_request_guardrails(
    state: &AppState,
    method: &Method,
    path: &str,
    body: &mut Value,
) -> Result<bool, ApiError> {
    if !request_guardrails_enabled(state)
        || matches!(
            *method,
            Method::GET | Method::HEAD | Method::DELETE | Method::OPTIONS
        )
    {
        return Ok(false);
    }
    let path = path.trim_matches('/');
    // Classify the same endpoint the upstream will see after URL parsing.
    // In particular, `files/../chat/completions` must not inherit the file
    // CRUD exemption and then normalize to an unscanned inference request.
    if path.split('/').any(|segment| {
        let decoded = percent_encoding::percent_decode_str(segment).decode_utf8_lossy();
        decoded == "." || decoded == ".." || decoded.contains(['/', '\\'])
    }) {
        return Err(unsupported());
    }
    let path = path.strip_prefix("v1/").unwrap_or(path);
    if path == "messages/batches" {
        let requests = body
            .get_mut("requests")
            .and_then(Value::as_array_mut)
            .ok_or_else(unsupported)?;
        let mut changed = false;
        for request in requests {
            let params = request.get_mut("params").ok_or_else(unsupported)?;
            changed |= Box::pin(apply_raw_request_guardrails(
                state, method, "messages", params,
            ))
            .await?;
        }
        return Ok(changed);
    }
    // File CRUD does not execute a prompt. Batch creation and stored-thread
    // runs are deliberately not exempt: their prompts live outside this body.
    if path == "files" || path.starts_with("files/") {
        return Ok(false);
    }

    let mut slots = Vec::new();
    match path {
        "chat/completions" => {
            if !body.is_object() {
                return Err(unsupported());
            }
            validate_messages(body)?;
        }
        "messages" | "messages/count_tokens" => {
            if !body.is_object() {
                return Err(unsupported());
            }
            validate_messages(body)?;
            if let Some(system) = body.get("system") {
                collect_content(system, "/system", &mut slots)?;
            }
            let messages = body
                .get("messages")
                .and_then(Value::as_array)
                .ok_or_else(unsupported)?;
            for (i, message) in messages.iter().enumerate() {
                if let Some(content) = message.get("content") {
                    collect_content(content, &format!("/messages/{i}/content"), &mut slots)?;
                }
            }
            if let Some(tools) = body.get("tools").and_then(Value::as_array) {
                for (i, tool) in tools.iter().enumerate() {
                    for field in ["description", "input_schema"] {
                        if let Some(value) = tool.get(field) {
                            collect_strings(value, &format!("/tools/{i}/{field}"), &mut slots)?;
                        }
                    }
                }
            }
        }
        "completions" | "complete" | "images/generations" => {
            collect_text_input(body.get("prompt"), "/prompt", &mut slots)?;
        }
        "audio/speech" | "moderations" => {
            collect_text_input(body.get("input"), "/input", &mut slots)?;
        }
        "rerank" => {
            collect_text_input(body.get("query"), "/query", &mut slots)?;
            let docs = body
                .get("documents")
                .and_then(Value::as_array)
                .ok_or_else(unsupported)?;
            for (i, doc) in docs.iter().enumerate() {
                if doc.is_string() {
                    push_slot(&format!("/documents/{i}"), false, &mut slots)?;
                } else {
                    collect_text_input(
                        doc.get("text"),
                        &format!("/documents/{i}/text"),
                        &mut slots,
                    )?;
                }
            }
        }
        // Unknown vendor formats (platform endpoints, operator-configured
        // `pass_through_endpoints`, raw Responses, token-id prompts, …) are
        // forwarded unscanned with a one-time warning rather than failing
        // closed: rejecting them here made every non-inference POST 400
        // whenever any guardrail was configured, breaking platform endpoints
        // and general pass-through targets entirely.
        _ => {
            warn_unscanned_once(path);
            return Ok(false);
        }
    }
    guard_projected_text(state, body, &slots).await
}

fn validate_messages(body: &Value) -> Result<(), ApiError> {
    if body
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| !messages.is_empty() && messages.iter().all(Value::is_object))
    {
        Ok(())
    } else {
        Err(ApiError::bad_request(
            "missing or empty required field: 'messages'",
        ))
    }
}

/// A pointer into the original request and whether masking it would invalidate
/// a provider signature (Anthropic thinking blocks).
type TextSlot = (String, bool);

fn push_slot(pointer: &str, signed: bool, slots: &mut Vec<TextSlot>) -> Result<(), ApiError> {
    if slots.len() >= MAX_PROJECTED_TEXT_FIELDS {
        return Err(ApiError::bad_request(
            "too many text fields for request guardrails",
        ));
    }
    slots.push((pointer.to_string(), signed));
    Ok(())
}

fn collect_text_input(
    value: Option<&Value>,
    pointer: &str,
    slots: &mut Vec<TextSlot>,
) -> Result<(), ApiError> {
    match value {
        Some(Value::String(_)) => push_slot(pointer, false, slots),
        Some(Value::Array(items)) => {
            for (i, item) in items.iter().enumerate() {
                if !item.is_string() {
                    return Err(unsupported());
                }
                push_slot(&format!("{pointer}/{i}"), false, slots)?;
            }
            Ok(())
        }
        _ => Err(unsupported()),
    }
}

fn collect_strings(
    value: &Value,
    pointer: &str,
    slots: &mut Vec<TextSlot>,
) -> Result<(), ApiError> {
    match value {
        Value::String(_) => push_slot(pointer, false, slots)?,
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                collect_strings(item, &format!("{pointer}/{i}"), slots)?;
            }
        }
        Value::Object(fields) => {
            for (key, value) in fields {
                let key = key.replace('~', "~0").replace('/', "~1");
                collect_strings(value, &format!("{pointer}/{key}"), slots)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn collect_content(
    value: &Value,
    pointer: &str,
    slots: &mut Vec<TextSlot>,
) -> Result<(), ApiError> {
    match value {
        Value::String(_) => push_slot(pointer, false, slots),
        Value::Array(blocks) => {
            for (i, block) in blocks.iter().enumerate() {
                let p = format!("{pointer}/{i}");
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        collect_text_input(block.get("text"), &format!("{p}/text"), slots)?
                    }
                    Some("tool_use") => {
                        if let Some(input) = block.get("input") {
                            collect_strings(input, &format!("{p}/input"), slots)?;
                        }
                    }
                    Some("tool_result") => {
                        if let Some(content) = block.get("content") {
                            collect_content(content, &format!("{p}/content"), slots)?;
                        }
                    }
                    Some("thinking") => {
                        if block.get("thinking").is_some_and(Value::is_string) {
                            push_slot(&format!("{p}/thinking"), true, slots)?;
                        }
                    }
                    Some("image") | Some("redacted_thinking") => {}
                    Some("document") => {
                        if block.pointer("/source/type").and_then(Value::as_str) == Some("text") {
                            collect_text_input(
                                block.pointer("/source/data"),
                                &format!("{p}/source/data"),
                                slots,
                            )?;
                        } else if block.pointer("/source/type").and_then(Value::as_str)
                            == Some("content")
                        {
                            let content =
                                block.pointer("/source/content").ok_or_else(unsupported)?;
                            collect_content(content, &format!("{p}/source/content"), slots)?;
                        }
                        for field in ["title", "context"] {
                            if let Some(value) = block.get(field) {
                                collect_strings(value, &format!("{p}/{field}"), slots)?;
                            }
                        }
                    }
                    _ => return Err(unsupported()),
                }
            }
            Ok(())
        }
        _ => Err(unsupported()),
    }
}

async fn guard_projected_text(
    state: &AppState,
    body: &mut Value,
    slots: &[TextSlot],
) -> Result<bool, ApiError> {
    // Keep all collected slots for the 4096-field admission cap, but don't
    // synthesize a second copy of text the direct chat walkers already cover.
    // In particular, catalog copyback must not overwrite a direct mask with
    // text independently rewritten from the original catalog value.
    let projected_slots: Vec<_> = slots
        .iter()
        .filter(|(pointer, _)| slot_needs_projection(body, pointer))
        .collect();
    let mut guarded = body.clone();
    let had_messages = guarded.get("messages").is_some();
    if !had_messages {
        guarded["messages"] = json!([]);
    }
    let messages = guarded["messages"].as_array_mut().ok_or_else(unsupported)?;
    let original_len = messages.len();
    for (pointer, _) in &projected_slots {
        let text = body
            .pointer(pointer)
            .and_then(Value::as_str)
            .ok_or_else(unsupported)?;
        let role = if pointer.starts_with("/system") {
            "system"
        } else if pointer.starts_with("/tools/") {
            "tool"
        } else if let Some(index) = pointer
            .strip_prefix("/messages/")
            .and_then(|rest| rest.split('/').next())
            .and_then(|index| index.parse::<usize>().ok())
        {
            body["messages"][index]
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
        } else {
            "user"
        };
        messages.push(json!({"role": role, "content": text}));
    }
    crate::handlers::chat::apply_request_guardrails(state, &mut guarded).await?;
    let messages = guarded["messages"].as_array_mut().ok_or_else(unsupported)?;
    if messages.len() != original_len + projected_slots.len() {
        return Err(unsupported());
    }
    let projected = messages.split_off(original_len);
    if !had_messages {
        guarded
            .as_object_mut()
            .ok_or_else(unsupported)?
            .remove("messages");
    }
    for ((pointer, signed), message) in projected_slots.into_iter().zip(projected) {
        let text = message
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(unsupported)?;
        if body.pointer(pointer).and_then(Value::as_str) != Some(text) {
            if *signed {
                return Err(ApiError::bad_request(
                    "request guardrails would invalidate a signed thinking block",
                ));
            }
            *guarded.pointer_mut(pointer).ok_or_else(unsupported)? =
                Value::String(text.to_string());
        }
    }
    let changed = guarded != *body;
    *body = guarded;
    Ok(changed)
}

/// Only skip slots whose exact surface is covered by the shared chat walkers.
/// Nested native tool results, tool inputs, documents, and thinking blocks
/// still need projection; malformed/hybrid catalogs keep their old coverage.
fn slot_needs_projection(body: &Value, pointer: &str) -> bool {
    if let Some((index, field)) = pointer
        .strip_prefix("/tools/")
        .and_then(|rest| rest.split_once('/'))
    {
        let tool = index.parse::<usize>().ok().and_then(|index| {
            body.get("tools")
                .and_then(Value::as_array)
                .and_then(|tools| tools.get(index))
        });
        if let Some(tool) = tool
            && tool.get("function").is_none()
            && (field == "description"
                || (tool.get("parameters").is_none()
                    && (field == "input_schema" || field.starts_with("input_schema/"))))
        {
            return false;
        }
    }
    if let Some((_, field)) = pointer
        .strip_prefix("/messages/")
        .and_then(|rest| rest.split_once('/'))
    {
        if field == "content"
            || field
                .strip_prefix("content/")
                .and_then(|rest| rest.split_once('/'))
                .is_some_and(|(_, field)| field == "text")
        {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pii_state() -> AppState {
        let config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                guardrails: Some(
                    serde_json::from_value(json!({"pii_masking":{"enabled":true}})).unwrap(),
                ),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).unwrap()
    }

    #[tokio::test]
    async fn native_catalog_keeps_the_projected_text_field_admission_cap() {
        let state = pii_state();
        for count in [MAX_PROJECTED_TEXT_FIELDS - 1, MAX_PROJECTED_TEXT_FIELDS] {
            let mut body = json!({
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"name": "lookup", "input_schema": {
                    "examples": vec!["safe"; count], "minimum": 7, "nullable": true
                }}]
            });
            let original = body.clone();
            let result =
                apply_raw_request_guardrails(&state, &Method::POST, "messages", &mut body).await;
            if count == MAX_PROJECTED_TEXT_FIELDS - 1 {
                assert!(!result.expect("exactly 4096 text fields must still be admitted"));
            } else {
                let error = result.expect_err("catalog leaves still count toward the cap");
                assert!(error.message.contains("too many text fields"));
            }
            assert_eq!(body, original);
        }
    }

    #[tokio::test]
    async fn raw_projection_keeps_nested_native_and_hybrid_catalog_text_coverage() {
        let state = pii_state();
        let mut body = json!({
            "system": [{"type": "text", "text": "a@example.com"}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_result", "tool_use_id": "unchanged", "content": [
                    {"type": "text", "text": "a@example.com"}
                ]},
                {"type": "tool_use", "name": "unchanged", "input": {
                    "email/~key": "a@example.com", "number": 4111111111111111_u64
                }}
            ]}],
            "tools": [
                {"function": null, "description": {"nested": "a@example.com"},
                    "input_schema": {"examples": ["a@example.com"]}},
                {"name": "unchanged", "description": "a@example.com",
                    "parameters": {"description": "a@example.com"},
                    "input_schema": {"examples": ["a@example.com"]}}
            ]
        });
        let mut expected = body.clone();
        expected["system"][0]["text"] = json!("[EMAIL]");
        expected["messages"][0]["content"][1]["content"][0]["text"] = json!("[EMAIL]");
        expected["messages"][0]["content"][2]["input"]["email/~key"] = json!("[EMAIL]");
        expected["tools"][0]["description"]["nested"] = json!("[EMAIL]");
        expected["tools"][0]["input_schema"]["examples"][0] = json!("[EMAIL]");
        expected["tools"][1]["description"] = json!("[EMAIL]");
        expected["tools"][1]["parameters"]["description"] = json!("[EMAIL]");
        expected["tools"][1]["input_schema"]["examples"][0] = json!("[EMAIL]");
        assert!(
            apply_raw_request_guardrails(&state, &Method::POST, "messages", &mut body)
                .await
                .unwrap()
        );
        assert_eq!(
            body, expected,
            "only fields already covered directly can skip projection"
        );
    }

    #[tokio::test]
    async fn guarded_raw_prompts_reject_uninspectable_tokens_and_path_aliases() {
        let state = pii_state();
        let mut token_prompt = json!({"model":"m","prompt":[1,2,3]});
        assert!(
            apply_raw_request_guardrails(&state, &Method::POST, "completions", &mut token_prompt)
                .await
                .is_err()
        );
        let mut body = json!({"model":"m","messages":[{"role":"user","content":"a@example.com"}]});
        for path in [
            "v1/files/../chat/completions",
            "v1/files/%2e%2e/chat/completions",
        ] {
            assert!(
                apply_raw_request_guardrails(&state, &Method::POST, path, &mut body)
                    .await
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn masking_does_not_corrupt_signed_thinking() {
        let state = pii_state();
        let mut body = json!({"model":"m","messages":[{"role":"assistant","content":[
            {"type":"thinking","thinking":"a@example.com","signature":"provider-signature"}
        ]}]});
        let original = body.clone();
        let error = apply_raw_request_guardrails(&state, &Method::POST, "messages", &mut body)
            .await
            .unwrap_err();
        assert!(error.message.contains("signed thinking"));
        assert_eq!(
            body, original,
            "failed projection must not partially mutate the raw body"
        );
    }

    /// A WARN-level event counter (mirrors the security.rs test subscriber) so
    /// tests can assert the unscanned-endpoint notice fires exactly once.
    struct WarnCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl tracing::Subscriber for WarnCounter {
        fn enabled(&self, _m: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _span: &tracing::span::Id, _event: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            // tracing's Level ordering is reversed from severity (DEBUG > WARN
            // > ERROR), so "WARN or more severe" is `level <= WARN`.
            if *event.metadata().level() <= tracing::Level::WARN {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        fn enter(&self, _span: &tracing::span::Id) {}
        fn exit(&self, _span: &tracing::span::Id) {}
        fn clone_span(&self, _span: &tracing::span::Id) -> tracing::span::Id {
            tracing::span::Id::from_u64(2)
        }
        fn try_close(&self, _id: tracing::span::Id) -> bool {
            true
        }
    }

    /// Platform endpoints (assistants/batches/fine-tuning/threads) and
    /// operator `pass_through_endpoints` are not formats the classifier
    /// recognizes. With a guardrail configured they must be allowed through
    /// unscanned with exactly one WARN per path — the previous fail-closed 400
    /// made guardrails and those features mutually exclusive.
    #[test]
    fn unscanned_platform_post_is_allowed_with_a_single_warning() {
        let warns = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = pii_state();
        let subscriber = WarnCounter(warns.clone());
        // with_default scopes the subscriber to this thread only, so parallel
        // test threads are unaffected.
        tracing::subscriber::with_default(subscriber, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let mut body = json!({"assistant_id": "asst_1"});
                let path = "assistants/unscanned-warn-once-test";
                assert!(
                    !apply_raw_request_guardrails(&state, &Method::POST, path, &mut body)
                        .await
                        .expect("unrecognized platform endpoint must be allowed"),
                    "unscanned endpoint must not report a body change"
                );
                assert_eq!(
                    warns.load(std::sync::atomic::Ordering::Relaxed),
                    1,
                    "the first unscanned request must warn"
                );
                // A repeated request to the same path must not spam the log.
                apply_raw_request_guardrails(&state, &Method::POST, path, &mut body)
                    .await
                    .expect("unrecognized platform endpoint must stay allowed");
                assert_eq!(
                    warns.load(std::sync::atomic::Ordering::Relaxed),
                    1,
                    "repeat requests to the same unscanned path must not warn again"
                );
            });
        });
    }

    /// Only endpoints the classifier recognizes get the non-object-body
    /// rejection; a bad shape on `messages` must still fail closed with 400.
    #[tokio::test]
    async fn recognized_inference_paths_still_reject_non_object_bodies() {
        let state = pii_state();
        let mut array_body = json!([1, 2, 3]);
        let error =
            apply_raw_request_guardrails(&state, &Method::POST, "messages", &mut array_body)
                .await
                .expect_err("non-object body on a recognized path must be rejected");
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(error.message.contains("cannot safely inspect"));
    }

    /// File CRUD does not execute a prompt, so it stays exempt for every body
    /// shape (including non-object bodies, which are never inspected).
    #[tokio::test]
    async fn files_crud_is_exempt_from_request_scanning() {
        let state = pii_state();
        let mut body = json!({"purpose": "fine-tune"});
        assert!(
            !apply_raw_request_guardrails(&state, &Method::POST, "v1/files", &mut body)
                .await
                .expect("files CRUD must be allowed")
        );
        assert!(
            !apply_raw_request_guardrails(
                &state,
                &Method::POST,
                "v1/files/file_1/content",
                &mut body
            )
            .await
            .expect("files sub-routes must be allowed")
        );
        let mut non_object = json!([1, 2, 3]);
        assert!(
            !apply_raw_request_guardrails(&state, &Method::POST, "files", &mut non_object)
                .await
                .expect("files exemption must win over the non-object check")
        );
    }
}
