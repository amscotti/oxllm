//! Identity realtime provider for OpenAI-compatible realtime APIs.
//!
//! oxllm exposes the OpenAI Realtime API wire format to clients, so any
//! upstream that already speaks that format (OpenAI, xAI, any
//! OpenAI-compatible realtime endpoint) needs **no translation** — events
//! pass through unchanged in both directions. This is the reference
//! [`RealtimeProvider`] implementation.
//!
//! The only provider-specific logic is connection setup: building the
//! upstream WebSocket URL ([`OpenAIRealtimeProvider::ws_url`]) and attaching
//! bearer auth ([`OpenAIRealtimeProvider::ws_auth_headers`]).

use oxllm_core::RealtimeProvider;
use oxllm_core::deployment::Deployment;
use oxllm_core::error::Result;
use oxllm_core::hook::UsageSummary;

/// Identity realtime provider — passes events through unchanged.
///
/// Used for OpenAI, xAI, and any OpenAI-compatible realtime API. Selected
/// by the realtime proxy for any deployment whose provider slug is not
/// `bedrock` / `bedrock_mantle` (those require a translating provider,
/// added in a later batch).
pub struct OpenAIRealtimeProvider;

impl OpenAIRealtimeProvider {
    /// Construct the identity realtime provider.
    pub fn new() -> Self {
        Self
    }
}

impl Default for OpenAIRealtimeProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Default upstream base for OpenAI realtime when a deployment sets no
/// `api_base`. Already in `wss://` form so no scheme rewriting is needed.
const DEFAULT_REALTIME_BASE: &str = "wss://api.openai.com/v1";

/// Extract the bare upstream model id from a deployment's
/// `oxllm_params.model`, stripping any `provider/` prefix.
///
/// `oxllm_params.model` is typically `"openai/gpt-4o-realtime"`; the
/// upstream realtime URL wants the bare id `gpt-4o-realtime`. When there is
/// no `/` (e.g. a bare `"gpt-4o-realtime"`), the whole string is returned.
pub(crate) fn bare_model(deployment: &Deployment) -> &str {
    // An ARN's resource slash is not a provider/model separator.
    if deployment.oxllm_params.model.starts_with("arn:") {
        return &deployment.oxllm_params.model;
    }
    match deployment.oxllm_params.model.split_once('/') {
        Some((_, rest)) => rest,
        None => deployment.oxllm_params.model.as_str(),
    }
}

impl RealtimeProvider for OpenAIRealtimeProvider {
    fn transform_client_event(&self, event: &str) -> Result<Vec<String>> {
        Ok(vec![event.to_string()])
    }

    fn transform_upstream_event(&self, event: &str) -> Result<Vec<String>> {
        Ok(vec![event.to_string()])
    }

    /// Extract token usage from an OpenAI Realtime `response.done` event.
    ///
    /// OpenAI Realtime reports usage (input/output/total tokens, plus audio and
    /// cache details) nested under `response.usage`. This enables the host to
    /// meter/cost a realtime session at the moment the upstream reports usage.
    /// Returns `None` for events without a usage object (the common case).
    fn extract_usage(&self, upstream_event: &str) -> Option<UsageSummary> {
        let v: serde_json::Value = serde_json::from_str(upstream_event).ok()?;
        if v.get("type").and_then(|t| t.as_str()) != Some("response.done") {
            return None;
        }
        // `response.done` nests usage under `response.usage`; some compatible
        // upstreams place it at the top level — accept both.
        let usage = v
            .get("response")
            .and_then(|r| r.get("usage"))
            .or_else(|| v.get("usage"))?;
        // OpenAI Realtime commonly reports `"usage": null` on
        // `response.created` (and any event where usage is not yet known).
        // Treat non-object usage (null, a bare number, …) as "no usage
        // reported" rather than returning a zero summary — a zero summary
        // gets folded into the accumulator and obscures the absence signal.
        // An **empty** object (xAI's `response.created` quirk: `usage: {}`)
        // is likewise absence, not a zero report.
        let usage = usage.as_object()?;
        if !["input_tokens", "output_tokens"]
            .iter()
            .any(|field| usage.get(*field).and_then(|t| t.as_u64()).is_some())
        {
            return None;
        }
        let input = usage
            .get("input_tokens")
            .and_then(|t| t.as_u64())
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(|t| t.as_u64())
            .unwrap_or(0);
        let total = usage
            .get("total_tokens")
            .and_then(|t| t.as_u64())
            .unwrap_or(input.saturating_add(output));
        // Realtime uses singular *_token_details; retain the old plural
        // spelling for compatible services that already emit it.
        let input_details = usage
            .get("input_token_details")
            .or_else(|| usage.get("input_tokens_details"));
        let output_details = usage
            .get("output_token_details")
            .or_else(|| usage.get("output_tokens_details"));
        Some(UsageSummary {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: total,
            cache_read_input_tokens: input_details
                .and_then(|d| d.get("cached_tokens").and_then(|t| t.as_u64())),
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: None,
            audio_input_tokens: input_details
                .and_then(|d| d.get("audio_tokens").and_then(|t| t.as_u64())),
            reasoning_tokens: output_details
                .and_then(|d| d.get("reasoning_tokens").and_then(|t| t.as_u64())),
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
        })
    }

    fn ws_url(&self, deployment: &Deployment) -> String {
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .map(|b| {
                let trimmed = b.trim_end_matches('/');
                super::scheme_rewrite_for_ws(trimmed)
            })
            .unwrap_or_else(|| DEFAULT_REALTIME_BASE.to_string());

        // Append the `model` query param via `url::Url` so an existing query
        // string on `api_base` (e.g. `?api-version=…`) is preserved *and* the
        // `/realtime` path lands in the URL path rather than being swallowed
        // into the query. A naive `format!("{base}/realtime")` produces a
        // malformed URL when `base` carries a `?` — see
        // `build_realtime_ws_url`.
        let model = bare_model(deployment);
        super::build_realtime_ws_url(&base, "/realtime", model)
    }

    fn ws_auth_headers(&self, deployment: &Deployment) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if let Some(key) = &deployment.oxllm_params.api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;
    use serde_json::Map;

    fn dep(api_base: Option<&str>) -> Deployment {
        Deployment {
            model_name: "gpt-4o-realtime".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o-realtime".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: api_base.map(String::from),
                extra: Map::new(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    // ---- identity transform ------------------------------------------------

    #[test]
    fn client_event_passes_through_unchanged() {
        let p = OpenAIRealtimeProvider::new();
        let event = r#"{"type":"session.update","session":{"voice":"alloy"}}"#;
        assert_eq!(
            p.transform_client_event(event)
                .expect("identity never errors"),
            vec![event.to_string()]
        );
    }

    #[test]
    fn upstream_event_passes_through_unchanged() {
        let p = OpenAIRealtimeProvider::new();
        let event = r#"{"type":"response.audio.delta","delta":"base64"}"#;
        assert_eq!(
            p.transform_upstream_event(event)
                .expect("identity never errors"),
            vec![event.to_string()]
        );
    }

    // ---- extract_usage (realtime metering) --------------------------------

    #[test]
    fn extract_usage_returns_none_for_events_without_usage() {
        let p = OpenAIRealtimeProvider::new();
        assert!(
            p.extract_usage(r#"{"type":"response.audio.delta","delta":"x"}"#)
                .is_none()
        );
        assert!(p.extract_usage("not json").is_none());
    }

    #[test]
    fn extract_usage_parses_response_done_nested_usage() {
        let p = OpenAIRealtimeProvider::new();
        let event = r#"{"type":"response.done","response":{"usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12,"input_tokens_details":{"cached_tokens":2,"audio_tokens":3},"output_tokens_details":{"reasoning_tokens":1}}}}"#;
        let u = p.extract_usage(event).expect("usage present");
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.completion_tokens, 7);
        assert_eq!(u.total_tokens, 12);
        assert_eq!(u.cache_read_input_tokens, Some(2));
        assert_eq!(u.audio_input_tokens, Some(3));
        assert_eq!(u.reasoning_tokens, Some(1));
    }

    #[test]
    fn extract_usage_accepts_top_level_usage_and_derives_total() {
        let p = OpenAIRealtimeProvider::new();
        let event = r#"{"type":"response.done","usage":{"input_tokens":10,"output_tokens":20}}"#;
        let u = p.extract_usage(event).expect("usage present");
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 30, "missing total must be derived");
    }

    #[test]
    fn native_realtime_usage_details_and_cancelled_responses_are_metered() {
        let event = r#"{"type":"response.done","response":{"status":"cancelled","usage":{
            "input_tokens":132,"output_tokens":121,"total_tokens":253,
            "input_token_details":{"text_tokens":119,"audio_tokens":13,"cached_tokens":64},
            "output_token_details":{"text_tokens":30,"audio_tokens":91}
        }}}"#;
        let usage = OpenAIRealtimeProvider.extract_usage(event).unwrap();
        assert_eq!(usage.audio_input_tokens, Some(13));
        assert_eq!(usage.cache_read_input_tokens, Some(64));
        assert_eq!(usage.total_tokens, 253);
        assert!(OpenAIRealtimeProvider.extract_usage(
            r#"{"type":"response.created","response":{"usage":{"input_tokens":0,"output_tokens":0}}}"#
        ).is_none());
        assert!(
            OpenAIRealtimeProvider
                .extract_usage(
                    r#"{"type":"response.done","response":{"usage":{"unexpected":true}}}"#
                )
                .is_none()
        );
    }

    /// LOW: OpenAI Realtime commonly reports `"usage": null` (e.g. on
    /// `response.created`). This must surface as `None`, not a zero
    /// `UsageSummary` — a zero summary folds into the accumulator and
    /// obscures the "no usage reported" signal.
    #[test]
    fn extract_usage_returns_none_for_null_usage() {
        let p = OpenAIRealtimeProvider::new();
        assert!(
            p.extract_usage(r#"{"type":"response.created","response":{"usage":null}}"#)
                .is_none(),
            "null usage must surface as None"
        );
        // Bare non-object usage (defensive) also surfaces as None.
        assert!(
            p.extract_usage(r#"{"type":"response.done","usage":5}"#)
                .is_none(),
            "non-object usage must surface as None"
        );
    }

    /// LOW (C10): xAI's documented quirk is an **empty** `usage: {}` object
    /// on `response.created`. An empty object carries no report, so it must
    /// surface as `None` like `null` — otherwise the session settles as
    /// "usage reported" with a zero summary if it ends before
    /// `response.done`.
    #[test]
    fn extract_usage_returns_none_for_empty_usage_object() {
        let p = OpenAIRealtimeProvider::new();
        assert!(
            p.extract_usage(r#"{"type":"response.created","response":{"usage":{}}}"#)
                .is_none(),
            "empty nested usage object must surface as None"
        );
        assert!(
            p.extract_usage(r#"{"type":"response.done","usage":{}}"#)
                .is_none(),
            "empty top-level usage object must surface as None"
        );
    }

    // ---- ws_url ------------------------------------------------------------

    #[test]
    fn url_default_openai() {
        let url = OpenAIRealtimeProvider.ws_url(&dep(None));
        assert_eq!(
            url,
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime"
        );
    }

    #[test]
    fn url_uses_api_base_https_to_wss() {
        let url = OpenAIRealtimeProvider.ws_url(&dep(Some("https://my.host/v1")));
        assert_eq!(url, "wss://my.host/v1/realtime?model=gpt-4o-realtime");
    }

    /// MEDIUM #5: a plaintext `http://localhost` api_base is allowed (local
    /// debug) and rewritten to `ws://`.
    #[test]
    fn url_http_localhost_allowed_and_rewritten_to_ws() {
        let url = OpenAIRealtimeProvider.ws_url(&dep(Some("http://localhost:8080/v1")));
        assert!(
            url.starts_with("ws://localhost:8080/"),
            "localhost http must be rewritten to ws, got {url}"
        );
    }

    /// MEDIUM #5: a plaintext `http://` api_base for a non-local host is
    /// still rewritten (to avoid blocking the request) but the URL itself
    /// carries the cleartext scheme so the misconfiguration is visible.
    #[test]
    fn url_http_non_localhost_rewritten_to_ws() {
        let url = OpenAIRealtimeProvider.ws_url(&dep(Some("http://api.openai.com/v1")));
        assert!(
            url.starts_with("ws://api.openai.com/"),
            "non-local http must be rewritten to ws (with a warning), got {url}"
        );
    }

    /// When the api_base already carries a query string (e.g. an api-version
    /// param), naive `?model=` concat would produce `?api-version=…?model=…`.
    /// Going through `url::Url` keeps the existing query and appends the new
    /// pair correctly.
    #[test]
    fn url_preserves_existing_query_params_when_appending_model() {
        let url =
            OpenAIRealtimeProvider.ws_url(&dep(Some("https://my.host/v1?api-version=2024-02-15")));
        assert!(
            url.contains("api-version=2024-02-15"),
            "existing query params must be preserved: {url}"
        );
        assert!(
            url.contains("model=gpt-4o-realtime"),
            "model must be appended: {url}"
        );
        // Exactly one `?` — the new pair is appended via `&`, not a second `?`.
        assert_eq!(
            url.matches('?').count(),
            1,
            "url must contain exactly one '?': {url}"
        );
    }

    #[test]
    fn url_replaces_a_template_selector_and_encodes_the_concrete_model_as_one_value() {
        let mut deployment = dep(Some(
            "https://my.host/v1?model=openai%2F*&api-version=2024-02-15",
        ));
        deployment.oxllm_params.model = "openai/模型&trace=true".into();
        let url = url::Url::parse(&OpenAIRealtimeProvider.ws_url(&deployment)).unwrap();
        let query: Vec<_> = url.query_pairs().collect();
        let models: Vec<_> = query.iter().filter(|(name, _)| name == "model").collect();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].1, "模型&trace=true");
        assert!(
            query
                .iter()
                .any(|(name, value)| name == "api-version" && value == "2024-02-15")
        );
        assert!(!query.iter().any(|(name, _)| name == "trace"));
    }

    /// HIGH: when `api_base` carries a query string, the `/realtime` path
    /// suffix must land in the URL **path**, not be swallowed into the query
    /// string. Covers the four shapes (no query, query-only, query+path,
    /// trailing slash + query). The previous substring-only assertions missed
    /// this because the malformed URL still *contains* `api-version=…` and
    /// `model=…` as substrings.
    #[test]
    fn url_path_is_correct_when_api_base_carries_query() {
        let cases: &[(&str, &str, bool)] = &[
            // (api_base, expected_path, has_existing_query)
            ("https://my.host/v1", "/v1/realtime", false),
            // Query-only (no path) — common Azure-style shape.
            (
                "https://my.host/v1?api-version=2024-02-15",
                "/v1/realtime",
                true,
            ),
            // Trailing slash + query (trimmed first).
            (
                "https://my.host/v1/?api-version=2024-02-15",
                "/v1/realtime",
                true,
            ),
            // Host root + query.
            ("https://my.host?api-version=2024-02-15", "/realtime", true),
        ];
        for (api_base, expected_path, has_query) in cases {
            let url = OpenAIRealtimeProvider.ws_url(&dep(Some(api_base)));
            let parsed = url::Url::parse(&url)
                .unwrap_or_else(|e| panic!("url parse failed for {api_base}: {e} ({url})"));
            assert_eq!(
                parsed.path(),
                *expected_path,
                "api_base={api_base} produced url={url} with path {} (expected {expected_path})",
                parsed.path()
            );
            // `model` is always appended; `api-version` preserved when present.
            assert!(
                parsed.query_pairs().any(|(k, _)| k == "model"),
                "model param missing for api_base={api_base}: {url}"
            );
            if *has_query {
                assert!(
                    parsed.query_pairs().any(|(k, _)| k == "api-version"),
                    "existing api-version query lost for api_base={api_base}: {url}"
                );
            }
        }
    }

    // ---- ws_auth_headers ---------------------------------------------------

    #[test]
    fn headers_include_auth_and_beta() {
        let h = OpenAIRealtimeProvider.ws_auth_headers(&dep(None));
        // The `OpenAI-Beta: realtime=v1` header was dropped; only the bearer
        // auth header is attached to the upstream WebSocket upgrade.
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].0, "Authorization");
        assert!(h[0].1.contains("sk-test"));
    }

    #[test]
    fn headers_empty_when_no_api_key() {
        let mut d = dep(None);
        d.oxllm_params.api_key = None;
        assert!(OpenAIRealtimeProvider.ws_auth_headers(&d).is_empty());
    }
}
