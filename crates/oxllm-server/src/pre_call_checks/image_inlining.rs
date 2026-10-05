//! Inline remote image URLs into `data:` URLs before provider transforms.
//!
//! Bedrock's Converse and InvokeModel APIs cannot fetch remote image URLs —
//! they require base64-encoded image bytes inline in the request body. When
//! a Bedrock deployment is the target, the orchestration layer invokes
//! [`inline_image_urls`] to walk the request body and download every
//! `image_url` content block whose URL starts with `http://` or `https://`,
//! then rewrite the block to a `data:{media_type};base64,{payload}` URL.
//! After this pass the body is safe to feed to any provider transform.
//!
//! Other providers (OpenAI, Gemini, Anthropic-native, Vertex) fetch HTTP
//! image URLs themselves, so the orchestration layer gates this work on
//! Bedrock (see [`needs_inlining`]).
//!
//! The download is server-layer work (async HTTP, size cap, SSRF guard) and
//! must NOT live inside `oxllm-providers`, which is pure-core (no I/O).
//!
//! # Failure handling
//!
//! Hard client errors (size cap exceeded, SSRF-blocked host, non-image
//! Content-Type, malformed URL) surface as
//! [`oxllm_core::Error::BadRequest`] (HTTP 400). Image-host network
//! failures (connection refused, DNS error, non-2xx status, chunked-read
//! failure) surface as [`oxllm_core::Error::Network`] (HTTP 502): an
//! unreachable image host is the upstream's problem, not the caller's
//! request shape. We never silently drop an image block: silently dropping
//! causes data loss the caller cannot detect (the model would simply not
//! see the image, with no signal that anything went wrong).

use base64::Engine;
use serde_json::Value;

use crate::state::AppState;

/// Maximum image size we will download and inline (10 MB). Larger payloads
/// would bloat the request body well past Bedrock's own per-request limits
/// and risk OOMing the proxy; a deliberate 4xx is the right signal. This
/// matches the [`crate::security::DEFAULT_BODY_LIMIT_BYTES`] request body
/// cap.
const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;

/// Maximum number of remote `image_url` blocks the inliner will download for
/// a single request. Each block is fetched sequentially and may pull up to
/// [`MAX_IMAGE_BYTES`], so without a per-request bound a tiny request body
/// (≈80 bytes per block) could force the proxy to fetch, base64-encode, and
/// buffer ~1 GB before the upstream POST even starts. The cap is a small
/// constant generous enough for legitimate multi-image prompts while closing
/// the amplification.
const MAX_IMAGE_BLOCKS: usize = 8;

/// Walk `body["messages"]` looking for content blocks of the form
/// `{"type":"image_url","image_url":{"url":"https://..."}}` and rewrite each
/// remote HTTP(S) URL into a `data:` URL by downloading and base64-encoding
/// the payload.
///
/// - Non-HTTP(S) URLs (`data:`, empty, malformed) are left untouched — the
///   downstream provider transform already handles `data:` URLs.
/// - The body is mutated in place. The function is a no-op when `messages`
///   is missing or no eligible `image_url` block is present.
/// - A request carrying more than [`MAX_IMAGE_BLOCKS`] eligible blocks is
///   rejected up-front with [`oxllm_core::Error::BadRequest`] (DoS
///   amplification guard — see the constant's docs).
/// - Hard client errors (size cap exceeded, SSRF-blocked host, non-image
///   content-type, too many blocks) surface as [`oxllm_core::Error::BadRequest`]
///   and abort the whole pass on the first offending block; image-host
///   network failures surface as [`oxllm_core::Error::Network`] (HTTP 502).
///
/// The pass is **transactional**: every `data:` URL is computed into a side
/// table first and only applied to `body` once the full scan succeeds. A
/// hard error on block *N* leaves blocks *0..N* untouched, so the body is
/// never observed in a half-rewritten state on `Err`.
pub(crate) async fn inline_image_urls(
    body: &mut Value,
    http_client: &reqwest::Client,
) -> Result<(), oxllm_core::Error> {
    let count = count_http_image_urls(body);
    if count > MAX_IMAGE_BLOCKS {
        return Err(oxllm_core::Error::BadRequest(format!(
            "request contains {count} remote image_url blocks; limit is {MAX_IMAGE_BLOCKS}"
        )));
    }
    if count == 0 {
        return Ok(());
    }
    let mut rewrites: Vec<(usize, usize, String)> = Vec::with_capacity(count);
    {
        let Some(messages) = body.get("messages").and_then(Value::as_array) else {
            return Ok(());
        };
        for (mi, msg) in messages.iter().enumerate() {
            let Some(parts) = msg.get("content").and_then(Value::as_array) else {
                continue;
            };
            for (pi, part) in parts.iter().enumerate() {
                if part.get("type").and_then(Value::as_str) != Some("image_url") {
                    continue;
                }
                let Some(url) = part
                    .get("image_url")
                    .and_then(|iu| iu.get("url"))
                    .and_then(Value::as_str)
                else {
                    continue;
                };
                let lower = url.to_ascii_lowercase();
                if !(lower.starts_with("http://") || lower.starts_with("https://")) {
                    continue;
                }
                ssrf_guard(url).await?;
                let data_url = download_as_data_url(http_client, url).await?;
                rewrites.push((mi, pi, data_url));
            }
        }
    }
    let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return Ok(());
    };
    for (mi, pi, data_url) in rewrites {
        if let Some(iu) = messages
            .get_mut(mi)
            .and_then(|m| m.get_mut("content"))
            .and_then(|c| c.as_array_mut())
            .and_then(|parts| parts.get_mut(pi))
            .and_then(|p| p.get_mut("image_url"))
            .and_then(|v| v.as_object_mut())
        {
            iu.insert("url".to_string(), Value::String(data_url));
        }
    }
    Ok(())
}

/// Should the orchestration layer inline HTTP image URLs for this request?
///
/// Two conditions must hold:
///
/// 1. At least one reachable deployment uses the `bedrock` provider (other
///    providers fetch HTTP URLs natively and would needlessly pay the
///    proxy-side download).
/// 2. `body` contains at least one `image_url` block whose URL starts with
///    `http://` or `https://`.
///
/// Cheap scan used by the orchestration layer to skip the clone-and-mutate
/// pass on the common no-image and non-Bedrock paths.
pub(crate) fn needs_inlining(state: &AppState, model: &str, body: &Value) -> bool {
    if !has_http_image_url(body) {
        return false;
    }
    let group_has_bedrock = |group: &str| {
        state
            .router
            .deployments_for_model(&state.deployments, group)
            .iter()
            .any(|d| d.provider_slug() == "bedrock")
    };
    if group_has_bedrock(model) {
        return true;
    }
    // This pass precedes deployment selection and retries. Include every
    // complexity pool and the default: classifications can be inherited from
    // a fallback leg, and cooldown changes can widen the pool during retries.
    // Use the router's alias/wildcard resolution rather than duplicating it.
    if state.router.strategy_kind_for(model) == oxllm_core::RoutingStrategyKind::ComplexityRouter {
        if let Some(config) = &state.router.config().complexity_router_config {
            return config
                .tiers
                .values()
                .flatten()
                .any(|group| group_has_bedrock(group))
                || config
                    .default_model
                    .as_deref()
                    .is_some_and(group_has_bedrock);
        }
    }
    false
}

/// Does `body` contain at least one `image_url` block carrying an HTTP(S) URL?
fn has_http_image_url(body: &Value) -> bool {
    count_http_image_urls(body) > 0
}

/// Count the `image_url` blocks in `body` whose URL starts with `http://` or
/// `https://`. Used both as the [`needs_inlining`] gate (> 0) and as the
/// per-request amplification cap check ([`MAX_IMAGE_BLOCKS`]). Non-array
/// `messages` / `content`, non-`image_url` parts, and non-HTTP(S) URLs
/// (`data:`, empty, malformed) are skipped.
fn count_http_image_urls(body: &Value) -> usize {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return 0;
    };
    let mut count = 0;
    for msg in messages {
        let Some(parts) = msg.get("content").and_then(Value::as_array) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }
            if let Some(url) = part
                .get("image_url")
                .and_then(|iu| iu.get("url"))
                .and_then(Value::as_str)
            {
                let lower = url.to_ascii_lowercase();
                if lower.starts_with("http://") || lower.starts_with("https://") {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Reject `url` when its host resolves to a private / loopback / link-local
/// IP address (SSRF defense). User-supplied image URLs are untrusted input
/// and are SSRF-checked regardless of `general_settings.allow_private_api_base`
/// (that setting governs upstream provider endpoints only — never image URLs
/// that arrive in request bodies).
async fn ssrf_guard(url: &str) -> Result<(), oxllm_core::Error> {
    let parsed = url::Url::parse(url)
        .map_err(|e| oxllm_core::Error::BadRequest(format!("invalid image URL '{url}': {e}")))?;
    let host = parsed.host_str().ok_or_else(|| {
        oxllm_core::Error::BadRequest(format!("image URL '{url}' is missing a host"))
    })?;
    if crate::security::is_private_host_async(host).await {
        return Err(oxllm_core::Error::BadRequest(format!(
            "image URL host '{host}' resolves to a private IP address (SSRF blocked)"
        )));
    }
    Ok(())
}

/// Fetch `url` and return a `data:{media_type};base64,{payload}` URL carrying
/// the response body. Enforces the 10 MB size cap both up-front (via
/// `Content-Length`) and during streaming (so a server that lied about the
/// length cannot OOM us). Sniffs the media type from `Content-Type` with a
/// fallback to the URL extension.
///
/// SSRF checks are the caller's responsibility (see [`ssrf_guard`]) — this
/// function performs only the HTTP fetch and re-encode.
async fn download_as_data_url(
    http_client: &reqwest::Client,
    url: &str,
) -> Result<String, oxllm_core::Error> {
    let resp = http_client
        .get(url)
        .send()
        .await
        .map_err(|e| oxllm_core::Error::Network(format!("image download failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(oxllm_core::Error::Network(format!(
            "image download failed: upstream returned status {}",
            resp.status()
        )));
    }
    let header_media_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| {
            ct.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        });
    if let Some(len) = resp.content_length() {
        if len as usize > MAX_IMAGE_BYTES {
            return Err(oxllm_core::Error::BadRequest(format!(
                "image at {url} is too large ({len} bytes); limit is {MAX_IMAGE_BYTES}"
            )));
        }
    }
    // Stream the body with a hard ceiling so a server that lied about
    // Content-Length cannot OOM us. Aborts as soon as the cap is exceeded.
    let bytes = read_capped(resp, url).await?;
    if bytes.is_empty() {
        return Err(oxllm_core::Error::BadRequest(format!(
            "image at {url} returned an empty body"
        )));
    }
    let media_type = resolve_media_type(header_media_type.as_deref(), url, &bytes)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{media_type};base64,{b64}"))
}

/// Stream a reqwest response body into a `Vec<u8>` with a hard size cap.
/// Returns a network-classified error when a chunk read fails; size-cap
/// violations are reported by [`append_with_cap`].
async fn read_capped(resp: reqwest::Response, url: &str) -> Result<Vec<u8>, oxllm_core::Error> {
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|e| oxllm_core::Error::Network(format!("chunked read failed: {e}")))?;
        append_with_cap(&mut buf, &chunk, url)?;
    }
    Ok(buf)
}

/// Extend `buf` with `chunk` and abort when the cumulative size would exceed
/// [`MAX_IMAGE_BYTES`]. The check runs **before** the allocation so a single
/// oversized chunk (e.g. a custom transport delivering a 100 MB chunk instead
/// of hyper's usual 8–64 KiB frames) is rejected without first being buffered.
/// Extracted from [`read_capped`] so the streaming size cap is unit-testable
/// without a real HTTP response.
fn append_with_cap(buf: &mut Vec<u8>, chunk: &[u8], url: &str) -> Result<(), oxllm_core::Error> {
    if chunk.len() > MAX_IMAGE_BYTES.saturating_sub(buf.len()) {
        return Err(oxllm_core::Error::BadRequest(format!(
            "image at {url} exceeded the {MAX_IMAGE_BYTES}-byte limit while streaming"
        )));
    }
    buf.extend_from_slice(chunk);
    Ok(())
}

/// Resolve the media type for the data URL. Preference order:
///
/// 1. The `Content-Type` header — but only when it looks like an image
///    (`image/*`). A non-image `Content-Type` is rejected (we are an image
///    inliner; ferrying arbitrary payloads to Bedrock as base64 would let
///    a caller smuggle non-image bytes past the type contract).
///    `application/octet-stream` is the one exception: S3, GCS, and many
///    CDNs serve objects with that type when their content-type metadata
///    was never set, so an image whose bytes are perfectly valid still
///    arrives as an opaque blob. We treat it as "unknown" and fall through
///    to the extension / magic-byte sniff rather than hard-rejecting.
/// 2. The URL path extension → well-known image media types.
/// 3. Magic-byte sniffing of the payload (`GIF8` → gif, PNG/JPEG/WebP/AVIF
///    signatures) — last-resort fallback so a misconfigured CDN that drops
///    `Content-Type` still produces a usable block.
///
/// Returns `BadRequest` when no image media type can be determined. The
/// magic-byte sniff is the real defence against HTML/JSON smuggling (their
/// bytes match no image signature); the strict non-image `Content-Type`
/// check is a faster, clearer rejection for the common error-page case.
fn resolve_media_type(
    header: Option<&str>,
    url: &str,
    bytes: &[u8],
) -> Result<String, oxllm_core::Error> {
    if let Some(h) = header {
        if h.starts_with("image/") && h.len() > "image/".len() {
            return Ok(h.to_string());
        }
        // `application/octet-stream` is a CDN / object-storage default that
        // frequently fronts valid images whose content-type metadata was
        // never configured — fall through to the extension / magic-byte
        // sniff instead of hard-rejecting a legitimate image.
        if h != "application/octet-stream" {
            // A present-but-non-image Content-Type is a hard error: an image
            // inliner downloading `text/html` (a misconfigured redirect page)
            // or `application/json` (an error envelope) would smuggle
            // non-image bytes into the Bedrock request.
            return Err(oxllm_core::Error::BadRequest(format!(
                "image at {url} has non-image Content-Type '{h}'"
            )));
        }
    }
    if let Some(mt) = guess_media_type_from_extension(url) {
        return Ok(mt);
    }
    if let Some(mt) = sniff_media_type_from_magic(bytes) {
        return Ok(mt);
    }
    Err(oxllm_core::Error::BadRequest(format!(
        "could not determine image content-type for {url}"
    )))
}

/// Map a URL's path extension onto a well-known image media type. Returns
/// `None` for unknown / missing extensions.
fn guess_media_type_from_extension(url: &str) -> Option<String> {
    let parsed = url::Url::parse(url).ok()?;
    let ext = parsed
        .path()
        .rsplit('.')
        .next()
        .map(|e| e.to_ascii_lowercase())?;
    // The first "chunk" from rsplit on a path like `/foo.bar/baz` is `baz`,
    // which is exactly the file extension we want when there is one. When
    // the path ends in a slash (a directory), the chunk is empty and we
    // return None via the match fallthrough.
    match ext.as_str() {
        "png" => Some("image/png".to_string()),
        "jpg" | "jpeg" => Some("image/jpeg".to_string()),
        "gif" => Some("image/gif".to_string()),
        "webp" => Some("image/webp".to_string()),
        "avif" => Some("image/avif".to_string()),
        "bmp" => Some("image/bmp".to_string()),
        "tiff" | "tif" => Some("image/tiff".to_string()),
        _ => None,
    }
}

/// Sniff a few well-known image magic-byte signatures. Returns `None` when
/// no signature matches. The set covers the formats Bedrock's Converse /
/// InvokeModel APIs document as supported (PNG / JPEG / GIF / WebP).
fn sniff_media_type_from_magic(bytes: &[u8]) -> Option<String> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png".to_string());
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return Some("image/jpeg".to_string());
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif".to_string());
    }
    // RIFF....WEBP
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp".to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;

    /// 1×1 PNG (transparent). Used as the canonical image payload in
    /// download / replacement tests.
    const TINY_PNG: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49,
        0x44, 0x41, 0x54, 0x78, 0x9C, 0x62, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A,
        0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    /// Minimal 1×1 JPEG (JFIF). Bytes match the `\xff\xd8\xff` signature
    /// that `sniff_media_type_from_magic` looks for; the rest is a tiny but
    /// syntactically valid JPEG body.
    const TINY_JPEG: &[u8] =
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00";

    fn client() -> reqwest::Client {
        reqwest::Client::builder()
            .build()
            .expect("plain reqwest client builds")
    }

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    // ----------------------------------------------------------------------
    // inline_image_urls — body mutation
    // ----------------------------------------------------------------------

    #[tokio::test]
    async fn no_messages_is_noop() {
        let mut body = json!({"model": "bedrock/claude"});
        inline_image_urls(&mut body, &client()).await.expect("ok");
        assert_eq!(body, json!({"model": "bedrock/claude"}));
    }

    #[tokio::test]
    async fn no_image_url_blocks_is_noop() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ]
        });
        let before = body.clone();
        inline_image_urls(&mut body, &client()).await.expect("ok");
        assert_eq!(body, before, "no image_url blocks → body unchanged");
    }

    #[tokio::test]
    async fn data_url_is_left_untouched() {
        // The Converse / InvokeModel provider transforms handle `data:`
        // URLs natively; the inliner only downloads HTTP(S) URLs.
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG="}},
                ],
            }]
        });
        let before = body.clone();
        inline_image_urls(&mut body, &client()).await.expect("ok");
        assert_eq!(body, before, "data: URLs must not be re-downloaded");
    }

    #[tokio::test]
    async fn image_url_with_non_string_url_is_skipped() {
        // Malformed block (url not a string) — skip rather than crash.
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": 42}},
                ],
            }]
        });
        inline_image_urls(&mut body, &client()).await.expect("ok");
        assert_eq!(body["messages"][0]["content"][0]["image_url"]["url"], 42);
    }

    #[tokio::test]
    async fn ssrf_blocks_loopback_host() {
        // httpmock binds 127.0.0.1, which is in the private CIDR list — the
        // perfect fixture for the SSRF guard. No real assertion on the
        // mock response is needed: the guard must reject before any HTTP
        // request is issued, so we register a sentinel mock and verify it
        // was never hit.
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/x.png");
            then.status(200).body(TINY_PNG);
        });
        let url = server.base_url();
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": format!("{url}/x.png")}},
                ],
            }]
        });
        let err = inline_image_urls(&mut body, &client())
            .await
            .expect_err("loopback must be SSRF-blocked");
        let msg = err.to_string();
        assert!(
            msg.contains("private IP") || msg.contains("SSRF"),
            "expected SSRF error, got: {msg}"
        );
        // Body untouched on failure.
        assert!(
            body["messages"][0]["content"][0]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("http")
        );
        // No HTTP request reached the mock.
        assert_eq!(mock.hits(), 0);
    }

    /// C13/MEDIUM: a body carrying more than [`MAX_IMAGE_BLOCKS`] eligible
    /// `image_url` blocks is rejected up-front with `BadRequest` — the count
    /// cap is a DoS-amplification guard (see the constant's docs). The cap
    /// trips before any SSRF check or HTTP GET, so the mock must never be
    /// hit and the error message must be the cap message (not the loopback
    /// SSRF message those 127.0.0.1 URLs would otherwise produce).
    #[tokio::test]
    async fn rejects_body_with_too_many_image_url_blocks() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/img.png");
            then.status(200)
                .header("content-type", "image/png")
                .body(TINY_PNG);
        });
        let url = format!("{}/img.png", server.base_url());
        let over = MAX_IMAGE_BLOCKS + 1;
        let blocks: Vec<Value> = (0..over)
            .map(|_| json!({"type": "image_url", "image_url": {"url": url.clone()}}))
            .collect();
        let mut body = json!({"messages": [{"role": "user", "content": blocks}]});
        let before = body.clone();

        let err = inline_image_urls(&mut body, &client())
            .await
            .expect_err("too many image_url blocks must be rejected");
        assert!(
            matches!(err, oxllm_core::Error::BadRequest(_)),
            "expected BadRequest, got: {err:?}"
        );
        assert!(
            err.to_string().contains("image_url blocks"),
            "expected cap message, got: {err}"
        );
        // No HTTP GET issued and body left untouched (transactional + upfront).
        assert_eq!(mock.hits(), 0, "count cap must trip before any download");
        assert_eq!(body, before, "body must be untouched on upfront reject");
    }

    /// C13/MEDIUM: `data:` URLs must NOT count toward the block cap. A
    /// legitimate body with many pre-inlined `data:` images is unaffected.
    #[tokio::test]
    async fn data_url_blocks_do_not_count_toward_cap() {
        let over = MAX_IMAGE_BLOCKS + 1;
        let blocks: Vec<Value> = (0..over)
            .map(|_| {
                json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR="}})
            })
            .collect();
        let mut body = json!({"messages": [{"role": "user", "content": blocks}]});
        let before = body.clone();
        inline_image_urls(&mut body, &client())
            .await
            .expect("data: blocks are not fetched and must not trip the cap");
        assert_eq!(body, before);
    }

    /// C13/LOW: `inline_image_urls` is transactional — a hard error on any
    /// block leaves the body untouched. We exercise a multi-block walk: a
    /// `data:` block (skipped, never rewritten) followed by a loopback
    /// `image_url` block that trips the SSRF guard. The body must come back
    /// byte-for-byte identical, proving no partial mutation leaks on `Err`.
    ///
    /// Note: the stricter sub-case — block *i* downloads successfully and is
    /// rewritten, then block *i+1* fails — cannot be exercised here because
    /// the SSRF guard rejects every loopback host (httpmock binds 127.0.0.1),
    /// so no `inline_image_urls` happy-path rewrite is reachable from this
    /// test module. The transactional contract for that branch is guaranteed
    /// by construction: downloads accumulate in a `rewrites` side table that
    /// is applied to the body only after the full scan succeeds.
    #[tokio::test]
    async fn multi_block_partial_failure_leaves_body_consistent() {
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR="}},
                    {"type": "image_url", "image_url": {"url": "http://127.0.0.1:1/blocked.png"}},
                ],
            }]
        });
        let before = body.clone();
        let err = inline_image_urls(&mut body, &client())
            .await
            .expect_err("second block must SSRF-fail");
        assert!(
            err.to_string().contains("private IP") || err.to_string().contains("SSRF"),
            "expected SSRF error, got: {err}"
        );
        assert_eq!(
            body, before,
            "body must be fully original on partial failure (transactional)"
        );
    }

    #[tokio::test]
    async fn malformed_url_is_rejected() {
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://[::1"}} ,
                ],
            }]
        });
        let err = inline_image_urls(&mut body, &client())
            .await
            .expect_err("malformed URL must error");
        assert!(err.to_string().contains("invalid image URL"));
    }

    // ----------------------------------------------------------------------
    // download_as_data_url — HTTP fetch / re-encode (no SSRF at this layer)
    // ----------------------------------------------------------------------

    #[tokio::test]
    async fn download_png_replaces_url() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/img.png");
            then.status(200)
                .header("content-type", "image/png")
                .body(TINY_PNG);
        });

        let data_url = download_as_data_url(&client(), &format!("{}/img.png", server.base_url()))
            .await
            .expect("download ok");
        assert!(
            data_url.starts_with("data:image/png;base64,"),
            "expected data:image/png URL, got: {data_url}"
        );
        let payload = data_url.strip_prefix("data:image/png;base64,").unwrap();
        assert_eq!(payload, b64(TINY_PNG));
    }

    #[tokio::test]
    async fn download_jpeg_replaces_url() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/img.jpg");
            then.status(200)
                .header("content-type", "image/jpeg")
                .body(TINY_JPEG);
        });

        let data_url = download_as_data_url(&client(), &format!("{}/img.jpg", server.base_url()))
            .await
            .expect("download ok");
        assert!(data_url.starts_with("data:image/jpeg;base64,"));
    }

    #[tokio::test]
    async fn download_uses_extension_when_content_type_missing() {
        // Misconfigured CDN drops Content-Type — fall back to the URL
        // extension (.png).
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/img.png");
            then.status(200).body(TINY_PNG);
        });

        let data_url = download_as_data_url(&client(), &format!("{}/img.png", server.base_url()))
            .await
            .expect("download ok");
        assert!(
            data_url.starts_with("data:image/png;base64,"),
            "extension fallback should produce image/png"
        );
    }

    #[tokio::test]
    async fn download_uses_magic_bytes_when_no_content_type_or_extension() {
        // No Content-Type, no extension — sniff PNG magic bytes.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/img");
            then.status(200).body(TINY_PNG);
        });

        let data_url = download_as_data_url(&client(), &format!("{}/img", server.base_url()))
            .await
            .expect("download ok");
        assert!(
            data_url.starts_with("data:image/png;base64,"),
            "magic-byte sniff should produce image/png"
        );
    }

    #[tokio::test]
    async fn download_rejects_non_image_content_type() {
        // A present-but-non-image Content-Type is a hard error: prevents
        // ferrying HTML error pages / JSON envelopes into the Bedrock body.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/not-an-image");
            then.status(200)
                .header("content-type", "text/html")
                .body(b"<html>redirect</html>");
        });

        let err = download_as_data_url(&client(), &format!("{}/not-an-image", server.base_url()))
            .await
            .expect_err("non-image Content-Type must error");
        assert!(
            err.to_string().contains("non-image Content-Type"),
            "expected non-image error, got: {err}"
        );
    }

    /// C13/MEDIUM: S3 / GCS / many CDNs serve images as
    /// `application/octet-stream` when content-type metadata was never set.
    /// That must NOT be a hard reject — fall through to the magic-byte sniff
    /// and produce a usable `data:image/png` URL from the valid PNG body.
    #[tokio::test]
    async fn download_png_with_octet_stream_content_type_succeeds() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/img.png");
            then.status(200)
                .header("content-type", "application/octet-stream")
                .body(TINY_PNG);
        });

        let data_url = download_as_data_url(&client(), &format!("{}/img.png", server.base_url()))
            .await
            .expect("octet-stream + valid PNG must succeed");
        assert!(
            data_url.starts_with("data:image/png;base64,"),
            "octet-stream should fall through to extension/magic sniff → image/png, got: {data_url}"
        );
        let payload = data_url.strip_prefix("data:image/png;base64,").unwrap();
        assert_eq!(payload, b64(TINY_PNG));
    }

    /// C13/MEDIUM: even with `application/octet-stream` and no recognisable
    /// extension, a valid PNG body is rescued by the magic-byte sniff.
    #[tokio::test]
    async fn download_octet_stream_falls_back_to_magic_bytes() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/blob");
            then.status(200)
                .header("content-type", "application/octet-stream")
                .body(TINY_PNG);
        });

        let data_url = download_as_data_url(&client(), &format!("{}/blob", server.base_url()))
            .await
            .expect("octet-stream + magic bytes must succeed");
        assert!(
            data_url.starts_with("data:image/png;base64,"),
            "magic-byte sniff should produce image/png, got: {data_url}"
        );
    }

    #[tokio::test]
    async fn download_rejects_oversized_content_length() {
        // Register a real body above the cap — httpmock derives
        // Content-Length from the body, so the up-front check trips before
        // we start streaming.
        let oversized = vec![0u8; MAX_IMAGE_BYTES + 1];
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/big.png");
            then.status(200)
                .header("content-type", "image/png")
                .body(oversized.as_slice());
        });

        let err = download_as_data_url(&client(), &format!("{}/big.png", server.base_url()))
            .await
            .expect_err("oversized Content-Length must error");
        assert!(
            err.to_string().contains("too large"),
            "expected size error, got: {err}"
        );
    }

    #[tokio::test]
    async fn download_rejects_stream_above_cap() {
        // The streaming cap is the defense-in-depth backstop for the case
        // where a server omits or lies about Content-Length. httpmock
        // always derives Content-Length from the body, so an end-to-end
        // test is not feasible without a chunked-encoding mock — exercise
        // the streaming-append guard directly instead.
        let mut buf = Vec::new();
        let oversized = vec![0u8; MAX_IMAGE_BYTES + 1];
        let err = append_with_cap(&mut buf, &oversized, "https://example/x.png")
            .expect_err("streaming cap must trip");
        let msg = err.to_string();
        assert!(
            msg.contains("exceeded") || msg.contains("too large"),
            "expected streaming cap error, got: {msg}"
        );
        // The pre-check rejects without buffering the offending chunk.
        assert!(
            buf.is_empty(),
            "buffer must not be extended when the chunk alone exceeds the cap"
        );
    }

    /// C13/LOW: a single chunk larger than the entire cap must be rejected
    /// **without** being buffered first. The pre-check compares the chunk
    /// against the remaining budget before `extend_from_slice`, so a custom
    /// transport that delivers an oversized frame cannot OOM us.
    #[test]
    fn append_with_cap_rejects_oversized_chunk_without_buffering() {
        let mut buf: Vec<u8> = Vec::new();
        let oversized = vec![0u8; MAX_IMAGE_BYTES + 1];
        let err = append_with_cap(&mut buf, &oversized, "https://example/x.png")
            .expect_err("oversized chunk must be rejected");
        assert!(err.to_string().contains("exceeded"));
        assert!(
            buf.is_empty(),
            "buffer must remain empty when the chunk alone exceeds the cap"
        );
    }

    #[test]
    fn append_with_cap_allows_at_limit() {
        let mut buf = Vec::new();
        let chunk = vec![0u8; MAX_IMAGE_BYTES];
        append_with_cap(&mut buf, &chunk, "https://example/x.png").expect("at-limit ok");
        assert_eq!(buf.len(), MAX_IMAGE_BYTES);
    }

    #[test]
    fn append_with_cap_rejects_one_byte_over() {
        let mut buf = vec![0u8; MAX_IMAGE_BYTES];
        assert!(
            append_with_cap(&mut buf, b"\x00", "https://example/x.png").is_err(),
            "one byte over the cap must error"
        );
    }

    #[tokio::test]
    async fn download_rejects_4xx_response() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/missing.png");
            then.status(404).body("not found".as_bytes());
        });

        let err = download_as_data_url(&client(), &format!("{}/missing.png", server.base_url()))
            .await
            .expect_err("404 must error");
        assert!(
            matches!(err, oxllm_core::Error::Network(_)),
            "non-2xx status must map to Network (502), got: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("404") || msg.contains("image download failed"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn download_rejects_connection_failure() {
        // Port 1 is reserved and unreachable on every reasonable OS — a
        // guaranteed connection failure with no mock server.
        let err = download_as_data_url(&client(), "http://example.invalid:1/x.png")
            .await
            .expect_err("DNS / connect failure must error");
        assert!(
            matches!(err, oxllm_core::Error::Network(_)),
            "expected Network error, got: {err:?}"
        );
    }

    /// C13/MEDIUM: a slow-loris image host sends valid headers + one byte
    /// then stalls the body forever. Without a client-level total deadline
    /// the `bytes_stream` future would hang indefinitely, pinning the
    /// orchestration task and the caller's connection. The image-download
    /// client ([`crate::state::build::build_image_http_client`]) carries a
    /// total `.timeout(...)`, so the request aborts and surfaces
    /// `Error::Network` within `timeout + ε`.
    ///
    /// The server here is a plain `std::net::TcpListener` on an OS thread
    /// (the workspace tokio build does not enable the `net` feature), and
    /// the client is a plain reqwest builder with a short `.timeout()`
    /// (no SSRF resolver, so the loopback connect succeeds). This exercises
    /// the consumer side of the fix: `download_as_data_url` honours the
    /// client-level total timeout for both the headers fetch and the body
    /// stream. The default applied by `build_image_http_client`
    /// ([`crate::state::build::DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECS`]) is the
    /// same mechanism at a production cadence.
    #[tokio::test]
    async fn download_aborts_when_server_slow_loris() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::time::Duration;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        let url = format!("http://127.0.0.1:{port}/img.png");

        std::thread::spawn(move || {
            let (mut sock, _) = match listener.accept() {
                Ok(p) => p,
                Err(_) => return,
            };
            // 200 OK + a Content-Length we will never fulfil, then a single
            // body byte before stalling forever.
            let header = b"HTTP/1.1 200 OK\r\n\
                           Content-Type: image/png\r\n\
                           Content-Length: 100\r\n\
                           \r\n\
                           X";
            let _ = sock.write_all(header);
            // Hold the connection open; the client timeout must fire.
            std::thread::sleep(Duration::from_secs(60));
            let _ = sock;
        });

        let slow_client = reqwest::Client::builder()
            .timeout(Duration::from_millis(250))
            .build()
            .expect("client builds");

        let start = tokio::time::Instant::now();
        let err = download_as_data_url(&slow_client, &url)
            .await
            .expect_err("slow-loris must be aborted by the total timeout");
        let elapsed = start.elapsed();
        assert!(
            matches!(err, oxllm_core::Error::Network(_)),
            "expected Network error from timeout, got: {err:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "total timeout should abort within ~250ms, took {elapsed:?}"
        );
    }

    /// C13/HIGH regression: when an operator opts into
    /// `allow_private_api_base=true`, the shared `http_client` loses its
    /// connect-time SSRF resolver. User-supplied image URLs must stay
    /// protected, so the dedicated image-download client
    /// ([`crate::state::build::build_image_http_client`]) always carries the
    /// resolver. Verify by building it against a relaxed config and asserting
    /// it cannot reach a loopback httpmock server via `localhost` (which
    /// resolves to 127.0.0.1) — the connect-time resolver must block before
    /// any byte is served. `download_as_data_url` is called directly to
    /// exercise the connect layer in isolation from the best-effort
    /// `ssrf_guard` pre-check.
    #[tokio::test]
    async fn image_client_blocks_loopback_when_allow_private_api_base_true() {
        use crate::config::types::{
            ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings,
        };
        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "claude".to_string(),
                oxllm_params: oxllm_core::deployment::OxllmParams {
                    model: "bedrock/anthropic.claude-3-5-sonnet".to_string(),
                    api_key: Some("k".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        // Sanity: the relaxed shared client would NOT install the resolver.
        // The image client must install it regardless.
        let image_client = crate::state::build_image_http_client(&config)
            .expect("image client builds with allow_private_api_base=true");

        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/x.png");
            then.status(200).body(TINY_PNG);
        });
        // Route through `localhost` so the request hits DNS (and thus the
        // connect-time resolver) rather than the literal-IP fast path.
        let url = format!(
            "{}/x.png",
            server.base_url().replace("127.0.0.1", "localhost")
        );

        let err = download_as_data_url(&image_client, &url)
            .await
            .expect_err("loopback host must be blocked at connect time");
        assert!(
            matches!(err, oxllm_core::Error::Network(_)),
            "expected Network (SSRF connect-time block), got: {err:?}"
        );
        assert_eq!(
            mock.hits(),
            0,
            "connect-time resolver must reject before the mock serves any byte"
        );
    }

    // ----------------------------------------------------------------------
    // needs_inlining — orchestration gate
    // ----------------------------------------------------------------------

    #[test]
    fn needs_inlining_false_when_no_image_url_block() {
        let state = bedrock_state();
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(!needs_inlining(&state, "claude", &body));
    }

    #[test]
    fn needs_inlining_true_for_bedrock_with_http_image_url() {
        let state = bedrock_state();
        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                ],
            }]
        });
        assert!(needs_inlining(&state, "claude", &body));
    }

    #[test]
    fn needs_inlining_false_for_data_url_only() {
        let state = bedrock_state();
        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR="}},
                ],
            }]
        });
        assert!(!needs_inlining(&state, "claude", &body));
    }

    #[test]
    fn needs_inlining_false_for_non_bedrock_model() {
        // An OpenAI deployment of the same model group: OpenAI fetches
        // HTTP URLs natively, so the proxy must not pre-download.
        let state = openai_state();
        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                ],
            }]
        });
        assert!(!needs_inlining(&state, "gpt", &body));
    }

    #[test]
    fn needs_inlining_unknown_model_is_false() {
        let state = bedrock_state();
        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                ],
            }]
        });
        assert!(!needs_inlining(&state, "unknown-model", &body));
    }

    #[test]
    fn inlining_follows_alias_and_wildcard_resolution() {
        use oxllm_core::router::{Router, RouterConfig};
        let body = json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"https://example.com/x.png"}}
        ]}]});
        let mut state = bedrock_state();
        state.router = std::sync::Arc::new(Router::new(RouterConfig {
            model_group_alias: [("alias".to_string(), vec!["claude".to_string()])].into(),
            ..Default::default()
        }));
        assert!(needs_inlining(&state, "alias", &body));
        let wildcard = state_with(&[("*", "bedrock/anthropic.*")]);
        assert!(needs_inlining(&wildcard, "bedrock/anthropic.claude", &body));
        assert!(!needs_inlining(&wildcard, "openai/gpt-4o", &body));
    }

    #[test]
    fn inlining_covers_complexity_tier_and_default_pools() {
        use oxllm_core::router::{ComplexityRouterConfig, Router, RouterConfig};
        let body = json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"https://example.com/x.png"}}
        ]}]});
        for default_pool in [false, true] {
            let mut state = state_with(&[
                ("auto", "openai/gpt-4o"),
                ("vision", "bedrock/anthropic.claude"),
            ]);
            let complexity = ComplexityRouterConfig {
                tiers: if default_pool {
                    Default::default()
                } else {
                    [("simple".to_string(), vec!["vision-alias".to_string()])].into()
                },
                default_model: default_pool.then(|| "vision-alias".to_string()),
                ..Default::default()
            };
            state.router = std::sync::Arc::new(Router::new(RouterConfig {
                routing_strategy: oxllm_core::RoutingStrategyKind::ComplexityRouter,
                complexity_router_config: Some(complexity),
                model_group_alias: [("vision-alias".to_string(), vec!["vision".to_string()])]
                    .into(),
                ..Default::default()
            }));
            let ctx = oxllm_core::RequestContext::new("auto");
            assert_eq!(
                state
                    .router
                    .pick_deployment(&state.deployments, &ctx)
                    .unwrap()
                    .provider_slug(),
                "bedrock"
            );
            assert!(needs_inlining(&state, "auto", &body));
        }
    }

    // ----------------------------------------------------------------------
    // resolve_media_type — pure unit tests
    // ----------------------------------------------------------------------

    #[test]
    fn resolve_media_type_prefers_image_content_type() {
        let mt = resolve_media_type(Some("image/png"), "https://x/y", TINY_PNG).unwrap();
        assert_eq!(mt, "image/png");
    }

    #[test]
    fn resolve_media_type_rejects_non_image_content_type() {
        let err = resolve_media_type(Some("text/html"), "https://x/y", b"<html>".as_ref());
        assert!(err.is_err());
    }

    #[test]
    fn resolve_media_type_falls_back_to_extension() {
        let mt = resolve_media_type(None, "https://x/y.jpeg", TINY_JPEG).unwrap();
        assert_eq!(mt, "image/jpeg");
    }

    #[test]
    fn resolve_media_type_falls_back_to_magic_bytes() {
        let mt = resolve_media_type(None, "https://x/y", TINY_PNG).unwrap();
        assert_eq!(mt, "image/png");
    }

    /// C13/MEDIUM: `application/octet-stream` (CDN / object-storage default)
    /// is treated as "unknown" and falls through to the extension / magic
    /// sniff rather than hard-rejecting a valid image.
    #[test]
    fn resolve_media_type_octet_stream_falls_through_to_magic() {
        // octet-stream + no extension + PNG magic bytes → sniff wins.
        let mt =
            resolve_media_type(Some("application/octet-stream"), "https://x/y", TINY_PNG).unwrap();
        assert_eq!(mt, "image/png");
    }

    #[test]
    fn resolve_media_type_octet_stream_falls_through_to_extension() {
        // octet-stream + .jpeg extension → extension wins over sniff.
        let mt = resolve_media_type(
            Some("application/octet-stream"),
            "https://x/y.jpeg",
            TINY_JPEG,
        )
        .unwrap();
        assert_eq!(mt, "image/jpeg");
    }

    #[test]
    fn resolve_media_type_octet_stream_rejects_non_image_bytes() {
        // octet-stream + non-image bytes (no extension, no magic) → the sniff
        // still rejects, so HTML/JSON smuggling via octet-stream is closed.
        let err = resolve_media_type(
            Some("application/octet-stream"),
            "https://x/y",
            b"<html>".as_ref(),
        );
        assert!(
            err.is_err(),
            "octet-stream must still reject non-image bytes"
        );
    }

    #[test]
    fn resolve_media_type_rejects_when_no_signal() {
        // No header, unknown extension, unknown magic.
        let err = resolve_media_type(None, "https://x/y.dat", b"\x00\x00\x00".as_ref());
        assert!(err.is_err());
    }

    // ----------------------------------------------------------------------
    // Test state fixtures
    // ----------------------------------------------------------------------

    fn bedrock_state() -> AppState {
        state_with(&[("claude", "bedrock/anthropic.claude-3-5-sonnet")])
    }

    fn openai_state() -> AppState {
        state_with(&[("gpt", "openai/gpt-4o")])
    }

    fn state_with(models: &[(&str, &str)]) -> AppState {
        use crate::config::types::{
            ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings,
        };
        let config = ProxyConfig {
            model_list: models
                .iter()
                .map(|(name, model)| ConfigModel {
                    model_name: name.to_string(),
                    oxllm_params: oxllm_core::deployment::OxllmParams {
                        model: model.to_string(),
                        api_key: Some("k".to_string()),
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .collect(),
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }
}
