//! Minimal multipart/form-data field inspection for inference endpoints whose
//! model selector lives in a form field (`images/edits`, `images/variations`,
//! `audio/transcriptions`, `audio/translations`).
//!
//! Nothing here parses the whole body into parts: the primary `model` field is
//! located for routing, and [`rewrite_field`] replaces exactly that field's
//! content bytes while preserving every other byte of the body (boundaries,
//! headers, binary payloads) verbatim. A request whose boundary or field
//! cannot be located is left for the caller to reject, so no unscanned or
//! unmetered inference request can fall back to a silent deployment pick.

use axum::body::Bytes;

/// Extract the multipart boundary from a `Content-Type` header value.
/// Returns `None` when the value is not multipart/form-data or the boundary
/// is missing/empty/oversized (RFC 2046 caps boundaries at 70 characters).
pub(crate) fn boundary_from_content_type(content_type: Option<&str>) -> Option<String> {
    let content_type = content_type?;
    if !content_type
        .to_ascii_lowercase()
        .contains("multipart/form-data")
    {
        return None;
    }
    for param in content_type.split(';').skip(1) {
        let param = param.trim();
        let lower = param.to_ascii_lowercase();
        let Some(rest) = lower.strip_prefix("boundary=") else {
            continue;
        };
        // Preserve the original casing of the boundary itself.
        let original = &param[param.len() - rest.len()..];
        let value = original.trim().trim_matches('"');
        if value.is_empty() || value.len() > 70 || value.contains('\r') || value.contains('\n') {
            return None;
        }
        return Some(value.to_string());
    }
    None
}

fn find_from(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() || from > haystack.len() - needle.len() {
        return None;
    }
    let first = needle[0];
    let mut index = from;
    let last = haystack.len() - needle.len();
    while index <= last {
        if haystack[index] == first && &haystack[index..index + needle.len()] == needle {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Does this part's header block declare `Content-Disposition: ...; name=".."`?
fn headers_declare_name(headers: &[u8], name: &str) -> bool {
    let Ok(headers) = std::str::from_utf8(headers) else {
        return false;
    };
    for line in headers.split("\r\n") {
        let Some((header, value)) = line.split_once(':') else {
            continue;
        };
        if !header.trim().eq_ignore_ascii_case("content-disposition") {
            continue;
        }
        for param in value.split(';').skip(1) {
            let param = param.trim();
            let Some((key, raw)) = param.split_once('=') else {
                continue;
            };
            if key.trim().eq_ignore_ascii_case("name") && raw.trim().trim_matches('"') == name {
                return true;
            }
        }
    }
    false
}

/// Byte range of the content of the first part whose `name` matches.
/// `None` when the field is absent or the framing is malformed.
fn field_bounds(body: &[u8], boundary: &str, name: &str) -> Option<(usize, usize)> {
    let delimiter = format!("--{boundary}");
    let delimiter = delimiter.as_bytes();
    let mut cursor = find_from(body, delimiter, 0)?;
    loop {
        let after = cursor + delimiter.len();
        // Closing delimiter (`--boundary--`): no more parts.
        if body.get(after..after + 2) == Some(b"--") {
            return None;
        }
        let mut cursor_after_padding = after;
        while matches!(body.get(cursor_after_padding), Some(b' ') | Some(b'\t')) {
            cursor_after_padding += 1;
        }
        if body.get(cursor_after_padding..cursor_after_padding + 2) != Some(b"\r\n") {
            return None;
        }
        let headers_start = cursor_after_padding + 2;
        let header_end = find_from(body, b"\r\n\r\n", headers_start)?;
        let content_start = header_end + 4;
        let next_delimiter = find_from(body, delimiter, content_start)?;
        // The CRLF that precedes the next delimiter belongs to the framing,
        // not to the field content.
        let content_end = if next_delimiter >= content_start + 2
            && &body[next_delimiter - 2..next_delimiter] == b"\r\n"
        {
            next_delimiter - 2
        } else {
            next_delimiter
        };
        if headers_declare_name(&body[headers_start..header_end], name) {
            return Some((content_start, content_end));
        }
        cursor = next_delimiter;
    }
}

/// Value of a simple text field, trimmed. `None` when the body is not
/// multipart or the field is absent/malformed.
pub(crate) fn extract_field(body: &[u8], content_type: Option<&str>, name: &str) -> Option<String> {
    let boundary = boundary_from_content_type(content_type)?;
    let (start, end) = field_bounds(body, &boundary, name)?;
    let value = String::from_utf8_lossy(&body[start..end]);
    Some(value.trim().to_string())
}

/// Body with the (first) matching field's content replaced by `value`.
/// Every other byte — boundary framing, part headers, binary payloads — is
/// preserved verbatim. `None` when the body is not multipart or the field is
/// absent/malformed.
pub(crate) fn rewrite_field(
    body: &[u8],
    content_type: Option<&str>,
    name: &str,
    value: &str,
) -> Option<Bytes> {
    let boundary = boundary_from_content_type(content_type)?;
    let (start, end) = field_bounds(body, &boundary, name)?;
    let mut rebuilt = Vec::with_capacity(body.len() + value.len());
    rebuilt.extend_from_slice(&body[..start]);
    rebuilt.extend_from_slice(value.as_bytes());
    rebuilt.extend_from_slice(&body[end..]);
    Some(Bytes::from(rebuilt))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CT: Option<&str> = Some("multipart/form-data; boundary=xyz");

    fn body(parts: &[(&str, &[u8])]) -> Vec<u8> {
        body_with_boundary("xyz", parts)
    }

    fn body_with_boundary(boundary: &str, parts: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, value) in parts {
            out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            out.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            out.extend_from_slice(value);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        out
    }

    #[test]
    fn boundary_parsing_handles_quotes_and_rejects_bad_values() {
        assert_eq!(
            boundary_from_content_type(Some("multipart/form-data; boundary=\"a b\"")).as_deref(),
            Some("a b")
        );
        assert_eq!(
            boundary_from_content_type(Some("multipart/form-data; boundary=ABC")).as_deref(),
            Some("ABC")
        );
        assert!(boundary_from_content_type(Some("multipart/form-data")).is_none());
        assert!(boundary_from_content_type(Some("application/json")).is_none());
        assert!(boundary_from_content_type(None).is_none());
        assert!(boundary_from_content_type(Some("multipart/form-data; boundary=")).is_none());
        assert!(
            boundary_from_content_type(Some(&format!(
                "multipart/form-data; boundary={}",
                "x".repeat(71)
            )))
            .is_none()
        );
    }

    #[test]
    fn extracts_the_model_field_from_a_multipart_body() {
        let body = body(&[("model", b"whisper-1"), ("file", b"binary\x00data")]);
        assert_eq!(
            extract_field(&body, CT, "model").as_deref(),
            Some("whisper-1")
        );
        assert_eq!(extract_field(&body, CT, "absent"), None);
        assert_eq!(
            extract_field(b"{\"model\":\"x\"}", Some("application/json"), "model"),
            None
        );
    }

    #[test]
    fn rewrites_only_the_model_value_and_preserves_binary_bytes() {
        let original = body(&[("file", b"raw\x00\xff\xfebytes"), ("model", b"my-alias")]);
        let rewritten = rewrite_field(&original, CT, "model", "whisper-1").expect("rewrite");
        assert_eq!(
            extract_field(&rewritten, CT, "model").as_deref(),
            Some("whisper-1")
        );
        // `extract_field` is lossy for binary content (it returns a String),
        // so verify byte preservation by scanning the rebuilt bytes directly.
        let needle = b"raw\x00\xff\xfebytes";
        assert!(
            rewritten
                .windows(needle.len())
                .any(|window| window == needle),
            "binary field content must be preserved byte-for-byte"
        );
        // Framing bytes outside the model value are untouched.
        assert!(rewritten.ends_with(b"--xyz--\r\n"));
    }

    #[test]
    fn rewrite_returns_none_when_the_field_is_missing() {
        let body = body(&[("file", b"x")]);
        assert!(rewrite_field(&body, CT, "model", "y").is_none());
    }

    #[test]
    fn stops_at_the_closing_delimiter_without_confusing_a_later_boundary() {
        // Content that happens to contain a boundary-like byte sequence is
        // not guaranteed by the contract and must not crash the scan.
        let mut body = body(&[("model", b"a"), ("other", b"b")]);
        body.extend_from_slice(b"epilogue");
        assert_eq!(extract_field(&body, CT, "model").as_deref(), Some("a"));
    }

    /// A quoted boundary containing spaces (legal per RFC 2046) must work end
    /// to end, not only when parsing the `Content-Type` header.
    #[test]
    fn quoted_boundary_with_spaces_round_trips_extraction_and_rewrite() {
        let ct = Some("multipart/form-data; boundary=\"spaced boundary\"");
        let original = body_with_boundary(
            "spaced boundary",
            &[("model", b"alias"), ("file", b"payload")],
        );
        assert_eq!(
            extract_field(&original, ct, "model").as_deref(),
            Some("alias")
        );
        let rewritten = rewrite_field(&original, ct, "model", "upstream-model").expect("rewrite");
        assert_eq!(
            extract_field(&rewritten, ct, "model").as_deref(),
            Some("upstream-model")
        );
        let file = b"payload";
        assert!(
            rewritten.windows(file.len()).any(|window| window == file),
            "the following part must survive the rewrite"
        );
        assert!(rewritten.ends_with(b"--spaced boundary--\r\n"));
    }

    /// Header and parameter names are matched case-insensitively (real
    /// clients and proxies do not normalize their casing).
    #[test]
    fn uppercase_header_and_parameter_names_are_matched() {
        let mut original = Vec::new();
        original.extend_from_slice(b"--xyz\r\n");
        original.extend_from_slice(b"CONTENT-DISPOSITION: form-data; NAME=\"model\"\r\n\r\n");
        original.extend_from_slice(b"my-alias");
        original.extend_from_slice(b"\r\n--xyz--\r\n");
        assert_eq!(
            extract_field(&original, CT, "model").as_deref(),
            Some("my-alias")
        );
        let rewritten = rewrite_field(&original, CT, "model", "upstream").expect("rewrite");
        assert_eq!(
            extract_field(&rewritten, CT, "model").as_deref(),
            Some("upstream")
        );
    }

    /// Transport padding (spaces/tabs) between the boundary and its CRLF is
    /// legal; it must not shift the header/content offsets.
    #[test]
    fn whitespace_after_the_boundary_is_tolerated() {
        let mut original = Vec::new();
        original.extend_from_slice(b"--xyz \t \r\n");
        original.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
        original.extend_from_slice(b"alias");
        original.extend_from_slice(b"\r\n--xyz--\r\n");
        assert_eq!(
            extract_field(&original, CT, "model").as_deref(),
            Some("alias")
        );
        let rewritten = rewrite_field(&original, CT, "model", "upstream").expect("rewrite");
        assert_eq!(
            extract_field(&rewritten, CT, "model").as_deref(),
            Some("upstream")
        );
        assert!(rewritten.starts_with(b"--xyz \t \r\n"));
    }

    /// A body whose final boundary is missing is malformed: the scan must
    /// stop safely, while a field that is delimited by a later part is still
    /// found.
    #[test]
    fn missing_closing_delimiter_fails_closed_for_the_open_ended_part() {
        let mut body = Vec::new();
        body.extend_from_slice(b"--xyz\r\n");
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
        body.extend_from_slice(b"first-alias\r\n");
        // Second part runs to EOF with no closing delimiter.
        body.extend_from_slice(b"--xyz\r\n");
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"file\"\r\n\r\n");
        body.extend_from_slice(b"dangling");
        assert_eq!(
            extract_field(&body, CT, "model").as_deref(),
            Some("first-alias")
        );
        assert!(extract_field(&body, CT, "file").is_none());
        assert!(rewrite_field(&body, CT, "file", "x").is_none());
        assert_eq!(
            rewrite_field(&body, CT, "model", "upstream")
                .and_then(|bytes| extract_field(&bytes, CT, "model")),
            Some("upstream".to_string())
        );
    }

    /// Boundary-like bytes inside a part's content are a contract violation
    /// (encoders must escape them). The scanner must fail closed instead of
    /// panicking, whether the lookalike has a CRLF-less suffix or an apparent
    /// closing delimiter.
    #[test]
    fn boundary_like_content_fails_closed_without_panicking() {
        let embedded = body(&[
            ("file", b"prefix--xyz-not-a-delimiter"),
            ("model", b"alias"),
        ]);
        assert!(extract_field(&embedded, CT, "model").is_none());
        assert!(rewrite_field(&embedded, CT, "model", "upstream").is_none());

        let embedded_closing = body(&[("file", b"payload--xyz--tail"), ("model", b"alias")]);
        assert!(extract_field(&embedded_closing, CT, "model").is_none());
        assert!(rewrite_field(&embedded_closing, CT, "model", "upstream").is_none());
    }

    /// Duplicate `model` fields are a contract violation: the first match
    /// wins, and a rewrite must not touch the later duplicate.
    #[test]
    fn duplicate_model_fields_use_the_first_match() {
        let original = body(&[
            ("model", b"first-alias"),
            ("model", b"second-alias"),
            ("file", b"payload"),
        ]);
        assert_eq!(
            extract_field(&original, CT, "model").as_deref(),
            Some("first-alias")
        );
        let rewritten = rewrite_field(&original, CT, "model", "upstream-model").expect("rewrite");
        assert_eq!(
            extract_field(&rewritten, CT, "model").as_deref(),
            Some("upstream-model")
        );
        let needle = b"second-alias";
        assert!(
            rewritten
                .windows(needle.len())
                .any(|window| window == needle),
            "the duplicate field must be preserved verbatim"
        );
    }

    /// Rewriting to a longer then a shorter value must preserve the boundary
    /// framing and every following part byte-for-byte.
    #[test]
    fn rewriting_longer_and_shorter_preserves_framing_and_following_parts() {
        let original = body(&[
            ("model", b"m"),
            ("file", b"raw\x00\xff\xfebytes"),
            ("tail", b"tail-value"),
        ]);
        let framing_count = original
            .windows(b"--xyz".len())
            .filter(|window| *window == b"--xyz")
            .count();
        assert_eq!(framing_count, 4, "three parts plus the closing delimiter");

        let longer = "a-very-long-upstream-model-identifier";
        let rewritten = rewrite_field(&original, CT, "model", longer).expect("longer rewrite");
        assert_eq!(
            extract_field(&rewritten, CT, "model").as_deref(),
            Some(longer)
        );
        assert_eq!(
            extract_field(&rewritten, CT, "tail").as_deref(),
            Some("tail-value")
        );
        let binary = b"raw\x00\xff\xfebytes";
        assert!(
            rewritten
                .windows(binary.len())
                .any(|window| window == binary),
            "binary payload must survive a longer model rewrite"
        );
        assert!(rewritten.ends_with(b"--xyz--\r\n"));
        assert_eq!(
            rewritten
                .windows(b"--xyz".len())
                .filter(|window| *window == b"--xyz")
                .count(),
            framing_count
        );

        let rewritten = rewrite_field(&rewritten, CT, "model", "s").expect("shorter rewrite");
        assert_eq!(extract_field(&rewritten, CT, "model").as_deref(), Some("s"));
        assert_eq!(
            extract_field(&rewritten, CT, "tail").as_deref(),
            Some("tail-value")
        );
        assert!(
            rewritten
                .windows(binary.len())
                .any(|window| window == binary),
            "binary payload must survive a shorter model rewrite"
        );
        assert!(rewritten.ends_with(b"--xyz--\r\n"));
        assert_eq!(
            rewritten
                .windows(b"--xyz".len())
                .filter(|window| *window == b"--xyz")
                .count(),
            framing_count
        );
    }
}
