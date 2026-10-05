//! Shared Server-Sent Events framing utilities.

/// An incremental UTF-8 and SSE-event decoder.
///
/// HTTP chunk boundaries can split a multibyte UTF-8 character. Decoding each
/// network chunk independently would inject replacement characters into the
/// event stream. This accumulator keeps trailing incomplete UTF-8 bytes until
/// the next chunk arrives. Event boundaries accept both LF and CRLF framing.
#[derive(Debug, Default)]
pub(crate) struct SseFramer {
    pending_utf8: Vec<u8>,
}

impl SseFramer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append a network chunk and return everything decoded so far.
    ///
    /// Valid UTF-8 is emitted directly. A hard-invalid byte sequence (one that
    /// can never start a valid character, e.g. `0xff`) is replaced with
    /// `U+FFFD` and consumed so it cannot stall the stream forever. Only a
    /// genuinely incomplete trailing multibyte sequence is retained for the
    /// next chunk.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> String {
        self.pending_utf8.extend_from_slice(bytes);
        let mut decoded = String::new();
        // Advance `pos` instead of draining the front of `pending_utf8` per
        // byte. Draining a single byte from the head of a `Vec` shifts every
        // remaining byte left, so a chunk with N invalid bytes (or N small
        // valid-then-invalid segments) would cost O(N^2); a cursor makes the
        // whole call linear and defers the single shift to the end.
        let mut pos = 0;
        loop {
            match std::str::from_utf8(&self.pending_utf8[pos..]) {
                Ok(s) => {
                    decoded.push_str(s);
                    self.pending_utf8.clear();
                    return decoded;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        // std validated this prefix for the same buffer, so it
                        // decodes by construction. `unwrap_or("")` preserves the
                        // data on the happy path and only degrades on a genuine
                        // bug; a `debug_assert!` + empty fallback would silently
                        // drop data in release builds if the invariant
                        // regressed. (`from_utf8_unchecked` would skip the
                        // re-validation but is forbidden under `forbid(unsafe_code)`.)
                        let valid = std::str::from_utf8(&self.pending_utf8[pos..pos + valid_up_to])
                            .unwrap_or("");
                        decoded.push_str(valid);
                        pos += valid_up_to;
                    }
                    match err.error_len() {
                        None => {
                            // Incomplete trailing sequence; retain [pos..] for
                            // the next chunk. One drain, not one per byte.
                            self.pending_utf8.drain(..pos);
                            return decoded;
                        }
                        Some(invalid_len) => {
                            // Hard-invalid bytes; replace with U+FFFD and skip.
                            decoded.push('\u{FFFD}');
                            pos += invalid_len;
                        }
                    }
                }
            }
        }
    }

    /// Decode any remaining bytes at end-of-stream. An invalid trailing
    /// sequence is a decode error rather than silently injecting U+FFFD.
    pub(crate) fn finish(&mut self) -> Result<String, std::string::FromUtf8Error> {
        String::from_utf8(std::mem::take(&mut self.pending_utf8))
    }
}

/// Extract the payload of the first `data:` line in `raw`.
///
/// Matches both `data:{...}` and `data: {...}` and strips at most one leading
/// U+0020 after the colon, per the SSE spec (HTML §9.2.6); further surrounding
/// whitespace is part of the value and is left intact. Returns `None` when no
/// `data:`/`data` line is present. Used by the completion/response/message
/// streaming handlers to pull the JSON body out of an SSE event frame.
///
/// Per HTML §9.2.6 a line with no U+003A colon is a field whose value is the
/// empty string, so a bare-word `data` line is recognised as a data field with
/// value `""`.
///
/// Lines are split on any of `\n`, `\r\n`, `\r` so that events framed with any
/// of the three SSE line-ending styles — including the lone-CR style that
/// `pop_sse_event` recognises — parse identically. (`str::lines` splits only on
/// `\n` and would silently drop multi-field lone-CR events.)
///
/// Only the **first** `data:` line is returned. The spec allows several
/// `data:` lines in one event block (concatenated with `\n`), but every
/// supported provider emits single-line JSON payloads, so joining is
/// unnecessary here and would force an allocation on the streaming hot path.
pub(crate) fn extract_sse_data(raw: &str) -> Option<&str> {
    for line in raw.split(['\n', '\r']) {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("data:") {
            return Some(rest.strip_prefix(' ').unwrap_or(rest));
        }
        if line == "data" {
            return Some("");
        }
    }
    None
}

/// Pop the next complete SSE event from `buffer`, recognising the three
/// **homogeneous** line-ending event separators the HTML SSE spec permits:
/// `\n\n`, `\r\n\r\n`, and lone-CR `\r\r`. Returns `None` when only a partial
/// event is present.
///
/// **Mixed-style separators** such as `\n\r\n` or `\r\n\n` are intentionally
/// NOT recognised: no shipped LLM provider emits mixed line endings within a
/// single separator, and detecting them would require a byte-by-byte scan
/// rather than three disjoint `find` calls. A mixed separator leaves the event
/// buffered until a later homogeneous separator arrives.
///
/// The EARLIEST boundary of any kind wins. Preferring one framing style over
/// an earlier boundary of another (e.g. via `or_else`) would let a later
/// separator swallow the event in front of it, merging two events into one for
/// buffers with mixed framing. The three patterns are byte-disjoint — no
/// `\n\n` or `\r\r` occurs inside `\r\n\r\n` — so the minimum index cannot
/// straddle two styles.
pub(crate) fn pop_sse_event(buffer: &mut String) -> Option<String> {
    let (index, delimiter_len) = [
        (buffer.find("\n\n"), 2),
        (buffer.find("\r\n\r\n"), 4),
        (buffer.find("\r\r"), 2),
    ]
    .into_iter()
    .filter_map(|(pos, len)| pos.map(|p| (p, len)))
    .min_by_key(|&(pos, _)| pos)?;
    let remainder = buffer.split_off(index + delimiter_len);
    Some(std::mem::replace(buffer, remainder))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_sse_data_handles_space_and_no_space() {
        assert_eq!(extract_sse_data("data: {\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(extract_sse_data("data:{\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(extract_sse_data("data: [DONE]"), Some("[DONE]"));
        assert_eq!(extract_sse_data("not data"), None);
        assert_eq!(
            extract_sse_data("event: chunk\ndata: {\"a\":1}\n\n"),
            Some("{\"a\":1}")
        );
    }

    #[test]
    fn pop_event_accepts_lf_and_crlf_boundaries() {
        let mut buffer = "data: one\n\ndata: two\r\n\r\ndata: three".to_string();
        assert_eq!(pop_sse_event(&mut buffer).as_deref(), Some("data: one\n\n"));
        assert_eq!(
            pop_sse_event(&mut buffer).as_deref(),
            Some("data: two\r\n\r\n")
        );
        assert!(pop_sse_event(&mut buffer).is_none());
        assert_eq!(buffer, "data: three");
    }

    #[test]
    fn pop_event_mixed_framing_earliest_boundary_wins() {
        // Regression: a buffer holding a CRLF-framed event followed by an
        // LF-framed event must pop the CRLF event first. Preferring `\n\n`
        // over the earlier `\r\n\r\n` would merge both events into one.
        let mut buffer = "data: one\r\n\r\ndata: two\n\n".to_string();
        assert_eq!(
            pop_sse_event(&mut buffer).as_deref(),
            Some("data: one\r\n\r\n"),
            "the earlier CRLF boundary must win over the later LF boundary"
        );
        assert_eq!(
            pop_sse_event(&mut buffer).as_deref(),
            Some("data: two\n\n"),
            "the LF-framed event must remain intact for the next pop"
        );
        assert!(pop_sse_event(&mut buffer).is_none());
        assert!(buffer.is_empty());

        // And the reverse order: an earlier LF boundary must win over a later
        // CRLF boundary.
        let mut buffer = "data: one\n\ndata: two\r\n\r\n".to_string();
        assert_eq!(pop_sse_event(&mut buffer).as_deref(), Some("data: one\n\n"));
        assert_eq!(
            pop_sse_event(&mut buffer).as_deref(),
            Some("data: two\r\n\r\n")
        );
        assert!(pop_sse_event(&mut buffer).is_none());
        assert!(buffer.is_empty());
    }

    #[test]
    fn framer_holds_split_multibyte_utf8() {
        let mut framer = SseFramer::new();
        let snowman = "☃".as_bytes();
        assert_eq!(framer.push(&snowman[..1]), "");
        assert_eq!(framer.push(&snowman[1..]), "☃");
        assert!(framer.finish().expect("finish").is_empty());
    }

    #[test]
    fn finish_errors_on_incomplete_trailing_utf8() {
        let mut framer = SseFramer::new();
        let snowman = "☃".as_bytes();
        // An incomplete multibyte sequence is held for the next chunk; if the
        // stream ends first, `finish` surfaces it as a decode error rather
        // than silently injecting U+FFFD.
        assert_eq!(framer.push(&snowman[..1]), "");
        assert!(framer.finish().is_err());
    }

    #[test]
    fn framer_skips_invalid_utf8_byte() {
        let mut framer = SseFramer::new();
        // `0xff` is a hard-invalid UTF-8 lead byte: it can never begin a valid
        // character. It must be consumed (replaced with U+FFFD) rather than
        // held forever, otherwise every subsequent chunk would decode to "".
        let first = framer.push(&[0xff]);
        assert!(
            first.contains('\u{FFFD}'),
            "hard-invalid byte must be replaced with U+FFFD, got {first:?}"
        );
        let second = framer.push(b"data: ok\n\n");
        assert!(
            second.contains("data: ok"),
            "valid data after the invalid byte must come through, got {second:?}"
        );
        assert!(framer.finish().expect("finish").is_empty());
    }

    #[test]
    fn framer_replaces_invalid_byte_between_valid_data() {
        let mut framer = SseFramer::new();
        // `0xc3 0x28` is an invalid 2-byte sequence; `0xc3` would start a
        // 2-byte char but `0x28` (`(`) is not a valid continuation byte.
        let out = framer.push(b"before\xffafter");
        assert_eq!(out, "before\u{FFFD}after");
        assert!(framer.finish().expect("finish").is_empty());
    }

    #[test]
    fn framer_handles_large_all_invalid_chunk() {
        // A pathological chunk of N hard-invalid bytes must decode in linear
        // time (previously O(N^2) via per-byte front-drains) and yield exactly
        // one U+FFFD per invalid byte. 40_000 keeps the unit test fast while
        // still exercising the many-invalid path the report flagged.
        let mut framer = SseFramer::new();
        let out = framer.push(&[0xff; 40_000]);
        let fffd_count = out.chars().filter(|&c| c == '\u{FFFD}').count();
        assert_eq!(fffd_count, 40_000);
        assert_eq!(out.chars().count(), 40_000);
        assert!(framer.finish().expect("finish").is_empty());
    }

    #[test]
    fn framer_replaces_multiple_invalid_bytes_among_valid_data() {
        let mut framer = SseFramer::new();
        // `0xff` and `0xfe` are both hard-invalid lead bytes (error_len ==
        // Some), surrounded by valid ASCII on both sides.
        let out = framer.push(b"ab\xff\xfecd");
        assert_eq!(out, "ab\u{FFFD}\u{FFFD}cd");
        assert!(framer.finish().expect("finish").is_empty());
    }

    #[test]
    fn framer_flushes_valid_prefix_before_incomplete_tail() {
        // Exercises the `valid_up_to > 0 && error_len == None` path: a valid
        // prefix is emitted immediately while the incomplete trailing byte is
        // held for the next chunk.
        let snowman = "☃".as_bytes();
        let mut framer = SseFramer::new();
        assert_eq!(framer.push(b"hi"), "hi");
        assert_eq!(framer.push(&snowman[..1]), "");
        // `hi` was already flushed; the lone `\xe2` is retained and completed
        // by the rest of the code point on the next chunk.
        assert_eq!(framer.push(&snowman[1..]), "☃");
        assert!(framer.finish().expect("finish").is_empty());
    }

    #[test]
    fn extract_sse_data_strips_at_most_one_leading_space() {
        // Per the SSE spec only a single leading U+0020 is stripped; the rest
        // of the value (including a second leading space) is preserved.
        assert_eq!(extract_sse_data("data:  two"), Some(" two"));
        assert_eq!(extract_sse_data("data:\t tab kept"), Some("\t tab kept"));
        assert_eq!(extract_sse_data("data: [DONE]"), Some("[DONE]"));
        assert_eq!(extract_sse_data("data:empty"), Some("empty"));
    }

    #[test]
    fn pop_event_accepts_lone_cr_boundary() {
        let mut buffer = "data: one\r\rdata: two\r\r".to_string();
        assert_eq!(pop_sse_event(&mut buffer).as_deref(), Some("data: one\r\r"));
        assert_eq!(pop_sse_event(&mut buffer).as_deref(), Some("data: two\r\r"));
        assert!(pop_sse_event(&mut buffer).is_none());
        assert!(buffer.is_empty());
    }

    #[test]
    fn pop_event_single_event_empties_buffer() {
        let mut buffer = "data: only\n\n".to_string();
        assert_eq!(
            pop_sse_event(&mut buffer).as_deref(),
            Some("data: only\n\n")
        );
        assert!(pop_sse_event(&mut buffer).is_none());
        assert!(buffer.is_empty());
    }

    #[test]
    fn extract_sse_data_handles_lone_cr_multifield() {
        // A lone-CR-framed event carrying more than one field must parse the
        // same as its LF/CRLF equivalents. `pop_sse_event` recognises `\r\r`
        // separators; `extract_sse_data` splits on any line ending so the
        // `data:` line is found regardless of framing style. Previously
        // `str::lines` (which splits only on `\n`) returned the whole block as
        // one line and the event was silently dropped.
        assert_eq!(
            extract_sse_data("event: chunk\rdata: {\"a\":1}\r\r"),
            Some("{\"a\":1}")
        );
        assert_eq!(
            extract_sse_data("event: chunk\r\ndata: {\"a\":1}\r\n\r\n"),
            Some("{\"a\":1}")
        );
        assert_eq!(
            extract_sse_data("event: chunk\ndata: {\"a\":1}\n\n"),
            Some("{\"a\":1}")
        );
    }

    #[test]
    fn pop_event_does_not_recognise_mixed_line_ending_separator() {
        // Documented scope: only homogeneous separators (`\n\n`, `\r\n\r\n`,
        // `\r\r`) are recognised. A mixed separator like `\n\r\n` is NOT a
        // boundary, so the event stays buffered until a later homogeneous
        // separator arrives. No shipped provider emits mixed line endings.
        let mut buffer = "data: x\n\r\n".to_string();
        assert!(pop_sse_event(&mut buffer).is_none());
        assert_eq!(buffer, "data: x\n\r\n");
    }

    #[test]
    fn extract_sse_data_bare_data_field() {
        // Per HTML §9.2.6 a line with no U+003A colon is a field whose value
        // is the empty string, so a bare-word `data` line is a data field with
        // value "".
        assert_eq!(extract_sse_data("data"), Some(""));
        assert_eq!(extract_sse_data("data\n"), Some(""));
        assert_eq!(extract_sse_data("event: x\ndata\n\n"), Some(""));
        // `data:` with colon and empty value is also the empty string.
        assert_eq!(extract_sse_data("data:"), Some(""));
        // A bare `datainfo:` line must NOT be mistaken for `data`.
        assert_eq!(extract_sse_data("datainfo: x"), None);
    }

    #[test]
    fn extract_sse_data_multiline_returns_first_only() {
        // The SSE spec concatenates multiple `data:` lines in one event block
        // with `\n`. This implementation intentionally returns only the FIRST
        // line: every supported provider emits single-line JSON payloads, and
        // joining would allocate on the streaming hot path. Pinned here so a
        // future regression (e.g. a provider emitting spec-conformant
        // multi-line data) surfaces as a test failure rather than silent data
        // loss.
        assert_eq!(
            extract_sse_data("data: {\"first\":1}\ndata: {\"second\":2}\n\n"),
            Some("{\"first\":1}")
        );
    }

    #[test]
    fn extract_sse_data_returns_none_when_no_data_field() {
        // An event block carrying only comment / event / id / retry lines (and
        // no data line) carries no payload and must return None.
        assert_eq!(extract_sse_data(": heartbeat\n"), None);
        assert_eq!(
            extract_sse_data("event: ping\nid: 42\nretry: 1000\n\n"),
            None
        );
    }

    #[test]
    fn multibyte_char_split_across_chunks_reassembles_into_one_event() {
        // Guards the core invariant the module exists to provide: a multibyte
        // UTF-8 character (`é` = 0xc3 0xa9) split by a TCP chunk boundary right
        // before an SSE event delimiter must be reassembled, and
        // `pop_sse_event` must fire exactly once for the complete event.
        let mut framer = SseFramer::new();
        // Chunk 1 ends with the lead byte 0xc3, which is incomplete and held.
        let out1 = framer.push(b"data: {\"t\":\"caf\xc3");
        assert_eq!(out1, "data: {\"t\":\"caf");
        // Chunk 2 begins with the continuation byte 0xa9, completing `é`, and
        // carries the closing quote and the event delimiter.
        let out2 = framer.push(b"\xa9\"}\n\n");
        assert_eq!(out2, "\u{e9}\"}\n\n");

        let mut buffer = format!("{out1}{out2}");
        assert_eq!(
            pop_sse_event(&mut buffer).as_deref(),
            Some("data: {\"t\":\"caf\u{e9}\"}\n\n"),
        );
        assert!(pop_sse_event(&mut buffer).is_none());
        assert!(buffer.is_empty());
    }

    #[test]
    fn four_byte_char_split_across_three_chunks_reassembles() {
        // Realistic TCP fragmentation for a 4-byte character (emoji): the
        // three continuation bytes each arrive in their own chunk. The framer
        // must hold the incomplete sequence across multiple pushes and emit
        // the character once complete.
        let mut framer = SseFramer::new();
        let emoji = "\u{1F600}".as_bytes(); // U+1F600, 4 bytes: f0 9f 98 80
        assert_eq!(emoji.len(), 4);
        assert_eq!(framer.push(&emoji[..1]), "");
        assert_eq!(framer.push(&emoji[1..2]), "");
        assert_eq!(framer.push(&emoji[2..3]), "");
        assert_eq!(framer.push(&emoji[3..]), "\u{1F600}");
        assert!(framer.finish().expect("finish").is_empty());
    }

    #[test]
    fn finish_returns_empty_tail_after_clean_push() {
        // Happy path: a fully-valid chunk flushes completely in `push`, so
        // `finish` returns an empty tail. Pins the clean `Ok` arm of `finish`
        // (other tests only exercise `push` output or the error tail).
        let mut framer = SseFramer::new();
        framer.push(b"data: ok\n\n");
        let tail = framer.finish().expect("clean tail decodes");
        assert!(tail.is_empty());
    }

    #[test]
    fn framer_invalid_continuation_after_lead_byte_replaced_then_held() {
        // Exercises the `error_len == Some(1)` "bad continuation after a lead
        // byte" path: a 2-byte lead (`0xc3`) followed by another lead byte
        // (`0xc3`) instead of a continuation byte. The first `0xc3` cannot be
        // completed, so it is replaced with U+FFFD and consumed (not held
        // forever); the second `0xc3` is then a fresh incomplete sequence that
        // is retained for the next chunk.
        let mut framer = SseFramer::new();
        assert_eq!(framer.push(&[0xc3]), "");
        assert_eq!(framer.push(&[0xc3]), "\u{FFFD}");
        // The second `0xc3` is held; completing it with a valid continuation
        // (`0xa9`) yields `é` (U+00E9 = 0xc3 0xa9).
        assert_eq!(framer.push(&[0xa9]), "\u{e9}");
        assert!(framer.finish().expect("finish").is_empty());
    }

    #[test]
    fn push_empty_slice_is_a_noop() {
        // An empty chunk must decode to the empty string and leave
        // `pending_utf8` untouched. Pins the no-op contract so a caller that
        // accidentally forwards a zero-length read does not perturb any held
        // partial sequence.
        let mut framer = SseFramer::new();
        assert_eq!(framer.push(&[]), "");
        assert!(framer.finish().expect("finish").is_empty());

        // Also a no-op while a partial sequence is mid-reassembly: the held
        // bytes must survive an empty push and still complete on the next
        // non-empty chunk.
        let snowman = "☃".as_bytes();
        let mut framer = SseFramer::new();
        assert_eq!(framer.push(&snowman[..1]), "");
        assert_eq!(framer.push(&[]), "");
        assert_eq!(framer.push(&snowman[1..]), "☃");
        assert!(framer.finish().expect("finish").is_empty());
    }

    #[test]
    fn pop_event_buffer_beginning_with_delimiter_pops_empty_then_real() {
        // A buffer that starts with the event separator yields an empty event
        // first, then the real event. `extract_sse_data` on the empty event
        // returns `None`, so it is harmlessly skipped downstream, but the
        // framing behaviour itself is pinned here.
        let mut buffer = "\n\ndata: x\n\n".to_string();
        assert_eq!(pop_sse_event(&mut buffer).as_deref(), Some("\n\n"));
        assert_eq!(extract_sse_data("\n\n"), None);
        assert_eq!(pop_sse_event(&mut buffer).as_deref(), Some("data: x\n\n"));
        assert!(pop_sse_event(&mut buffer).is_none());
        assert!(buffer.is_empty());
    }

    #[test]
    fn pop_event_lone_cr_and_crlf_candidates_coexist_earliest_wins() {
        // Extends `pop_event_mixed_framing_earliest_boundary_wins` (which mixes
        // LF and CRLF) to the case where lone-CR (`\r\r`) and CRLF (`\r\n\r\n`)
        // candidates coexist in the same buffer. The earlier boundary must win
        // regardless of style, so two adjacent events are never merged.
        let mut buffer = "data: one\r\rdata: two\r\n\r\n".to_string();
        assert_eq!(pop_sse_event(&mut buffer).as_deref(), Some("data: one\r\r"));
        assert_eq!(
            pop_sse_event(&mut buffer).as_deref(),
            Some("data: two\r\n\r\n")
        );
        assert!(pop_sse_event(&mut buffer).is_none());
        assert!(buffer.is_empty());

        // Reverse order: earlier CRLF boundary wins over the later lone-CR one.
        let mut buffer = "data: one\r\n\r\ndata: two\r\r".to_string();
        assert_eq!(
            pop_sse_event(&mut buffer).as_deref(),
            Some("data: one\r\n\r\n")
        );
        assert_eq!(pop_sse_event(&mut buffer).as_deref(), Some("data: two\r\r"));
        assert!(pop_sse_event(&mut buffer).is_none());
        assert!(buffer.is_empty());
    }
}
