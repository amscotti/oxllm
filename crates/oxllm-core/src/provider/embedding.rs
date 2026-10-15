use serde_json::Value;

/// A provider-native embedding request, before host body/header overrides.
pub struct EmbeddingRequest {
    /// Upstream URL, including any operation-specific API version.
    pub url: String,
    /// Native JSON body. Mapped inputs can be moved here without copying.
    pub body: Value,
    /// Source parameters for explicitly allowed extras. Mapped inputs may be
    /// absent; the host must not restore mapped fields from this source.
    pub params: Value,
    /// Optional native top-level allow-list, enforced after host overrides.
    /// `None` retains the ordinary provider/host forwarding policy.
    pub allowed_fields: Option<&'static [&'static str]>,
}

/// One mapped part of a logical embedding request.
pub struct EmbeddingRequestBatch {
    /// Prepared native request for this non-empty batch.
    pub upstream: EmbeddingRequest,
    /// Map each upstream response index to its original input index.
    pub input_indices: Vec<usize>,
}

/// Distinguish an unchanged single response from a mapped batch response.
pub enum EmbeddingRequestPlan {
    /// Preserve the ordinary single-request response schema verbatim.
    Single(EmbeddingRequest),
    /// Validate and recombine these requests in original input order.
    Batched(Vec<EmbeddingRequestBatch>),
}

/// Test for a case-sensitive `data:` prefix with canonical standard base64
/// comprising the entire payload after the first comma. No allocation or
/// decoding is needed. Media types and image content remain upstream-owned.
pub fn is_base64_data_uri(value: &str) -> bool {
    let Some((_, payload)) = value
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
    else {
        return false;
    };
    let bytes = payload.as_bytes();
    if bytes.len() % 4 != 0 {
        return false;
    }
    let padding = bytes.iter().rev().take_while(|&&byte| byte == b'=').count();
    if padding > 2 {
        return false;
    }
    let data = &bytes[..bytes.len() - padding];
    fn sextet(byte: u8) -> Option<u8> {
        match byte {
            b'A'..=b'Z' => Some(byte - b'A'),
            b'a'..=b'z' => Some(byte - b'a' + 26),
            b'0'..=b'9' => Some(byte - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    if !data.iter().all(|&byte| sextet(byte).is_some()) {
        return false;
    }
    match padding {
        0 => true,
        1 => data
            .last()
            .and_then(|&byte| sextet(byte))
            .is_some_and(|last| last & 3 == 0),
        2 => data
            .last()
            .and_then(|&byte| sextet(byte))
            .is_some_and(|last| last & 15 == 0),
        _ => false,
    }
}
