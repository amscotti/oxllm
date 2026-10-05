//! Bounded response decoding shared by every remote guardrail.

use futures_util::StreamExt;
use serde::de::DeserializeOwned;

/// Verdicts are normally a few KiB. Allow room for Presidio/AIM redactions,
/// while bounding both declared-length and chunked bodies before JSON parsing.
pub(crate) const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

pub(super) async fn read_json<T: DeserializeOwned>(
    response: reqwest::Response,
) -> anyhow::Result<T> {
    anyhow::ensure!(
        response.status().is_success(),
        "guardrail returned HTTP {}",
        response.status()
    );
    anyhow::ensure!(
        response
            .content_length()
            .is_none_or(|len| len <= MAX_RESPONSE_BYTES as u64),
        "guardrail response exceeds {MAX_RESPONSE_BYTES} bytes"
    );
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        anyhow::ensure!(
            chunk.len() <= MAX_RESPONSE_BYTES.saturating_sub(body.len()),
            "guardrail response exceeds {MAX_RESPONSE_BYTES} bytes"
        );
        body.extend_from_slice(&chunk);
    }
    Ok(serde_json::from_slice(&body)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn decodes_verdict_and_rejects_declared_oversize() {
        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.path("/ok");
            then.status(200).json_body(json!({"blocked": false}));
        });
        server.mock(|when, then| {
            when.path("/large");
            then.status(200).body(" ".repeat(MAX_RESPONSE_BYTES + 1));
        });
        let client = reqwest::Client::new();
        let response = client.get(server.url("/ok")).send().await.unwrap();
        assert_eq!(
            read_json::<Value>(response).await.unwrap(),
            json!({"blocked": false})
        );
        let response = client.get(server.url("/large")).send().await.unwrap();
        assert!(
            read_json::<Value>(response)
                .await
                .unwrap_err()
                .to_string()
                .contains("exceeds")
        );
    }

    #[tokio::test]
    async fn rejects_chunked_oversize_before_waiting_for_eof() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            let read = socket.read(&mut request).await.unwrap();
            assert!(read > 0, "client request should arrive");
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                .await
                .unwrap();
            // Multiple chunks cross the cumulative cap. Never send the final
            // chunk: a reader that waits for EOF instead of enforcing the cap hangs.
            let chunk = vec![b' '; MAX_RESPONSE_BYTES / 2];
            for _ in 0..3 {
                socket
                    .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                    .await
                    .unwrap();
                socket.write_all(&chunk).await.unwrap();
                socket.write_all(b"\r\n").await.unwrap();
            }
            std::future::pending::<()>().await;
        });
        let response = reqwest::Client::new()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            read_json::<Value>(response),
        )
        .await
        .expect("oversize must be rejected before EOF");
        server.abort();
        assert!(result.unwrap_err().to_string().contains("exceeds"));
    }

    #[tokio::test]
    async fn built_in_remote_scanners_use_the_shared_body_limit() {
        use crate::config::types::{AimConfig, PresidioConfig};
        use crate::guardrails::{
            AimGuardrail, LakeraGuardrail, ModelArmorGuardrail, PresidioGuardrail,
        };
        fn oversized<T>(result: anyhow::Result<T>) {
            assert!(
                result
                    .err()
                    .expect("oversized response must fail")
                    .to_string()
                    .contains("exceeds")
            );
        }
        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.path("/analyze");
            then.status(200).json_body(json!([{
                "entity_type":"x".repeat(MAX_RESPONSE_BYTES), "start":0,"end":1,"score":1.0
            }]));
        });
        server.mock(|_when, then| {
            // Valid response shapes for every scanner. Without the shared cap,
            // each deserializer succeeds instead of returning a size error.
            then.status(200).json_body(json!({
                "required_action":{"action_type":"monitor_action"},
                "flagged":false,"blocked":false,"risk_score":0.0,
                "text":"x".repeat(MAX_RESPONSE_BYTES)
            }));
        });
        let aim: AimConfig =
            serde_json::from_value(json!({"url":server.base_url(),"api_key":"test"})).unwrap();
        let presidio: PresidioConfig =
            serde_json::from_value(json!({"url":server.base_url()})).unwrap();
        let messages = vec![json!({"role":"user","content":"hello"})];
        oversized(AimGuardrail::new(&aim, true).unwrap().scan(&messages).await);
        oversized(
            LakeraGuardrail::new("test".into(), Some(server.base_url()), None, true, false)
                .unwrap()
                .scan(&messages, None)
                .await,
        );
        oversized(
            ModelArmorGuardrail::new(server.url("/armor"), None, None, None, true)
                .unwrap()
                .scan(&messages)
                .await,
        );
        let presidio = PresidioGuardrail::new(&presidio, true).unwrap();
        oversized(presidio.analyze("hello").await);
        oversized(presidio.anonymize("hello", &[]).await);

        let custom = crate::guardrails::custom::HttpCustomGuardrail::new(
            "custom",
            server.url("/custom"),
            None,
            vec![],
            "blocked".into(),
            json!(true),
            None,
            false,
            true,
        )
        .unwrap();
        let mut body = json!({"messages":messages});
        assert!(matches!(custom.apply(&mut body).await,
            crate::guardrails::GuardrailResult::Block(reason) if reason.contains("unavailable")));
    }
}
