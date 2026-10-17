//! Native Anthropic service tiers survive host stripping and API bridges.

mod common;

use httpmock::{Method::POST, MockServer};
use serde_json::{Value, json};

#[tokio::test]
async fn anthropic_tiers_are_value_aware_across_drop_policies_and_bridges() {
    for provider in ["anthropic", "azure_ai"] {
        for drop_params in [false, true] {
            let upstream = MockServer::start_async().await;
            let mut config = common::proxy_config(&upstream.url("/anthropic/v1"));
            config.model_list[0].model_name = "model".into();
            config.model_list[0].oxllm_params.model = format!("{provider}/claude-sonnet-4-5");
            config.oxllm_settings.drop_params = Some(drop_params);
            config.router_settings.routing.num_retries = Some(0);
            config.router_settings.routing.cooldown_time = Some(0);
            let (handle, _) = common::start_oxllm(config).await.unwrap();
            let client = common::http_client();
            for (tier, expected) in [
                ("default", Some("standard_only")),
                ("auto", Some("auto")),
                ("standard_only", Some("standard_only")),
                ("flex", None),
            ] {
                let mut wire = json!({"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"Hello"}],"max_tokens":64});
                if let Some(tier) = expected {
                    wire["service_tier"] = json!(tier);
                }
                let mock = upstream.mock_async(|when, then| {
                    when.method(POST).path("/anthropic/v1/messages").json_body(wire);
                    then.status(200).json_body(json!({"id":"msg-tier","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}));
                }).await;
                for api in ["/v1/chat/completions", "/v1/messages", "/v1/responses"] {
                    let mut payload = if api == "/v1/responses" {
                        json!({"model":"model","input":"Hello","max_output_tokens":64})
                    } else {
                        json!({"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64})
                    };
                    payload["service_tier"] = json!(tier);
                    let response = client
                        .post(handle.url(api))
                        .json(&payload)
                        .send()
                        .await
                        .unwrap();
                    let status = response.status();
                    let body: Value = response.json().await.unwrap();
                    assert_eq!(
                        status.as_u16(),
                        if expected.is_none() && !drop_params {
                            400
                        } else {
                            200
                        },
                        "{provider} drop={drop_params} {api} {tier}: {body}"
                    );
                    if status == 400 {
                        assert!(
                            body["error"]["message"]
                                .as_str()
                                .unwrap()
                                .contains("service_tier")
                        );
                    }
                }
                mock.assert_hits_async(if expected.is_none() && !drop_params {
                    0
                } else {
                    3
                })
                .await;
                mock.delete_async().await;
            }
            handle.shutdown();
        }
    }
}
