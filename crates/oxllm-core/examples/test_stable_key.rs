use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::router::stores::stable_key;

fn make_deployment(i: usize) -> Deployment {
    Deployment {
        model_name: "gpt-4o".to_string(),
        oxllm_params: OxllmParams {
            model: format!("openai/gpt-4o-{i}"),
            api_key: Some(format!("key-{i}")),
            rpm: Some(480),
            tpm: Some(1_000_000),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn main() {
    for i in 0..3 {
        let d = make_deployment(i);
        let key = stable_key(&d);
        let bench_key = format!("gpt-4o:{i}");
        println!(
            "deployment {i}: stable_key = {:?}, bench_key = {:?}",
            key, bench_key
        );
        println!("  match: {}", key == bench_key);
    }
}
