#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // The global tracing subscriber is installed inside `oxllm_admin::run`
    // (after CLI parsing, via `try_init`) — NOT unconditionally here. This
    // mirrors `src/main.rs`: `set_global_default` is one-shot, so burning it
    // in the binary preamble would prevent any embedder (or a future
    // config-driven / OTel-composed subscriber) from installing its own.
    oxllm_admin::run().await
}
