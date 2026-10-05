# Examples

Sample and test configurations for oxllm. All secrets are read from the
environment (`.env`, `os.environ/…` — see `.env.example` at the repo root).

| File | Purpose |
| --- | --- |
| `config.example.yaml` | Fully annotated reference config (see README). |
| `config.docker.yaml` | Config baked into the Docker image. |
| `config.local-redis.yaml` | Local dev stack with Redis (used by `scripts/smoke-local.sh`). |
| `config.local-ui.yaml` | Local dev with the admin UI (see `docs/admin-production.md`). |
| `config.openrouter-lb.yaml` | OpenRouter load-balancing + embeddings/images smoke (used by `scripts/test-openrouter-lb.sh`). |
| `config.provider-smoke.yaml` | Multi-provider live smoke config. |
| `config.e2e.yaml` | End-to-end live testing across providers (OpenAI/Anthropic/Bedrock/…). |
| `config.realtest.yaml` | Live "real" provider round-trip tests, incl. multi-deploy LB. |
| `config.anthropic-test.yaml` | Anthropic-only live tests (Claude models). |
| `config.bedrock-test.yaml` | AWS Bedrock / Mantle live tests. |
| `config.review-live.yaml` | Multi-deploy load-balancing example: one model name routed across OpenAI/Anthropic/xAI. Used by `docker-compose.review.yml`, which runs two replicas sharing the Redis-backed cache configured here. |
