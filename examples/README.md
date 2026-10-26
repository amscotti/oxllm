# Examples

Sample and test configurations for oxllm. All secrets are read from the
environment (`.env`, `os.environ/…` — see `.env.example` at the repo root).

| File | Purpose |
| --- | --- |
| `config.minimal.yaml` | Smallest runnable quickstart (see the root README). |
| `config.example.yaml` | Fully annotated reference config; not a ready-to-run quickstart. |
| `config.docker.yaml` | Config baked into the Docker image. |
| `config.local-redis.yaml` | Local dev stack with Redis. |
| `config.local-ui.yaml` | Local dev with the admin UI (see `docs/admin-production.md`). |
| `config.multi-replica.yaml` | Two-replica load-balancing example with shared Redis, used by `docker-compose.multi-replica.yml`. |
| `config.openrouter-lb.yaml` | OpenRouter load-balancing plus embeddings/images smoke. |
| `config.foundry.yaml` | Azure AI Foundry examples, including native Claude and Cohere image embeddings. |
| `config.anthropic-wif.yaml` | Anthropic workload identity federation example. |
| `config.provider-smoke.yaml` | Multi-provider live smoke config. |
| `config.e2e.yaml` | End-to-end live testing across providers (OpenAI/Anthropic/Bedrock/…). |
| `config.realtest.yaml` | Live provider round-trip tests, including multi-deployment LB. |
| `config.anthropic-test.yaml` | Anthropic-only live tests (Claude models). |
| `config.bedrock-test.yaml` | AWS Bedrock / Mantle live tests. |

The `*-test`, `*-smoke`, `e2e`, and `realtest` configs are primarily used by
the live test scripts and require the matching provider credentials.
