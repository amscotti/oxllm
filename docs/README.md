# oxllm documentation

Guides for people and agents working on or extending oxllm.

| Doc | Audience |
|-----|----------|
| [Architecture](architecture.md) | Anyone changing code — crates, request path, layering rules |
| [Adding a feature](adding-a-feature.md) | Where a change belongs + steps per feature kind |
| [Adding a provider](adding-a-provider.md) | Adding a new upstream LLM provider |
| [Operations](operations.md) | Deployment, reload/restart rules, backups, migrations and cache privacy |
| [Helm chart](../helm/oxllm/README.md) | Images, secrets, Postgres, Redis and rollout settings |
| [Admin UI operations](admin-production.md) | Running oxllm-admin; original planning notes retained below |
| [Troubleshooting](troubleshooting.md) | Startup, auth, backend health and streaming diagnostics |
| [Examples](../examples/) | Sample YAML configs (Docker, Redis, multi-provider LB) |

Start with **architecture** if you are new to the repo.
