# oxllm documentation

Guides for operators and contributors working on or extending oxllm.

| Doc | Audience |
|-----|----------|
| [Architecture](architecture.md) | Anyone changing code — crates, request path, layering rules |
| [Adding a feature](adding-a-feature.md) | Where a change belongs + steps per feature kind |
| [Adding a provider](adding-a-provider.md) | Adding a new upstream LLM provider |
| [Operations](operations.md) | Deployment, reload/restart rules, backups, migrations and cache privacy |
| [Helm chart](../helm/oxllm/README.md) | Images, secrets, Postgres, Redis and rollout settings |
| [Admin UI operations](admin-production.md) | Running oxllm-admin |
| [Troubleshooting](troubleshooting.md) | Startup, auth, backend health and streaming diagnostics |
| [Examples](../examples/) | Sample YAML configs (Docker, Redis, multi-provider LB) |
| [Adaptive router design](plans/adaptive-router-design.md) | Historical design rationale for the adaptive router |
| [Bedrock Converse design](plans/bedrock-converse-plan.md) | Historical design rationale for the dual InvokeModel/Converse path |

Start with **architecture** if you are new to the repo.
