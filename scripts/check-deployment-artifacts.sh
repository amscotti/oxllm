#!/usr/bin/env bash
# Nonbuilding checks: Helm render semantics, Compose interpolation, YAML examples,
# toolchain consistency and Docker input allow-list. Never loads the local .env.
# Requires Python 3.11+ / PyYAML. --require-tools also requires Helm + Compose.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python3 - "$@" <<'PY'
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

import yaml

if sys.argv[1:] not in ([], ["--require-tools"]):
    sys.exit("usage: check-deployment-artifacts.sh [--require-tools]")
required = "--require-tools" in sys.argv
checks = 0


def check(description, condition):
    global checks
    if not condition:
        sys.exit(f"FAIL: {description}")
    checks += 1
    print(f"PASS: {description}")


def run(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, **kwargs)


def document(docs, kind, name="oxllm"):
    return next(d for d in docs if d["kind"] == kind and d["metadata"]["name"] == name)


def pod(docs, name="oxllm"):
    return document(docs, "Deployment", name)["spec"]["template"]


def environment(container):
    return {entry["name"]: entry for entry in container.get("env", [])}


base = {"env": {"OPENAI_API_KEY": "validation-provider", "OXLLM_MASTER_KEY": "validation-master"}}


def render(overlay=None, error=None):
    values = dict(base)
    values.update(overlay or {})
    result = run(["helm", "template", "oxllm", "helm/oxllm", "-f", "-"], input=yaml.safe_dump(values))
    if error:
        check(f"render rejects {error}", result.returncode != 0 and error in result.stderr)
        return
    if result.returncode:
        sys.exit(result.stderr)
    return [d for d in yaml.safe_load_all(result.stdout) if d]


if shutil.which("helm"):
    result = run(["helm", "lint", "helm/oxllm", "-f", "-"], input=yaml.safe_dump(base))
    check("helm lint", result.returncode == 0)
    default = render()
    main = pod(default)
    spec = main["spec"]
    check("single replica by default", document(default, "Deployment")["spec"]["replicas"] == 1)
    check("local image uses AppVersion", spec["containers"][0]["image"] == "oxllm:" + yaml.safe_load(Path("helm/oxllm/Chart.yaml").read_text())["appVersion"])
    check("default 600s drain gets 630s termination grace", spec["terminationGracePeriodSeconds"] == 630)
    check("no migration or DB env by default", not spec.get("initContainers") and "OXLLM_DATABASE_URL" not in environment(spec["containers"][0]))
    check("config mounted as directory, with explicit serve path", all("subPath" not in v for v in spec["containers"][0]["volumeMounts"]) and "/etc/oxllm/config.yaml" in spec["containers"][0]["args"])
    check("auth requirement defaults true", environment(spec["containers"][0])["OXLLM_REQUIRE_AUTH"]["value"] == "true")
    no_auth = render({"requireAuth": False, "serviceAccount": {"create": False}})
    check("requireAuth=false is honored", environment(pod(no_auth)["spec"]["containers"][0])["OXLLM_REQUIRE_AUTH"]["value"] == "false")
    check("default SA fallback still disables token automount", "serviceAccountName" not in pod(no_auth)["spec"] and pod(no_auth)["spec"]["automountServiceAccountToken"] is False)

    persistent = render({"persistence": {"enabled": True, "databaseUrl": "postgres://u:p@db/oxllm", "databaseUrlSecretKey": "custom-db-key"}})
    ps = pod(persistent)["spec"]
    config = yaml.safe_load(document(persistent, "ConfigMap", "oxllm-config")["data"]["config.yaml"])
    check("serve config actually references database env", config["general_settings"]["database_url"] == "os.environ/OXLLM_DATABASE_URL")
    init_env = environment(ps["initContainers"][0])["OXLLM_DATABASE_URL"]
    main_env = environment(ps["containers"][0])
    check("migrate and serve use same database Secret/key", init_env == main_env["OXLLM_DATABASE_URL"] and init_env["valueFrom"]["secretKeyRef"]["key"] == "custom-db-key")
    check("custom DB Secret key is rendered", "custom-db-key" in document(persistent, "Secret", "oxllm-secrets")["stringData"])
    check("serve disables repeated migrations", main_env["OXLLM_AUTO_MIGRATE"]["value"] == "false")
    external = render({"persistence": {"enabled": True, "existingSecret": "external-db", "migrate": False}})
    check("external migration mode has no init container", not pod(external)["spec"].get("initContainers"))
    check("external database Secret reaches serve", environment(pod(external)["spec"]["containers"][0])["OXLLM_DATABASE_URL"]["valueFrom"]["secretKeyRef"]["name"] == "external-db")
    render({"persistence": {"enabled": True}}, error="persistence.databaseUrl")
    render({"persistence": {"enabled": True, "databaseUrl": "sqlite:///data/test.db"}}, error="external Postgres")
    render({"replicas": 2}, error="replicas > 1")
    render({"terminationGracePeriodSeconds": 30}, error="drain budget")
    short = render({"config": {"oxllm_settings": {"request_timeout": 120}}})
    long = render({"config": {"general_settings": {"shutdown_timeout_secs": 900}}})
    check("grace follows request and shutdown overrides", pod(short)["spec"]["terminationGracePeriodSeconds"] == 150 and pod(long)["spec"]["terminationGracePeriodSeconds"] == 930)

    other_provider = render({"config": {"model_list": [{"model_name": "claude", "oxllm_params": {"model": "anthropic/claude", "api_key": "os.environ/ANTHROPIC_API_KEY"}}]}, "env": {"ANTHROPIC_API_KEY": "validation", "OXLLM_MASTER_KEY": "validation-master", "RUST_LOG": "info"}})
    data = document(other_provider, "Secret", "oxllm-env")["stringData"]
    check("generic env works without an OpenAI key", data["RUST_LOG"] == "info" and "ANTHROPIC_API_KEY" in data and "OPENAI_API_KEY" not in data)
    zero_value = render({"env": {**base["env"], "OPENAI_API_KEY": 0}})
    check("generic env stringifies zero instead of treating it as absent", document(zero_value, "Secret", "oxllm-env")["stringData"]["OPENAI_API_KEY"] == "0")
    literal = render({"config": {"model_list": [{"model_name": "chat-os.environ/LABEL", "oxllm_params": {"model": "openai/gpt-4o-mini", "api_key": "os.environ/OPENAI_API_KEY"}}]}})
    check("env validation matches scalar prefixes, not embedded literal text", "chat-os.environ/LABEL" in document(literal, "ConfigMap", "oxllm-config")["data"]["config.yaml"])
    refs = render({"env": {}, "envSecretRefs": {"OPENAI_API_KEY": {"name": "credentials", "key": "provider"}, "OXLLM_MASTER_KEY": {"name": "credentials", "key": "master"}}})
    check("external provider and master secrets supported", environment(pod(refs)["spec"]["containers"][0])["OPENAI_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "credentials")
    render({"env": {}}, error="config references")
    render({"env": {**base["env"], "OXLLM_MASTER_KEY": "   "}}, error="config references")
    render({"env": {**base["env"], "OXLLM_PORT": "4001"}}, error="managed by chart")

    full = render({"replicas": 3, "config": {"oxllm_settings": {"cache": {"type": "redis", "host": "redis"}}, "general_settings": {"allow_private_api_base": True}}, "persistence": {"enabled": True, "existingSecret": "external-db"}, "admin": {"enabled": True, "sessionSecret": "validation-session-secret", "uiPassword": "validation-password"}, "networkPolicy": {"enabled": True}, "port": 4001, "service": {"port": 4111}})
    for name in ("oxllm", "oxllm-admin"):
        spec = pod(full, name)["spec"]
        check(f"{name} restricted pod security", spec["automountServiceAccountToken"] is False and spec["securityContext"]["runAsNonRoot"] is True and spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault")
        for container in spec["containers"] + spec.get("initContainers", []):
            security = container["securityContext"]
            check(f"{name}/{container['name']} restricted container security", security["allowPrivilegeEscalation"] is False and security["capabilities"]["drop"] == ["ALL"] and security["readOnlyRootFilesystem"] is True)
    check("admin proxy URL uses Service port", environment(pod(full, "oxllm-admin")["spec"]["containers"][0])["OXLLM_PROXY_URL"]["value"] == "http://oxllm:4111")
    container = pod(full)["spec"]["containers"][0]
    check("port override preserves named probes", container["ports"][0]["containerPort"] == 4001 and container["livenessProbe"]["httpGet"]["port"] == container["readinessProbe"]["httpGet"]["port"] == "http")
    check("PDB scoped by instance", document(full, "PodDisruptionBudget")["spec"]["selector"]["matchLabels"]["app.kubernetes.io/instance"] == "oxllm")
    policy = document(full, "NetworkPolicy")["spec"]
    check("egress rules constrain ports", all(rule.get("ports") for rule in policy["egress"]))
    for rule in policy["egress"]:
        for target in rule.get("to", []):
            if "ipBlock" in target:
                block = target["ipBlock"]
                parent = ipaddress.ip_network(block["cidr"])
                check(f"NetworkPolicy exceptions belong to {parent}", all(ipaddress.ip_network(exc).subnet_of(parent) for exc in block.get("except", [])))
    check("admin ingress port is separate", document(full, "NetworkPolicy", "oxllm-admin")["spec"]["ingress"][0]["ports"][0]["port"] == 8080)
    render({"admin": {"enabled": True, "sessionSecret": "short", "uiPassword": "validation"}}, error="at least 16 characters")
    admin_external = render({"admin": {"enabled": True, "sessionSecretRef": {"name": "admin-auth", "key": "session"}, "passwordSecret": "admin-password"}})
    check("admin supports external session/password secrets", environment(pod(admin_external, "oxllm-admin")["spec"]["containers"][0])["ADMIN_UI_PASSWORD"]["valueFrom"]["secretKeyRef"]["name"] == "admin-password")
    env_from = render({"env": {}, "envFrom": [{"secretRef": {"name": "external-environment"}}], "admin": {"enabled": True, "sessionSecretRef": {"name": "admin-auth", "key": "session"}, "passwordSecret": "admin-password"}})
    check("external envFrom reaches proxy and admin", all(pod(env_from, name)["spec"]["containers"][0]["envFrom"] == [{"secretRef": {"name": "external-environment"}}] for name in ("oxllm", "oxllm-admin")))
    changed_key = render({"env": {**base["env"], "OXLLM_MASTER_KEY": "rotated-validation-master"}})
    check("managed config and secret edits change rollout checksums", main["metadata"]["annotations"]["checksum/config"] != pod(short)["metadata"]["annotations"]["checksum/config"] and main["metadata"]["annotations"]["checksum/env"] != pod(changed_key)["metadata"]["annotations"]["checksum/env"])
    check("external Secret rotation supports explicit rollout revision", pod(render({"rolloutRevision": "rotation-2"}))["metadata"]["annotations"]["oxllm/rollout-revision"] == "rotation-2")
else:
    check("Helm available", not required)
    print("SKIP: Helm render checks (helm unavailable)")

clean_env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
compose_available = shutil.which("docker") and run(["docker", "compose", "version"], env=clean_env).returncode == 0
if compose_available:
    def compose(files, extra=None):
        cmd = ["docker", "compose", "--env-file", "/dev/null"]
        for file in files:
            cmd.extend(["-f", file])
        cmd.extend(["--profile", "admin", "config", "--format", "json"])
        return run(cmd, env={**clean_env, **(extra or {})})

    env = {"OXLLM_MASTER_KEY": "validation-master"}
    base_files = ["docker-compose.yml"]
    result = compose(base_files, env)
    check("base Compose parses with no admin secrets", result.returncode == 0)
    services = json.loads(result.stdout)["services"]
    check("base Compose keeps Redis and writable SQLite volume", "redis" in services and any(v.get("target") == "/data" for v in services["oxllm"]["volumes"]) and services["oxllm"]["environment"]["DATABASE_URL"].startswith("sqlite:"))
    check("Redis eviction policy preserves enforcement state", "noeviction" in services["redis"]["command"])
    check("base Compose retains master-key constraint", compose(base_files).returncode != 0)
    admin_files = base_files + ["docker-compose.admin.yml"]
    check("admin overlay rejects missing credentials", compose(admin_files, env).returncode != 0)
    result = compose(admin_files, {**env, "ADMIN_SESSION_SECRET": "validation-session-secret", "ADMIN_UI_PASSWORD": "validation-password"})
    check("admin overlay parses with credentials", result.returncode == 0)
    admin = json.loads(result.stdout)["services"]["oxllm-admin"]
    check("admin waits for proxy health and restarts", admin["depends_on"]["oxllm"]["condition"] == "service_healthy" and admin["restart"] == "unless-stopped")
    result = compose(admin_files + ["docker-compose.ci.yml"], {**env, "ADMIN_SESSION_SECRET": "validation-session-secret", "ADMIN_UI_PASSWORD": "validation-password"})
    check("runtime CI Compose overlay uses locally built images", result.returncode == 0 and json.loads(result.stdout)["services"]["oxllm"]["image"] == "oxllm:ci" and json.loads(result.stdout)["services"]["oxllm-admin"]["image"] == "oxllm-admin:ci")
    check("review stack parses without local .env", compose(["docker-compose.review.yml"], {**env, "OPENAI_API_KEY": "validation", "ANTHROPIC_API_KEY": "validation", "XAI_API_KEY": "validation"}).returncode == 0)
else:
    check("Docker Compose available", not required)
    print("SKIP: Compose checks (plugin unavailable)")

class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
for index, snippet in enumerate(re.findall(r"```yaml\n(.*?)```", Path("README.md").read_text(), re.S), 1):
    data = yaml.load(snippet, Loader=UniqueLoader)
    if isinstance(data, dict) and "model_list" in data:
        check(f"README YAML #{index} has correctly nested deployments", all("oxllm_params" in row and "model" in row["oxllm_params"] for row in data["model_list"]))
check("README YAML fences parse without duplicate keys", True)
for file in Path("examples").glob("*.yaml"):
    yaml.load(file.read_text(), Loader=UniqueLoader)
check("example YAML parses without duplicate keys", True)

channel = tomllib.loads(Path("rust-toolchain.toml").read_text())["toolchain"]["channel"]
workflows = {}
for file in Path(".github/workflows").glob("*.yml"):
    workflow = yaml.load(file.read_text(), Loader=yaml.BaseLoader)
    workflows[file.stem] = workflow
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "run" in step:
                result = run(["bash", "-n"], input=step["run"])
                check(f"{file.name}: shell syntax for {step.get('name', 'run step')}", result.returncode == 0)
            if "uses" in step and not step["uses"].startswith("./"):
                check(f"{file.name}: immutable action reference", bool(re.search(r"@[0-9a-f]{40}$", step["uses"])))
ci = workflows["ci"]
check("CI primary toolchain matches rust-toolchain", ci["jobs"]["check"]["strategy"]["matrix"]["toolchain"] == [channel])
for job in ci["jobs"].values():
    for step in job.get("steps", []):
        toolchain = step.get("with", {}).get("toolchain")
        if toolchain and not toolchain.startswith("${{"):
            check("CI fixed toolchain matches source pin", toolchain == channel)
check("CI covers main and master", set(ci["on"]["push"]["branches"]) == {"main", "master"})
check("Redis CI enables gated integration tests", ci["jobs"]["redis-integration"]["env"]["OXLLM_REDIS_URL"].startswith("redis://127.0.0.1:"))
check("Postgres CI enables gated integration tests", "OXLLM_TEST_POSTGRES_URL" in ci["jobs"]["db-postgres"]["env"])
check("release depends on Rust and packaged runtime checks", set(workflows["release"]["jobs"]["images"]["needs"]) == {"rust-checks", "deployment-checks"})
check("reusable checks have distinct concurrency groups", ci["concurrency"]["group"] != workflows["deployment"]["concurrency"]["group"])
for file in ("Dockerfile", "Dockerfile.admin"):
    text = Path(file).read_text()
    check(f"{file} builder matches rust-toolchain and uses --locked", f"FROM rust:{channel}-bookworm" in text and "cargo build --locked" in text)
    check(f"{file} health probe present", "HEALTHCHECK" in text)
patterns = [line.strip() for line in Path(".dockerignore").read_text().splitlines() if line.strip() and not line.startswith("#")]
check("Docker context starts with deny-all and only explicit build inputs", patterns[0] == "*" and all(not line.startswith("!") or line in {"!rust-toolchain.toml", "!Cargo.toml", "!Cargo.lock", "!crates", "!crates/**", "!src", "!src/**", "!examples", "!examples/config.docker.yaml"} for line in patterns))
print(f"All {checks} nonbuilding deployment-artifact checks passed.")
PY
