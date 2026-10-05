#!/usr/bin/env bash
# Start proxy (:4000) + admin UI (:8080) for local development.
# Keeps processes alive until Ctrl-C (then kills both).
#
#   ./scripts/dev-stack.sh
#
# Requires: Redis on 127.0.0.1:6379 (or edit examples/config.local-ui.yaml).
# Optional: .env with OXLLM_MASTER_KEY and provider keys.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export OXLLM_MASTER_KEY="${OXLLM_MASTER_KEY:-sk-ox-local-ui-master}"
export OXLLM_REQUIRE_AUTH="${OXLLM_REQUIRE_AUTH:-true}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-placeholder-openai}"
export XAI_API_KEY="${XAI_API_KEY:-xai-placeholder}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-placeholder-anthropic}"
export ADMIN_SESSION_SECRET="${ADMIN_SESSION_SECRET:-local-dev-session-secret-32chars-min}"
# This script serves local HTTP; use secure cookies for an HTTPS deployment.
export ADMIN_COOKIE_SECURE="${ADMIN_COOKIE_SECURE:-false}"
export RUST_LOG="${RUST_LOG:-info,oxllm=info,oxllm_server=info,oxllm_admin=info}"

CFG="${OXLLM_CONFIG:-examples/config.local-ui.yaml}"
PROXY_PORT="${OXLLM_PORT:-4000}"
ADMIN_LISTEN="${OXLLM_ADMIN_LISTEN:-127.0.0.1:8080}"
PROXY_URL="${OXLLM_PROXY_URL:-http://127.0.0.1:${PROXY_PORT}}"

if ! redis-cli -h 127.0.0.1 -p 6379 ping >/dev/null 2>&1; then
  echo "WARN: Redis not responding on 6379 — readiness may show degraded." >&2
fi

# Existing listeners may be unrelated processes. Do not stop them implicitly.
for port in "$PROXY_PORT" "${ADMIN_LISTEN##*:}"; do
  if lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is busy; choose another port or stop its owner explicitly." >&2
    exit 1
  fi
done

echo "Building (db + admin features)..."
cargo build --locked --features db,admin --bin oxllm --bin oxllm-admin

cleanup() {
  echo ""
  echo "Stopping stack..."
  [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null || true
  [[ -n "${ADMIN_PID:-}" ]] && kill "$ADMIN_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

./target/debug/oxllm serve --config "$CFG" --port "$PROXY_PORT" > /tmp/oxllm-proxy.log 2>&1 &
PROXY_PID=$!
sleep 1
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
  echo "Proxy failed to start; see /tmp/oxllm-proxy.log" >&2
  cat /tmp/oxllm-proxy.log >&2 || true
  exit 1
fi

./target/debug/oxllm-admin serve --listen "$ADMIN_LISTEN" --proxy-url "$PROXY_URL" \
  > /tmp/oxllm-admin.log 2>&1 &
ADMIN_PID=$!
sleep 1
if ! kill -0 "$ADMIN_PID" 2>/dev/null; then
  echo "Admin failed to start; see /tmp/oxllm-admin.log" >&2
  cat /tmp/oxllm-admin.log >&2 || true
  exit 1
fi

echo ""
echo "oxllm stack is up"
echo "  Proxy:  $PROXY_URL   (logs: /tmp/oxllm-proxy.log)"
echo "  Admin:  http://$ADMIN_LISTEN  (logs: /tmp/oxllm-admin.log)"
echo "  Master: \$OXLLM_MASTER_KEY (from env)"
echo "  Ctrl-C to stop both"
echo ""

wait
