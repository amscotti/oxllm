FROM rust:1.96-bookworm AS builder
WORKDIR /app

# Copy the real workspace (not a dummy-crate stub). The stub layer used to
# omit workspace members (`oxllm-cache-{s3,gcs,azure}`) and extra Cargo.toml
# targets (e.g. oxllm-providers benches), which made `docker build` fail
# before the real `COPY . .`.
COPY rust-toolchain.toml Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY src ./src
COPY examples/config.docker.yaml ./examples/config.docker.yaml
RUN cargo build --locked --release --bin oxllm --features db,otel

# Slim runtime with CA certs (required for HTTPS to OpenAI/Anthropic/etc.).
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/oxllm /oxllm
COPY --from=builder /app/examples/config.docker.yaml /config.yaml

# Run as an unprivileged user (UID 10001) so the long-running daemon does not
# write volumes/logs as root and a runtime escape does not land as UID 0.
RUN groupadd -g 10001 oxllm \
    && useradd -r -u 10001 -g oxllm oxllm \
    && mkdir /data \
    && chown oxllm:oxllm /data /config.yaml
USER 10001:10001

# Default port; override via `docker run -e OXLLM_PORT=4001 ...` for
# multi-instance deployments behind a load balancer. The CLI reads this
# env var (clap `env = "OXLLM_PORT"`) so both the server and the
# HEALTHCHECK stay in sync.
ENV OXLLM_PORT=4000
ENV DATABASE_URL=sqlite:///data/oxllm.db?mode=rwc
EXPOSE ${OXLLM_PORT}

# Shell form so ${OXLLM_PORT} expands; works for any port override.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD curl -fsS http://127.0.0.1:${OXLLM_PORT}/health || exit 1

# ENTRYPOINT is the binary only; CMD supplies the subcommand + default
# config. Override `command:` in compose to change config/port:
#   command: ["serve", "--config", "/etc/oxllm/config.yaml"]
ENTRYPOINT ["/oxllm"]
CMD ["serve", "--config", "/config.yaml"]
