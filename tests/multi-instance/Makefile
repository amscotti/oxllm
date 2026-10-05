.PHONY: help build up down status logs probes scenario soak clean

# Convenience targets for the 3-instance + Redis stress stack.
# Run from the repo root:  make -f tests/multi-instance/Makefile up

COMPOSE := docker compose -f tests/multi-instance/docker-compose.yml
BASE    := http://127.0.0.1:8080
A       := http://127.0.0.1:4000
B       := http://127.0.0.1:4001
C       := http://127.0.0.1:4002
PYTHON  ?= python3
SCENARIO_FLAGS ?=

help:
	@echo "Targets: build up down status logs probes scenario soak clean"
	@echo "  build    - build all images (oxllm/migrate use the same Dockerfile and build cache)"
	@echo "  up       - start the full stack (redis, postgres, mock, 3x oxllm, lb, prom, grafana)"
	@echo "  down     - stop the stack (keeps pgdata volume)"
	@echo "  status   - ps + health endpoints"
	@echo "  probes   - one smoke request per path"
	@echo "  scenario - run the Python invariant scenarios"
	@echo "             PYTHON selects the interpreter; SCENARIO_FLAGS=--allow-skips explicitly permits SKIP"
	@echo "  soak     - 10,000 non-streaming chat requests via oha (if installed)"
	@echo "  clean    - down AND delete the pgdata volume"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d
	@echo "LB $(BASE)  |  A $(A)  |  B $(B)  |  C $(C)  |  Prometheus :9090  |  Grafana :3000"

down:
	$(COMPOSE) down

status:
	$(COMPOSE) ps
	@echo "--- health ---"
	@curl -fsS $(A)/health && echo "  [A ok]" || echo "  [A DOWN]"
	@curl -fsS $(B)/health && echo "  [B ok]" || echo "  [B DOWN]"
	@curl -fsS $(C)/health && echo "  [C ok]" || echo "  [C DOWN]"
	@curl -fsS $(BASE)/health && echo "  [LB ok]" || echo "  [LB DOWN]"

logs:
	$(COMPOSE) logs -f --tail=100 oxllm-a oxllm-b oxllm-c

probes:
	@echo "--- chat via LB ---"
	@curl -fsS -X POST $(BASE)/v1/chat/completions \
	  -H 'Authorization: Bearer sk-stress-master-key' -H 'Content-Type: application/json' \
	  -d '{"model":"mock-unthrottled","messages":[{"role":"user","content":"hi"}]}' | jq -c '{id,model,content:.choices[0].message.content}'
	@echo "--- stream via A ---"
	@curl -fsS -N -X POST $(A)/v1/chat/completions \
	  -H 'Authorization: Bearer sk-stress-master-key' -H 'Content-Type: application/json' \
	  -d '{"model":"mock-unthrottled","stream":true,"messages":[{"role":"user","content":"hi"}]}' | tail -2
	@echo "--- models on B ---"
	@curl -fsS $(B)/v1/models -H 'Authorization: Bearer sk-stress-master-key' | jq -c '.data[].id'

scenario:
	$(PYTHON) tests/multi-instance/scenarios.py --base $(BASE) --a $(A) --b $(B) --c $(C) $(SCENARIO_FLAGS)

soak:
	@command -v oha >/dev/null || { echo "install oha: brew install oha"; exit 1; }
	oha -n 10000 -c 100 -m POST -T 'application/json' \
	  -H 'Authorization: Bearer sk-stress-master-key' \
	  -d '{"model":"mock-unthrottled","messages":[{"role":"user","content":"soak load test"}]}' \
	  $(BASE)/v1/chat/completions

clean: down
	$(COMPOSE) down -v
