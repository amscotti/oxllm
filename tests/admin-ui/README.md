# Admin UI e2e (Playwright)

Browser tests for the **oxllm-admin** MVP: login, create team/user/key, spend refresh.

These sit beside Rust workspace e2e (`tests/*_e2e.rs`). Rust covers proxy + admin
handler contracts; Playwright covers HTMX/Alpine/CSRF in a real browser.

## Prerequisites

1. Local stack running:

```bash
./scripts/dev-stack.sh
```

2. Node 20+ and Chromium for Playwright:

```bash
cd tests/admin-ui
npm install
npx playwright install chromium
```

## Run

```bash
cd tests/admin-ui
npm test
```

Optional:

```bash
OXLLM_ADMIN_URL=http://127.0.0.1:8080 npm test
```
