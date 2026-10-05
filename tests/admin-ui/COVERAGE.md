# Admin frontend / e2e coverage map

Layers:

| Layer | Tool | What it proves |
|-------|------|----------------|
| Handler + template (Rust) | `cargo test -p oxllm-admin` (**195** tests) | Routes, CSRF, proxy mapping, HTML fragments |
| Browser e2e | Playwright `tests/admin-ui` (**17** tests) | HTMX/Alpine/CSRF/dialogs as a real user |

There is **no Istanbul-style %** for MiniJinja/HTML (not a JS SPA). This matrix is the coverage source of truth.

## Surface matrix

| Surface | Rust | Playwright | Status |
|---------|------|------------|--------|
| Login | yes | yes | covered |
| Logout + confirm | yes | yes | covered |
| CSRF via `document` listener | yes | yes | covered (regression) |
| Alpine `initTree` after HTMX | yes (script) | yes (key modal + disable) | covered |
| Dashboard totals / health | yes | yes | covered |
| Dashboard activity **bar width** | yes (`bar_pct`) | yes | covered + **bug fixed** |
| Keys create + modal | yes | yes | covered |
| Keys create **team/user selects** | yes | yes | LiteLLM-style dropdowns |
| Keys disable (soft-delete) | yes | yes | covered + **bug fixed** |
| Teams / users create | yes | yes | covered |
| Spend list + costs + **key/team labels** | yes | yes | master → `master`; alias join |
| Spend refresh no-dupe | yes | yes | covered |
| Spend filter (model/key/team) | yes | yes | key/team selects |
| Spend column sort | yes | yes | page-level sort |
| Spend CSV export | yes | yes | `/spend/export.csv` |
| Models list | yes | yes | covered |
| Theme toggle | — | yes | covered |
| Key rotate (secret shown once) | yes | yes | covered |
| Cost calculator estimate | yes | yes | covered |
| Spend date presets (7d/30d) | yes | yes | covered |
| Team/user edit + delete | yes | yes | covered |
| Spend load-more click | pagination partial | — | gap if few rows |
| Clipboard copy key | — | — | gap (headless perms) |
| Wrong password UI | rate-limit unit | — | gap unless password set |
| Mobile sidebar | — | — | gap |

## Bugs found in this pass

1. **Dashboard bars always 0%** — MiniJinja loop `{% set max_spend %}` does not update outer scope. Fixed by computing `bar_pct` in Rust.
2. **Alpine not re-inited after HTMX** — key modal / interactive fragments needed `Alpine.initTree` on `htmx:afterSwap`.
3. **Key “delete” UX inconsistency** — soft-delete left inactive keys in API list, but HTMX returned empty body so the row vanished until reload, then reappeared as Disabled. Fixed: disable refreshes table; button labeled **Disable**.

## Commands

```bash
cargo test -p oxllm-admin

./scripts/dev-stack.sh   # other terminal
cd tests/admin-ui && npm test
```
