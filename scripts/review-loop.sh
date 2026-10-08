#!/usr/bin/env bash
# =============================================================================
# oxllm review-loop.sh
#
# Drives an autonomous code-review loop over every component in
# .review/checklist.md using opencode (headless `opencode run`).
#
# Checklist format (gaggle-style, see .review/checklist.md):
#   - [ ] <slug> — <description>
#     paths: <comma-separated paths>
#     verify: <command>
#
# For each unchecked item, in order:
#   1. REVIEW phase  — GLM-5.3 (Z.AI Coding Plan) reviews the component,
#                      writes a findings report to .review/findings/<slug>.txt
#   2. FIX phase     — GLM-5.3 reads the findings, fixes the issues, runs
#                      fmt/clippy/test, then checks the box `- [ ]` -> `- [x]`
#
# The loop ends when no unchecked items remain. Resumable: re-run anytime and it
# picks up at the first remaining `- [ ]`.
#
# Usage:
#   scripts/review-loop.sh                     # run the whole loop (next unchecked onward)
#   scripts/review-loop.sh build-deploy        # review + fix exactly one component (by slug)
#   scripts/review-loop.sh --list              # print checklist status, no changes
#   scripts/review-loop.sh --dry-run           # show what would run, do not call opencode
#   scripts/review-loop.sh --review-only build-deploy   # skip the fix phase
#   scripts/review-loop.sh --backfill          # rebuild .review/metrics.tsv from past sessions
#   scripts/review-loop.sh --report            # print time/token totals logged so far
#   scripts/review-loop.sh --reset             # uncheck all boxes for a fresh run (clears metrics)
#
# Env:
#   MODEL         opencode model id            (default: zai-coding-plan/glm-5.3)
#   OPENCODE_BIN  path to opencode             (default: opencode)
#   EXTRA_FLAGS   extra flags appended to every `opencode run`
#   REVIEW_FLAGS  extra flags for the review phase only
#   FIX_FLAGS     extra flags for the fix phase only
#   COMMIT        1 = git-commit each component's fixes before checking it off
#                 (default 1). The script owns the checklist + commits; the
#                 fix phase only edits code.
#   SINGLE_COMMIT 1 = bundle code + checkbox into one commit. 0 (default) =
#                 two commits: "fix: …" then "chore(review): mark … done".
#   VERIFY_BEFORE_COMMIT  1 = run fmt/clippy/test in the script before
#                 committing (independent of the fix phase). default 0.
#   METRICS       1 = log per-step wall-clock + tokens + cost to
#                 .review/metrics.tsv (default 1). Cost is $0 — Z.AI Coding
#                 Plan is a flat subscription.
# =============================================================================

set -euo pipefail

# --- paths -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
# Review state lives under .review/ (see .gitignore: the checklist and
# workflow/config files are committed; findings/logs/metrics are per-run).
CHECKLIST="$REPO/.review/checklist.md"
REVIEW_TPL="$SCRIPT_DIR/prompts/REVIEW_PROMPT.md"
FIX_TPL="$SCRIPT_DIR/prompts/FIX_PROMPT.md"
FINDINGS_DIR="$REPO/.review/findings"
LOG_DIR="$REPO/.review/logs"
METRICS_FILE="$REPO/.review/metrics.tsv"
METRICS_PY="$SCRIPT_DIR/review-metrics.py"
# Upstream clone (sibling) — the authoritative reference for behavior.
# `|| true` keeps `set -e` from killing the script on machines with no sibling
# upstream clone (the `[ -z … ] &&` fallback below then supplies a placeholder).
LITELLM_PATH="$(cd "$REPO/../litellm" 2>/dev/null && pwd || true)"
[ -z "$LITELLM_PATH" ] && LITELLM_PATH="(no local LiteLLM clone — use the curated docs + web/context7 lookup)"
mkdir -p "$FINDINGS_DIR" "$LOG_DIR"

# --- config ----------------------------------------------------------------
MODEL="${MODEL:-zai-coding-plan/glm-5.3}"
OPENCODE_BIN="${OPENCODE_BIN:-opencode}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
REVIEW_FLAGS="${REVIEW_FLAGS:-}"
FIX_FLAGS="${FIX_FLAGS:-}"
COMMIT="${COMMIT:-1}"
SINGLE_COMMIT="${SINGLE_COMMIT:-0}"
VERIFY_BEFORE_COMMIT="${VERIFY_BEFORE_COMMIT:-0}"
METRICS="${METRICS:-1}"   # 1 = log per-step time/tokens/cost to .review/metrics.tsv
# Cost-estimate rates (USD per 1M tokens). GLM-5.3 via the Z.AI Coding Plan is
# a flat subscription, so all rates are $0 (cost reported is always $0).
# Override via env for pay-per-token models. NOTE: GLM-5.3 per-token API rates
# are not yet published (Z.AI table ends at GLM-5.2: 1.40/0.26/4.40) — use those
# as the closest reference for a what-if estimate.
export PRICE_INPUT="${PRICE_INPUT:-0}"
export PRICE_CACHE="${PRICE_CACHE:-0}"
export PRICE_OUTPUT="${PRICE_OUTPUT:-0}"

TODAY="$(date +%Y-%m-%d)"
DRY_RUN=0
REVIEW_ONLY=0
ONLY_ID=""

# --- helpers ---------------------------------------------------------------
c_red()    { printf '\033[31m%s\033[0m' "$*"; }
c_green()  { printf '\033[32m%s\033[0m' "$*"; }
c_yellow() { printf '\033[33m%s\033[0m' "$*"; }
c_cyan()   { printf '\033[36m%s\033[0m' "$*"; }
c_bold()   { printf '\033[1m%s\033[0m' "$*"; }

log()  { printf '%s\n' "$*"; }
hdr()  { printf '\n%s\n' "$(c_bold "$(c_cyan "=== $* ===")")"; }

require() { command -v "$1" >/dev/null 2>&1 || { log "$(c_red "missing dependency: $1")"; exit 127; }; }

# Find the first unchecked component line. Strict: box + slug + em dash.
# Prints the raw line, or empty if none remain.
next_line() {
  grep -m1 -E '^- \[ \] [a-z0-9_-]+ — ' "$CHECKLIST" || true
}

# Find a specific component line regardless of checked state (slug-keyed).
line_for_id() {
  local id="$1"
  grep -m1 -E "^- \[[ x]\] ${id} — " "$CHECKLIST" || true
}

# Parse "- [ ] build-deploy — Build and deployment" (+ the `paths:` sub-line)
# into globals. Sets: COMP_ID, COMP_SLUG, COMP_NAME, COMP_PATHS
parse_line() {
  local line="$1"
  local rest="${line#- \[[ x]\] }"      # strip "- [ ] " / "- [x] "
  COMP_ID="$(printf '%s' "$rest" | sed -E 's/ — .*$//')"
  COMP_SLUG="$COMP_ID"
  COMP_NAME="$(printf '%s' "$rest" | sed -E 's/^[^—]*—[[:space:]]*//')"
  # The `paths:` sub-line follows the component line in the checklist.
  COMP_PATHS="$(grep -A2 -E "^- \[[ x]\] ${COMP_ID} — " "$CHECKLIST" \
    | sed -n 's/^[[:space:]]*paths:[[:space:]]*//p' | head -1)"
  [[ -z "$COMP_PATHS" ]] && COMP_PATHS="$COMP_NAME" || true  # no paths line
  return 0
}

# Render a prompt template by substituting {{PLACEHOLDERS}}.
# Uses '|' as the sed delimiter; component paths contain no '|'.
render() {
  local tmpl="$1" out="$2"
  sed -e "s|{{COMPONENT_ID}}|${COMP_ID}|g" \
      -e "s|{{COMPONENT_NAME}}|${COMP_NAME}|g" \
      -e "s|{{COMPONENT_PATHS}}|${COMP_PATHS}|g" \
      -e "s|{{LITELLM_PATH}}|${LITELLM_PATH}|g" \
      -e "s|{{FINDINGS_FILE}}|${3}|g" \
      -e "s|{{DATE}}|${TODAY}|g" \
      "$tmpl" > "$out"
}

# Force-check a box: flip "- [ ] <slug> —" -> "- [x] <slug> —".
check_box() {
  local id="$1"
  # in-place edit; BSD sed needs the '' after -i
  if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' -E "s/^- \[ \] ${id} — /- [x] ${id} — /" "$CHECKLIST"
  else
    sed -i    -E "s/^- \[ \] ${id} — /- [x] ${id} — /" "$CHECKLIST"
  fi
}

is_checked() {
  local id="$1"
  grep -qE "^- \[x\] ${id} — " "$CHECKLIST"
}

# --- metrics (per-step time / tokens / cost) --------------------------------
init_metrics() {
  [[ "$METRICS" -eq 1 ]] || return 0
  [[ -f "$METRICS_FILE" ]] || \
    printf 'component\tphase\tduration_s\ttok_in\ttok_out\treasoning\tcache_read\tcache_write\tcost_usd\tsession_id\n' \
      > "$METRICS_FILE"
}

# Resolve the opencode session id for a titled run (review-Cxx / fix-Cxx).
session_id_for_title() { "$OPENCODE_BIN" session list 2>/dev/null | awk -v t="$1" '$2==t {print $1; exit}'; }

# Append one metrics row. $1=component $2=phase(review|fix) $3=measured_s(optional)
# Tokens/cost come from the session export; duration falls back to the export's
# session span when no measured wall-clock is supplied (used by --backfill).
record_metrics() {
  [[ "$METRICS" -eq 1 ]] || return 0
  [[ "$DRY_RUN" -eq 1 ]] && return 0
  local comp="$1" phase="$2" meas="${3:-}" sid
  local title="${phase}-${comp}"
  sid="$(session_id_for_title "$title")" || sid=""
  [[ -n "$sid" ]] || { log "  $(c_yellow "metrics: no session found for $title")"; return 0; }
  local j; j="$(mktemp)"
  if ! "$OPENCODE_BIN" export "$sid" > "$j" 2>/dev/null; then
    rm -f "$j"; log "  $(c_yellow "metrics: export failed for $sid")"; return 0
  fi
  local row; row="$(python3 "$METRICS_PY" "$j" 2>/dev/null || true)"; rm -f "$j"
  [[ -n "$row" ]] || return 0
  local t_in t_out t_re c_read c_write cost span dur
  IFS=$'\t' read -r t_in t_out t_re c_read c_write cost span <<< "$row"
  dur="$meas"; { [[ -z "$dur" || "$dur" == "0" ]]; } && dur="${span:-0}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$comp" "$phase" "$dur" "$t_in" "$t_out" "$t_re" "$c_read" "$c_write" "$cost" "$sid" \
    >> "$METRICS_FILE"
}

# Sum the metrics file and print a totals report.
print_totals() {
  [[ -f "$METRICS_FILE" ]] || { log "$(c_yellow "no metrics file yet ($METRICS_FILE)")"; return 0; }
  awk -F'\t' 'NR>1 { d+=$3; ti+=$4; to+=$5; re+=$6; cr+=$7; cw+=$8; co+=$9; n++ }
  END {
    if (!n) { printf "(no rows)\n"; exit }
    printf "\n——— review-loop totals (%d steps) ———\n", n
    printf "wall-clock time  : %dh %02dm\n", d/3600, (d%3600)/60
    printf "tokens  input    : %\47d\n", ti
    printf "tokens  output   : %\47d\n", to
    printf "tokens  reasoning: %\47d\n", re
    printf "cache read       : %\47d  (prompt-cache reuse)\n", cr
    printf "cache write      : %\47d\n", cw
    printf "est. cost (USD)  : $%.2f   (GLM-5.3 via Z.AI Coding Plan — flat subscription)\n", co
  }' "$METRICS_FILE"
}

# Rebuild metrics.tsv from already-completed components' session exports (read-only).
backfill_metrics() {
  log "backfilling metrics from completed session exports (read-only)…"
  printf 'component\tphase\tduration_s\ttok_in\ttok_out\treasoning\tcache_read\tcache_write\tcost_usd\tsession_id\n' \
    > "$METRICS_FILE"
  grep -E '^- \[x\] [a-z0-9_-]+ — ' "$CHECKLIST" | while IFS= read -r ln; do
    parse_line "$ln"
    record_metrics "$COMP_ID" review ""
    record_metrics "$COMP_ID" fix ""
  done
  print_totals
}

# --- phases ----------------------------------------------------------------
run_review() {
  local findings="$FINDINGS_DIR/${COMP_SLUG}.txt"
  local logf="$LOG_DIR/${COMP_SLUG}-review.log"
  local rendered
  rendered="$(mktemp)"
  render "$REVIEW_TPL" "$rendered" "$findings"
  local prompt; prompt="$(cat "$rendered")"
  rm -f "$rendered"

  hdr "REVIEW  $(c_yellow "$COMP_ID") $COMP_NAME"
  log "paths: $COMP_PATHS"
  log "findings -> $findings"
  log "log     -> $logf"

  if [[ "$DRY_RUN" -eq 1 ]]; then log "$(c_yellow "[dry-run] skip opencode")"; return 0; fi

  local start; start="$(date +%s)"
  # shellcheck disable=SC2086
  "$OPENCODE_BIN" run \
    --dir "$REPO" \
    -m "$MODEL" \
    --auto \
    --title "review-$COMP_ID" \
    $EXTRA_FLAGS $REVIEW_FLAGS \
    "$prompt" 2>&1 | tee "$logf" || {
      log "$(c_red "review phase exited non-zero for $COMP_ID (see $logf)")"
      return 1
    }
  record_metrics "$COMP_ID" review "$(( $(date +%s) - start ))"

  if [[ ! -s "$findings" ]]; then
    log "$(c_red "no findings file produced at $findings")"
    return 1
  fi
}

run_fix() {
  local findings="$FINDINGS_DIR/${COMP_SLUG}.txt"
  local logf="$LOG_DIR/${COMP_SLUG}-fix.log"
  local rendered
  rendered="$(mktemp)"
  render "$FIX_TPL" "$rendered" "$findings"
  local prompt; prompt="$(cat "$rendered")"
  rm -f "$rendered"

  hdr "FIX     $(c_yellow "$COMP_ID") $COMP_NAME"
  log "reading findings <- $findings"
  log "log     -> $logf"

  if [[ "$DRY_RUN" -eq 1 ]]; then log "$(c_yellow "[dry-run] skip opencode")"; return 0; fi

  local start; start="$(date +%s)"
  # shellcheck disable=SC2086
  "$OPENCODE_BIN" run \
    --dir "$REPO" \
    -m "$MODEL" \
    --auto \
    --title "fix-$COMP_ID" \
    $EXTRA_FLAGS $FIX_FLAGS \
    "$prompt" 2>&1 | tee "$logf" || {
      log "$(c_red "fix phase exited non-zero for $COMP_ID (see $logf)")"
      return 1
    }
  record_metrics "$COMP_ID" fix "$(( $(date +%s) - start ))"
}

# Stage + commit. mode: "fixes" (code+findings), "progress" (checklist),
# or "bundle" (code+findings+checklist in one commit). No-op if nothing staged.
commit_work() {
  [[ "$COMMIT" -eq 1 ]] || { log "  $(c_yellow "[COMMIT=0] skip commit ($1)")"; return 0; }
  local mode="$1"
  ( cd "$REPO"
    local paths=()
    case "$mode" in
      fixes)
        # Code + docs only. Findings live under .review/findings which is
        # gitignored by design (per-run artifacts), so they are not committed.
        for p in crates src tests examples docs Cargo.toml Cargo.lock; do [ -e "$p" ] && paths+=("$p"); done
        git add -- "${paths[@]}"
        git diff --cached --quiet && { echo "  (no code changes to commit)"; exit 0; }
        git -c commit.gpgsign=false commit -q \
          -m "fix: ${COMP_NAME} — review fixes (${COMP_ID})" \
          -m "Component: ${COMP_ID} (${COMP_SLUG})" \
          -m "Findings: .review/findings/${COMP_SLUG}.txt" \
          -m "Auto-reviewed+fixed by review-loop.sh (GLM-5.3, Z.AI Coding Plan)." ;;
      progress)
        git add -- .review/checklist.md
        git diff --cached --quiet && exit 0
        git -c commit.gpgsign=false commit -q -m "chore(review): mark ${COMP_ID} (${COMP_SLUG}) done" ;;
      bundle)
        for p in crates src tests examples docs Cargo.toml Cargo.lock; do [ -e "$p" ] && paths+=("$p"); done
        paths+=(".review/checklist.md")
        git add -- "${paths[@]}"
        git diff --cached --quiet && { echo "  (no changes to commit)"; exit 0; }
        git -c commit.gpgsign=false commit -q \
          -m "fix: ${COMP_NAME} — review fixes + mark done (${COMP_ID})" \
          -m "Component: ${COMP_ID} (${COMP_SLUG})" \
          -m "Auto-reviewed+fixed by review-loop.sh (GLM-5.3, Z.AI Coding Plan)." ;;
      *) exit 2 ;;
    esac
    echo "  committed: $(git rev-parse --short HEAD)  ($mode)"
  )
}

# Optional independent gate run before committing. Off by default — the fix
# phase already runs fmt/clippy/test.
verify_gate() {
  [[ "$VERIFY_BEFORE_COMMIT" -eq 1 ]] || return 0
  log "VERIFY_BEFORE_COMMIT: running fmt + clippy + test …"
  ( cd "$REPO"
    cargo fmt --all -- --check \
      && cargo clippy --workspace --lib --bins --tests -- -D warnings \
      && cargo test --workspace
  ) || { log "$(c_red "verify gate failed — NOT committing, NOT checking off")"; return 1; }
}

# Process one component: review -> fix -> commit -> check off.
# The script owns the checklist and the commits; the fix phase only edits code.
do_component() {
  local line="$1"
  parse_line "$line"

  run_review || return 1
  if [[ "$REVIEW_ONLY" -eq 1 ]]; then
    log "$(c_yellow "review-only: leaving box unchecked")"
    return 0
  fi
  run_fix || return 1

  # dry-run is just a preview
  [[ "$DRY_RUN" -eq 1 ]] && { log "$(c_yellow "[dry-run] would commit + check box for $COMP_ID")"; return 0; }

  verify_gate || return 1

  if [[ "$SINGLE_COMMIT" -eq 1 ]]; then
    # flip first, then one commit bundling code + findings + checklist
    check_box "$COMP_ID"
    commit_work "bundle" || return 1
  else
    # COMMIT THE FIXES FIRST (before marking done), then check off, then a
    # tiny follow-up commit for the checkbox. This ordering is crash-safe:
    # if the script dies after the fix commit, the work is saved and the
    # component is re-run (box still unchecked) on relaunch.
    commit_work "fixes" || return 1
    check_box "$COMP_ID"
    commit_work "progress" || return 1
  fi
  log "$(c_green "$COMP_ID complete")"
}

# Reset for a fresh run: uncheck every box, clear metrics + findings + logs.
reset_run() {
  local n
  n="$(grep -cE '^- \[x\] [a-z0-9_-]+ — ' "$CHECKLIST" || true)"
  if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' -E 's/^- \[x\] ([a-z0-9_-]+ — )/- [ ] \1/' "$CHECKLIST"
  else
    sed -i    -E 's/^- \[x\] ([a-z0-9_-]+ — )/- [ ] \1/' "$CHECKLIST"
  fi
  printf 'component\tphase\tduration_s\ttok_in\ttok_out\treasoning\tcache_read\tcache_write\tcost_usd\tsession_id\n' > "$METRICS_FILE"
  rm -f "$FINDINGS_DIR"/*.md "$FINDINGS_DIR"/*.txt "$LOG_DIR"/*.log 2>/dev/null || true
  log "$(c_green "reset complete: $n checked → unchecked; cleared metrics.tsv, findings/, logs/")"
  log "$(c_yellow "tip: back up first if you want to keep run-1 reports: cp -r .review/findings .review/findings-run1")"
}

# --- main ------------------------------------------------------------------
require "$OPENCODE_BIN"

# Fail loudly (instead of silently reporting "ALL COMPONENTS CHECKED") when the
# checklist or a prompt template is missing — `grep … || true` in next_line()
# would otherwise mask a missing checklist as "nothing left to do".
for required in "$CHECKLIST" "$REVIEW_TPL" "$FIX_TPL"; do
  [[ -f "$required" ]] || { log "$(c_red "missing required file: $required")"; exit 1; }
done

# arg parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)        U="$(grep -cE '^- \[ \] [a-z0-9_-]+ — ' "$CHECKLIST" || true)"
                   C="$(grep -cE '^- \[x\] [a-z0-9_-]+ — ' "$CHECKLIST" || true)"
                   printf 'unchecked: %s\n' "$U"
                   printf 'checked:   %s\n' "$C"
                   exit 0 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --review-only) REVIEW_ONLY=1; shift ;;
    --report)      init_metrics; print_totals; exit 0 ;;
    --backfill)    init_metrics; backfill_metrics; exit 0 ;;
    --reset)       reset_run; exit 0 ;;
    -h|--help)     sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)            log "$(c_red "unknown flag: $1")"; exit 2 ;;
    [a-z0-9_-]*)   ONLY_ID="$1"; shift ;;
    *)             log "$(c_red "unknown arg: $1")"; exit 2 ;;
  esac
done

init_metrics

log "repo:    $REPO"
log "model:   $(c_bold "$MODEL")"
log "dry-run: $DRY_RUN  review-only: $REVIEW_ONLY  commit: $COMMIT  single-commit: $SINGLE_COMMIT"

if [[ -n "$ONLY_ID" ]]; then
  line="$(line_for_id "$ONLY_ID")"
  [[ -n "$line" ]] || { log "$(c_red "no component $ONLY_ID in checklist")"; exit 1; }
  do_component "$line"
  exit $?
fi

# main loop
ITER=0
while true; do
  ITER=$((ITER + 1))
  line="$(next_line)"
  if [[ -z "$line" ]]; then
    hdr "$(c_green "ALL COMPONENTS CHECKED — review complete")"
    log "iterations: $ITER"
    print_totals
    exit 0
  fi
  log "$(c_cyan "[loop $ITER]") next -> $line"
  if ! do_component "$line"; then
    log "$(c_red "aborting loop at failed component")"
    exit 1
  fi
  # dry-run never checks boxes; preview one iteration then stop
  [[ "$DRY_RUN" -eq 1 ]] && { log "$(c_yellow "[dry-run] stopping after preview")"; exit 0; }
done
