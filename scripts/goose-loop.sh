#!/usr/bin/env bash
# =============================================================================
# oxllm goose-loop.sh — legacy driver (bundled recipe is no longer present)
#
# The agentic work (review, fix, verify-iterate, checklist, commits) happens
# INSIDE goose via the review-loop recipe + subagents; the recipe lives at
# .review/workflows/review-loop.yaml in older checkouts. This checkout does not
# ship that recipe: the execution path is obsolete without a user-supplied
# GOOSE_REVIEW_RECIPE. Review state (checklist, findings,
# metrics) lives under .review/ per .gitignore. This script only: keeps CLI
# parity with review-loop.sh, guards dry-run, and records wall-clock metrics
# from the goose JSON result.
#
# Usage:
#   scripts/goose-loop.sh                       # run the whole loop (next unchecked onward)
#   scripts/goose-loop.sh build-deploy          # review + fix exactly one component
#   scripts/goose-loop.sh --list                # print checklist status, no changes
#   scripts/goose-loop.sh --dry-run             # report only (goose dry_run=true)
#   scripts/goose-loop.sh --review-only build-deploy   # review phase only
#   scripts/goose-loop.sh --report              # print totals logged so far
#   scripts/goose-loop.sh --reset               # uncheck all boxes (clears metrics)
#
# Env:
#   MODEL          goose provider/model override (default: use active config)
#   METRICS        1 = append wall-clock row to .review/metrics.tsv (default 1)
#   GOOSE_REVIEW_RECIPE  optional path to your reviewed, compatible legacy recipe
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKLIST="$REPO/.review/checklist.md"
RECIPE="${GOOSE_REVIEW_RECIPE:-$REPO/.review/workflows/review-loop.yaml}"
METRICS_FILE="$REPO/.review/metrics.tsv"
LOG_FILE="$REPO/.review/last-run.log"
METRICS="${METRICS:-1}"
DRY_RUN=0; REVIEW_ONLY=0; ONLY_ID=""; EXTRA_PARAMS=()

require() { command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1"; exit 127; }; }

# --- arg parsing -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) U=$(grep -cE '^- \[ \] [a-z0-9_-]+ — ' "$CHECKLIST" || true)
            C=$(grep -cE '^- \[x\] [a-z0-9_-]+ — ' "$CHECKLIST" || true)
            printf 'unchecked: %s\n' "$U"
            printf 'checked:   %s\n' "$C"
            exit 0 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --review-only) REVIEW_ONLY=1; shift ;;
    --report)      [[ -f "$METRICS_FILE" ]] && awk -F'\t' 'NR>1{s+=$3;n++}END{printf "goose-loop steps: %d, wall-clock: %dm\n", n, s/60}' "$METRICS_FILE" || echo "(no metrics yet)"; exit 0 ;;
    --reset)       if [[ "$(uname)" == "Darwin" ]]; then sed -i '' -E 's/^- \[x\] ([a-z0-9_-]+ — )/- [ ] \1/' "$CHECKLIST"; else sed -i -E 's/^- \[x\] ([a-z0-9_-]+ — )/- [ ] \1/' "$CHECKLIST"; fi
                   printf 'component\tphase\tduration_s\tstatus\n' > "$METRICS_FILE"
                   echo "reset complete"; exit 0 ;;
    -*)            echo "unknown flag: $1"; exit 2 ;;
    [a-z0-9_-]*)   ONLY_ID="$1"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

[[ -f "$CHECKLIST" ]] || { echo "checklist not found: $CHECKLIST" >&2; exit 1; }
[[ -f "$RECIPE" ]] || {
  echo "Legacy goose review driver is obsolete in this checkout: recipe missing at $RECIPE" >&2
  echo "Use the current review harness, or set GOOSE_REVIEW_RECIPE to a compatible recipe you maintain." >&2
  exit 1
}
require goose
cd "$REPO"

[[ "$DRY_RUN" -eq 1 ]] && EXTRA_PARAMS+=(--params "dry_run=true")
[[ "$REVIEW_ONLY" -eq 1 ]] && EXTRA_PARAMS+=(--params "review_only=true")
[[ -n "$ONLY_ID" ]] && EXTRA_PARAMS+=(--params "component_only=$ONLY_ID")

echo "repo:    $REPO"
echo "recipe:  $RECIPE"
echo "dry-run: $DRY_RUN  review-only: $REVIEW_ONLY  only: ${ONLY_ID:-all}"

START="$(date +%s)"
# The whole loop is ONE goose run; let goose's normal output flow to the
# terminal. tee keeps a copy for debugging (.review/last-run.log).
goose run --no-session --recipe "$RECIPE" ${EXTRA_PARAMS[@]+"${EXTRA_PARAMS[@]}"} 2>&1 | tee "$LOG_FILE" \
  || { echo "$(tput setaf 1)goose run failed (exit $?) — tail $LOG_FILE; checklist state persists; re-run to resume$(tput sgr0)"; exit 1; }
DURATION="$(( $(date +%s) - START ))"

# Record wall-clock (token/cost rows need a goose session-export parser — TBD).
if [[ "$METRICS" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  printf '%s\tloop\t%s\tok\n' "$(date +%Y-%m-%d)" "$DURATION" >> "$METRICS_FILE"
  echo "logged ${DURATION}s to $METRICS_FILE"
fi
