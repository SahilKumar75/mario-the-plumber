#!/usr/bin/env bash
#
# validate-submission.sh - OpenEnv submission validator

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}"
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC}\n" "$1"
  exit 1
}

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/4: Pinging HF Space${NC} ($PING_URL/reset) ..."
CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable"
  hint "Check the Space URL and confirm the app is running."
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE"
  hint "Verify the deployed URL and your reset endpoint."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/4: Running docker build${NC} ..."
if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
else
  fail "No Dockerfile found in repo root or server/"
  stop_at "Step 2"
fi

BUILD_OK=false
BUILD_OUTPUT=$(
  cd "$DOCKER_CONTEXT" &&
  run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -f server/Dockerfile .
  2>&1
) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/4: Running openenv validate${NC} ..."
if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

log "${BOLD}Step 4/4: Verifying inference stdout protocol${NC} ..."
if command -v python3 &>/dev/null; then
  PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
  PYTHON_CMD="python"
else
  fail "python command not found"
  stop_at "Step 4"
fi

INFERENCE_STDOUT=$(portable_mktemp "validate-inference")
CLEANUP_FILES+=("$INFERENCE_STDOUT")
INFERENCE_OK=false
(
  cd "$REPO_DIR" &&
  "$PYTHON_CMD" -m inference --seed 1 --split eval --policy-mode heuristic >"$INFERENCE_STDOUT" 2>&1
) && INFERENCE_OK=true

if [ "$INFERENCE_OK" != true ]; then
  fail "inference protocol run failed"
  tail -20 "$INFERENCE_STDOUT"
  stop_at "Step 4"
fi

PROTO_CHECK_OK=false
PROTO_CHECK_OUTPUT=$("$PYTHON_CMD" - "$INFERENCE_STDOUT" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
  lines = [line.strip() for line in handle if line.strip()]

if not lines:
  raise SystemExit("inference output is empty")

has_bracket_tags = any(
  line.startswith("[START]") or line.startswith("[STEP]") or line.startswith("[END]")
  for line in lines
)

if has_bracket_tags:
  start_count = sum(1 for line in lines if line.startswith("[START]"))
  step_count = sum(1 for line in lines if line.startswith("[STEP]"))
  end_count = sum(1 for line in lines if line.startswith("[END]"))
  if start_count != 1:
    raise SystemExit(f"expected exactly one [START] line, got {start_count}")
  if end_count != 1:
    raise SystemExit(f"expected exactly one [END] line, got {end_count}")
  if step_count < 1:
    raise SystemExit("expected at least one [STEP] line")
  print(f"steps={step_count} mode=bracket")
  raise SystemExit(0)

start = None
end = None
steps = []

for raw in lines:
  if " " not in raw:
    raise SystemExit(f"malformed protocol line: {raw}")
  tag, payload_blob = raw.split(" ", 1)
  if tag not in {"START", "STEP", "END"}:
    raise SystemExit(f"unknown protocol tag: {tag}")
  try:
    payload = json.loads(payload_blob)
  except json.JSONDecodeError as exc:
    raise SystemExit(f"invalid JSON payload for tag {tag}: {exc}") from exc
  if not isinstance(payload, dict):
    raise SystemExit(f"payload for {tag} must be an object")

  if tag == "START":
    if start is not None:
      raise SystemExit("multiple START lines found")
    if end is not None:
      raise SystemExit("START appeared after END")
    start = payload
  elif tag == "STEP":
    if start is None:
      raise SystemExit("STEP appeared before START")
    if end is not None:
      raise SystemExit("STEP appeared after END")
    steps.append(payload)
  else:
    if start is None:
      raise SystemExit("END appeared before START")
    if end is not None:
      raise SystemExit("multiple END lines found")
    end = payload

if start is None or end is None:
  raise SystemExit("protocol missing START or END")

task_ids = sorted({int(step.get("task_id")) for step in steps if "task_id" in step})
if task_ids != [1, 2, 3, 4, 5]:
  raise SystemExit(f"expected STEP task coverage [1,2,3,4,5], got {task_ids}")

if end.get("status") != "complete":
  raise SystemExit(f"END payload status must be complete, got {end.get('status')!r}")

print(f"steps={len(steps)} tasks={task_ids}")
PY
) && PROTO_CHECK_OK=true

if [ "$PROTO_CHECK_OK" = true ]; then
  pass "inference stdout protocol is parser-compliant"
  [ -n "$PROTO_CHECK_OUTPUT" ] && log "  $PROTO_CHECK_OUTPUT"
else
  fail "inference stdout protocol check failed"
  printf "%s\n" "$PROTO_CHECK_OUTPUT"
  stop_at "Step 4"
fi

printf "\n${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 4/4 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n\n"
