#!/usr/bin/env bash
#
# validate-live-space.sh - quick live benchmark validator for deployed Mario Space

set -euo pipefail

SPACE_URL="${1:-https://sahilksingh-mario-the-plumber.hf.space}"
SPACE_URL="${SPACE_URL%/}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required"
  exit 1
fi

check_json() {
  local label="$1"
  local method="$2"
  local url="$3"
  local body="${4:-}"

  echo "== $label =="
  if [ "$method" = "POST" ]; then
    curl -s -X POST -H "Content-Type: application/json" -d "$body" "$url"
  else
    curl -s "$url"
  fi
  echo
  echo
}

echo "Validating live Space: $SPACE_URL"
echo

check_json "Health" "GET" "$SPACE_URL/health"
check_json "Tasks" "GET" "$SPACE_URL/tasks"
check_json "Benchmark metadata" "GET" "$SPACE_URL/benchmark/metadata"
check_json "Benchmark tasks" "GET" "$SPACE_URL/benchmark/tasks"
check_json "Reset task 3 eval seed 42" "POST" "$SPACE_URL/reset" '{"task_id":3,"split":"eval","seed":42}'

echo "Done."
