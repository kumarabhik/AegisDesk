#!/usr/bin/env bash
#
# Local mirror of the posted hackathon pre-validation flow.
# This is not the official download, but it checks the same core gates:
#   - live HF Space responds
#   - /reset works
#   - openenv validate passes
#   - Docker builds
#
# Usage:
#   chmod +x validate-submission.sh
#   ./validate-submission.sh <ping_url> [repo_dir]
#

set -euo pipefail

PING_URL="${1:?Usage: ./validate-submission.sh <ping_url> [repo_dir]}"
REPO_DIR="${2:-.}"
IMAGE_TAG="support-ops-env-validate"
PYTHON_CMD=()

case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "[validate] Windows shell detected. Use validate-submission.ps1 from PowerShell for the most reliable local pre-validation run." >&2
    exit 1
    ;;
esac

resolve_python_cmd() {
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
    return 0
  fi
  if command -v py >/dev/null 2>&1; then
    PYTHON_CMD=(py -3)
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=(python3)
    return 0
  fi
  echo "[validate] could not find python, python3, or py on PATH" >&2
  return 1
}

resolve_openenv_cli() {
  if command -v openenv >/dev/null 2>&1; then
    command -v openenv
    return 0
  fi

  "${PYTHON_CMD[@]}" - <<'PY'
import os
import sysconfig

scripts = sysconfig.get_path("scripts") or ""
candidates = [
    os.path.join(scripts, "openenv"),
    os.path.join(scripts, "openenv.exe"),
]
for candidate in candidates:
    if os.path.exists(candidate):
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY

  echo "[validate] could not find the openenv CLI in this shell. Install openenv-core in the active environment before running this script." >&2
  return 1
}

run_openenv_validate() {
  if command -v openenv >/dev/null 2>&1; then
    openenv validate
    return 0
  fi

  OPENENV_CLI="$(resolve_openenv_cli)"
  "${OPENENV_CLI}" validate
}

resolve_python_cmd

echo "[validate] checking root endpoint: ${PING_URL}"
curl -fsS "${PING_URL}" >/dev/null

echo "[validate] checking reset endpoint"
curl -fsS -X POST "${PING_URL%/}/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"billing_seat_adjustment","seed":1}' >/dev/null

echo "[validate] running openenv validate"
(
  cd "${REPO_DIR}"
  run_openenv_validate
)

echo "[validate] building docker image"
(
  cd "${REPO_DIR}"
  docker build -t "${IMAGE_TAG}" .
)

echo "[validate] all checks passed"
