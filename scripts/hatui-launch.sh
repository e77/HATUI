#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_SCRIPT="${REPO_ROOT}/scripts/install_deps.sh"
APP="${REPO_ROOT}/ha_status_tui.py"

MODULES=("aiohttp" "yaml" "rich" "textual")

log() {
  printf "%s\n" "$*"
}

missing_modules=()
for module in "${MODULES[@]}"; do
  if ! python3 - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module}")
PY
  then
    missing_modules+=("${module}")
  fi
done

if [[ ${#missing_modules[@]} -gt 0 ]]; then
  log "HATUI preflight: missing Python modules: ${missing_modules[*]}"
  if [[ -x "${INSTALL_SCRIPT}" ]]; then
    log "Attempting dependency install via ${INSTALL_SCRIPT}..."
    if "${INSTALL_SCRIPT}"; then
      missing_modules=()
      for module in "${MODULES[@]}"; do
        if ! python3 - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module}")
PY
        then
          missing_modules+=("${module}")
        fi
      done
    fi
  else
    log "Install script not found: ${INSTALL_SCRIPT}"
  fi

  if [[ ${#missing_modules[@]} -gt 0 ]]; then
    log "HATUI preflight failed. Missing modules: ${missing_modules[*]}"
    log "Run ${INSTALL_SCRIPT} and restart the service."
    exit 0
  fi
fi

exec python3 "${APP}"
