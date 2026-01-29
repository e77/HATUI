#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements.txt"

declare -A APT_PACKAGES=(
  [aiohttp]="python3-aiohttp"
  [yaml]="python3-yaml"
  [rich]="python3-rich"
  [textual]="python3-textual"
)

declare -A PIP_PACKAGES=(
  [aiohttp]="aiohttp"
  [yaml]="pyyaml"
  [rich]="rich"
  [textual]="textual"
)

MODULES=("aiohttp" "yaml" "rich" "textual")

log() {
  printf "%s\n" "$*"
}

have_module() {
  local module="$1"
  python3 - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module}")
PY
}

apt_available() {
  local package="$1"
  command -v apt-cache >/dev/null 2>&1 && apt-cache show "${package}" >/dev/null 2>&1
}

apt_installed() {
  local package="$1"
  dpkg -s "${package}" >/dev/null 2>&1
}

run_apt_install() {
  local package="$1"
  local sudo_cmd=()
  if [[ "${EUID}" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo_cmd=(sudo)
    else
      log "sudo not found; cannot install ${package} via apt."
      return 1
    fi
  fi
  "${sudo_cmd[@]}" apt-get update -y
  "${sudo_cmd[@]}" apt-get install -y "${package}"
}

ensure_pip() {
  if python3 -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  if apt_available python3-pip; then
    log "python3-pip not found; installing via apt."
    run_apt_install "python3-pip"
    return 0
  fi
  log "pip is not available. Please install python3-pip manually."
  return 1
}

missing_modules=()
for module in "${MODULES[@]}"; do
  if ! have_module "${module}"; then
    missing_modules+=("${module}")
  fi
done

if [[ ${#missing_modules[@]} -eq 0 ]]; then
  log "All dependencies already available."
  exit 0
fi

log "Missing Python modules: ${missing_modules[*]}"

for module in "${missing_modules[@]}"; do
  if have_module "${module}"; then
    continue
  fi
  apt_pkg="${APT_PACKAGES[${module}]}"
  if [[ -n "${apt_pkg}" ]] && apt_available "${apt_pkg}"; then
    if apt_installed "${apt_pkg}"; then
      log "Apt package already installed: ${apt_pkg}"
    else
      log "Installing ${module} via apt (${apt_pkg})..."
      if ! run_apt_install "${apt_pkg}"; then
        log "Failed to install ${apt_pkg} via apt."
      fi
    fi
  fi
  if have_module "${module}"; then
    continue
  fi
  if ensure_pip; then
    pip_pkg="${PIP_PACKAGES[${module}]}"
    if [[ -n "${pip_pkg}" ]]; then
      log "Installing ${module} via pip (${pip_pkg})..."
      python3 -m pip install --user "${pip_pkg}"
    fi
  fi
done

still_missing=()
for module in "${MODULES[@]}"; do
  if ! have_module "${module}"; then
    still_missing+=("${module}")
  fi
done

if [[ ${#still_missing[@]} -gt 0 ]]; then
  log "Still missing modules after install attempt: ${still_missing[*]}"
  if [[ -f "${REQUIREMENTS_FILE}" ]]; then
    log "You may also run: python3 -m pip install --user -r ${REQUIREMENTS_FILE}"
  fi
  exit 1
fi

log "Dependencies installed successfully."
