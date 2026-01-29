#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "Error: not inside a git repository." >&2
  exit 1
}

cd "$repo_root"

config_file=${HATUI_GIT_CONFIG:-config/hatui_git.json}
remote_url=${HATUI_GIT_REMOTE:-}

if [[ -z "$remote_url" && -f "$config_file" ]]; then
  remote_url=$(python - <<'PY' "$config_file"
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    print(data.get("remote_url", ""))
except Exception:
    pass
PY
  )
fi

if [[ -z "$remote_url" ]]; then
  echo "Error: set HATUI_GIT_REMOTE or add remote_url to $config_file." >&2
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$remote_url"
else
  git remote add origin "$remote_url"
fi

git fetch origin --prune

branch=$(git rev-parse --abbrev-ref HEAD)
if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
  git branch --set-upstream-to "origin/$branch" "$branch"
else
  echo "Error: origin/$branch does not exist. Push with: git push -u origin $branch" >&2
  exit 1
fi

repo_owner=$(stat -c %u "$repo_root")
current_uid=$(id -u)
if [[ "$current_uid" -eq 0 || "$current_uid" -ne "$repo_owner" ]]; then
  git config --global --add safe.directory "$repo_root"
fi

echo "Git integration configured for $repo_root on branch $branch."
