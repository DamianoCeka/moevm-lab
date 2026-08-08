#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="${1:-DamianoCeka/moevm-lab}"
VISIBILITY="${2:---private}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v gh >/dev/null 2>&1 || {
  echo "GitHub CLI is required. Install it and run: gh auth login" >&2
  exit 1
}
gh auth status

if [[ ! -d .git ]]; then
  git init -b main
  git add .
  git commit -m "Initialize MoEVM Lab v0.1"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$REPOSITORY" "$VISIBILITY" --source . --remote origin --push
else
  git push -u origin main
fi

echo "Published $REPOSITORY"
