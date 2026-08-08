#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="${1:-DamianoCeka/moevm-lab}"
VISIBILITY="${2:---private}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Run this helper from an existing Git repository." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  git status --short
  echo "The working tree is not clean. Commit or remove every change before publishing." >&2
  exit 1
fi

branch="$(git branch --show-current)"
if [[ "$branch" != "main" ]]; then
  echo "The publisher only releases main; current branch is '$branch'." >&2
  exit 1
fi

head_commit="$(git rev-parse HEAD)"
main_commit="$(git rev-parse refs/heads/main)"
if [[ "$head_commit" != "$main_commit" ]]; then
  echo "HEAD ($head_commit) and main ($main_commit) do not point to the same commit." >&2
  exit 1
fi

project_version="$(
  awk '
    /^\[project\][[:space:]]*$/ { in_project = 1; next }
    in_project && /^\[/ { exit }
    in_project && /^[[:space:]]*version[[:space:]]*=/ {
      line = $0
      sub(/^[^"]*"/, "", line)
      sub(/".*$/, "", line)
      print line
      exit
    }
  ' pyproject.toml
)"
if [[ -z "$project_version" ]]; then
  echo "Could not read [project].version from pyproject.toml." >&2
  exit 1
fi

release_tag="v${project_version}"
if ! git check-ref-format "refs/tags/$release_tag" >/dev/null; then
  echo "Release tag '$release_tag' is not a valid Git ref." >&2
  exit 1
fi
if ! git show-ref --verify --quiet "refs/tags/$release_tag"; then
  echo "Release tag '$release_tag' does not exist locally. Create it on the final release commit first." >&2
  exit 1
fi
tag_commit="$(git rev-parse "${release_tag}^{commit}")"
if [[ "$tag_commit" != "$head_commit" ]]; then
  echo "Release tag '$release_tag' points to $tag_commit, not HEAD $head_commit." >&2
  exit 1
fi

command -v gh >/dev/null 2>&1 || {
  echo "GitHub CLI is required. Install it and run: gh auth login" >&2
  exit 1
}
gh auth status

origin_url="$(git remote get-url origin 2>/dev/null || true)"
if [[ "$origin_url" == *.bundle ]]; then
  source_remote="source-bundle"
  if git remote | grep -Fxq "$source_remote"; then
    source_remote="source-bundle-$(date +%Y%m%d%H%M%S)"
  fi
  git remote rename origin "$source_remote"
  echo "Preserved the local bundle remote as '$source_remote'."
  origin_url=""
fi

expected_https="https://github.com/${REPOSITORY}"
expected_ssh="git@github.com:${REPOSITORY}"
normalized_origin="${origin_url%.git}"
if [[ -n "$origin_url" && "$normalized_origin" != "$expected_https" && "$normalized_origin" != "$expected_ssh" ]]; then
  echo "origin points to '$origin_url', not github.com/$REPOSITORY. Refusing to overwrite it." >&2
  exit 1
fi

if [[ -z "$origin_url" ]]; then
  gh repo create "$REPOSITORY" "$VISIBILITY" --source . --remote origin
fi

git push --set-upstream origin HEAD:main
git push origin "refs/tags/${release_tag}:refs/tags/${release_tag}"

echo "Published $REPOSITORY from $release_tag ($head_commit)."
