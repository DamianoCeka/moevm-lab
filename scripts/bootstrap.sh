#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
SITE_PACKAGES="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$ROOT/src" > "$SITE_PACKAGES/moevm_lab.pth"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m moevm compare --config configs/toy.toml --tokens 64 --output-dir results/toy

echo "MoEVM Lab is ready."
echo "Run: .venv/bin/python -m moevm --help"
echo "Optional standard install: .venv/bin/python -m pip install -e ."
