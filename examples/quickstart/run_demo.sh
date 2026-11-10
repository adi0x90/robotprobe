#!/usr/bin/env bash
# Thin wrapper kept for anyone who bookmarked this path. The real entry point is
# `robotprobe demo`, which works from an installed package as well as a checkout.
#
# Deliberately no arrays: `RUN=()` expanded as `"${RUN[@]}"` under `set -u` is an
# unbound-variable error on bash 3.2, which is stock /bin/bash on macOS
# (issue #35). Deliberately no bare `python`: stock macOS ships `python3`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-"$ROOT/examples/quickstart/out"}"

if command -v uv >/dev/null 2>&1; then
  exec uv run python -m robotprobe demo --out "$OUT"
fi

# Fall back to an interpreter that can actually run it. Probing `robotprobe`
# alone is not enough: from the repo root the package directory makes that import
# succeed while `mcap` is still missing, so probe a real runtime dependency too.
for CANDIDATE in python3 python; do
  if command -v "$CANDIDATE" >/dev/null 2>&1 &&
     "$CANDIDATE" -c "import robotprobe, mcap" >/dev/null 2>&1; then
    exec "$CANDIDATE" -m robotprobe demo --out "$OUT"
  fi
done

echo "run_demo.sh: no interpreter here can run robotprobe." >&2
echo "  Need uv, or a python3 with robotprobe and its dependencies installed." >&2
echo "  Fix: run 'make setup' (installs uv + the venv), or" >&2
echo "       'python3 -m pip install -e .' from the repo root." >&2
exit 1
