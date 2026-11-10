#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROBOTPROBE_GPU_SMOKE_OUT:-/tmp/robotprobe-gpu-smoke}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"

cd "$ROOT"

echo "== RobotProbe GPU smoke =="
echo "root=$ROOT"
echo "out=$OUT"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
elif [[ "$REQUIRE_GPU" == "1" ]]; then
  echo "nvidia-smi not found and REQUIRE_GPU=1" >&2
  exit 2
else
  echo "nvidia-smi not found; continuing because REQUIRE_GPU is not set"
fi

if command -v uv >/dev/null 2>&1; then
  uv sync --extra simulators
  RP=(uv run robotprobe)
  PY=(uv run python)
else
  python --version
  python -m pip install -e .
  RP=(python -m robotprobe)
  PY=(python)
fi

"${PY[@]}" --version

rm -rf "$OUT"
mkdir -p "$OUT"

"${RP[@]}" doctor
bash examples/quickstart/run_demo.sh "$OUT/quickstart"
"${RP[@]}" validate-trace examples/quickstart/traces/clean.jsonl
"${RP[@]}" pack claims examples/packs/libero-action-contracts > "$OUT/libero-action-contracts.claims.md"
"${RP[@]}" pack test examples/packs/libero-action-contracts

"${PY[@]}" examples/simulators/security_keepout_demo.py --out "$OUT/security-keepout"
"${RP[@]}" contracts check \
  --contract examples/packs/keepout-safety/contracts/keepout_margin.yaml \
  --trace "$OUT/security-keepout/traces/stealth_velocity_bias_keepout.jsonl" \
  > "$OUT/security-keepout/contract_check.json"
"${RP[@]}" export sidecar \
  --bundle "$OUT/security-keepout" \
  --out "$OUT/security-keepout/robotprobe.sidecar.json"
"${RP[@]}" export mcap \
  --bundle "$OUT/security-keepout" \
  --out "$OUT/security-keepout/robotprobe.mcap"
"${RP[@]}" import mcap \
  --mcap "$OUT/security-keepout/robotprobe.mcap" \
  --out "$OUT/security-keepout-restored"

"${RP[@]}" mech validate --capture-config examples/mechprobe/capture_config.yaml
"${RP[@]}" pack validate examples/packs/mechprobe-openvla-libero
"${RP[@]}" pack test examples/packs/mechprobe-openvla-libero --allow-code

cat > "$OUT/README.txt" <<EOF
RobotProbe GPU smoke passed.

Artifacts:
- $OUT/quickstart
- $OUT/libero-action-contracts.claims.md
- $OUT/security-keepout
- $OUT/security-keepout-restored

Next GPU-specific checks:
- vla-eval smoke: python scripts/vla_eval_robotprobe_smoke.py --out /tmp/robotprobe-vla-eval-smoke
- real model-server smoke: python scripts/vla_eval_real_model_smoke.py ...
- full official benchmark-container gate if Docker privileges are available
- real white-box MechProbe capture on a GPU host with an instrumentable policy
EOF

echo "wrote $OUT/README.txt"
