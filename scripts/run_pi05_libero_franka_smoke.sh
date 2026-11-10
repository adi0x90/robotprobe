#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PACK_DIR="${REPO_ROOT}/examples/packs/franka-pi05-runtime"
BASE_CONFIG="${PACK_DIR}/configs/pi05-libero.yaml"
OUT_DIR="${ROBOTPROBE_SMOKE_OUT:-${TMPDIR:-/tmp}/robotprobe-pi05-libero-smoke}"
SUITE="${ROBOTPROBE_LIBERO_SUITE:-libero_spatial}"
TASK_ID="${ROBOTPROBE_LIBERO_TASK_ID:-0}"
SEED="${ROBOTPROBE_LIBERO_SEED:-7}"
MAX_STEPS="${ROBOTPROBE_LIBERO_MAX_STEPS:-10}"
STEP_CAP="${ROBOTPROBE_LIBERO_STEP_CAP:-100}"
MIN_DISK_GB="${ROBOTPROBE_MIN_DISK_GB:-20}"
CAPTURE_STORAGE="${ROBOTPROBE_CAPTURE_STORAGE:-summary}"
CAPTURE_MAX_BYTES="${ROBOTPROBE_CAPTURE_MAX_BYTES:-67108864}"
LOCAL_FILES_ONLY="${ROBOTPROBE_LOCAL_FILES_ONLY:-0}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_GL

STATE_PATH="${OUT_DIR}/evidence-state.json"
RUNTIME_CONFIG="${OUT_DIR}/generated-runtime.yaml"
TRACE_PATH="${OUT_DIR}/pi05-libero-trace.jsonl"
RUNTIME_LOG="${OUT_DIR}/runtime-output.log"
TRACE_VALIDATION="${OUT_DIR}/trace-validation.json"
PACK_VALIDATION="${OUT_DIR}/pack-validation.json"
RUNTIME_EVIDENCE_DIR="${OUT_DIR}/runtime-evidence"
SMOKE_SUMMARY="${OUT_DIR}/smoke-summary.json"
PREFLIGHT_LOG="${OUT_DIR}/preflight.log"
CURRENT_STAGE="initialization"
FINAL_STATE_WRITTEN=0

mkdir -p "${OUT_DIR}"

json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "${value}"
}

write_state() {
  local status="$1"
  local reason="$2"
  local timestamp
  local task_id_json="${TASK_ID}"
  local seed_json="${SEED}"
  local max_steps_json="${MAX_STEPS}"
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  [[ "${task_id_json}" =~ ^[0-9]+$ ]] || task_id_json="null"
  [[ "${seed_json}" =~ ^[0-9]+$ ]] || seed_json="null"
  [[ "${max_steps_json}" =~ ^[0-9]+$ ]] || max_steps_json="null"
  printf '{\n' >"${STATE_PATH}"
  printf '  "format": "robotprobe.pi05_libero_smoke.evidence.v0",\n' >>"${STATE_PATH}"
  printf '  "status": "%s",\n' "$(json_escape "${status}")" >>"${STATE_PATH}"
  printf '  "reason": "%s",\n' "$(json_escape "${reason}")" >>"${STATE_PATH}"
  printf '  "stage": "%s",\n' "$(json_escape "${CURRENT_STAGE}")" >>"${STATE_PATH}"
  printf '  "suite": "%s",\n' "$(json_escape "${SUITE}")" >>"${STATE_PATH}"
  printf '  "task_id": %s,\n' "${task_id_json}" >>"${STATE_PATH}"
  printf '  "seed": %s,\n' "${seed_json}" >>"${STATE_PATH}"
  printf '  "max_steps": %s,\n' "${max_steps_json}" >>"${STATE_PATH}"
  printf '  "timestamp_utc": "%s"\n' "${timestamp}" >>"${STATE_PATH}"
  printf '}\n' >>"${STATE_PATH}"
}

finish_skipped() {
  write_state "skipped" "$1"
  FINAL_STATE_WRITTEN=1
  echo "PI0.5/LIBERO smoke skipped: $1"
  echo "Evidence state: ${STATE_PATH}"
  exit 0
}

finish_failed() {
  write_state "failed" "$1"
  FINAL_STATE_WRITTEN=1
  echo "PI0.5/LIBERO smoke failed: $1" >&2
  echo "Evidence state: ${STATE_PATH}" >&2
  exit 1
}

on_exit() {
  local exit_code="$?"
  if [[ "${exit_code}" -ne 0 && "${FINAL_STATE_WRITTEN}" -eq 0 ]]; then
    write_state "failed" "unexpected_failure"
  fi
}
trap on_exit EXIT

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    finish_failed "invalid_${name}"
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  require_nonnegative_integer "${name}" "${value}"
  if [[ "${value}" -eq 0 ]]; then
    finish_failed "invalid_${name}"
  fi
}

CURRENT_STAGE="argument_validation"
if [[ ! "${SUITE}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  finish_failed "invalid_suite"
fi
require_nonnegative_integer "task_id" "${TASK_ID}"
require_nonnegative_integer "seed" "${SEED}"
require_positive_integer "max_steps" "${MAX_STEPS}"
require_positive_integer "step_cap" "${STEP_CAP}"
require_positive_integer "min_disk_gb" "${MIN_DISK_GB}"
require_positive_integer "capture_max_bytes" "${CAPTURE_MAX_BYTES}"
if [[ "${MAX_STEPS}" -gt "${STEP_CAP}" ]]; then
  finish_failed "max_steps_exceeds_step_cap"
fi
if [[ "${CAPTURE_STORAGE}" != "summary" &&
      "${CAPTURE_STORAGE}" != "npy" &&
      "${CAPTURE_STORAGE}" != "safetensors" ]]; then
  finish_failed "invalid_capture_storage"
fi
if [[ "${LOCAL_FILES_ONLY}" != "0" && "${LOCAL_FILES_ONLY}" != "1" ]]; then
  finish_failed "invalid_local_files_only"
fi

CURRENT_STAGE="platform_preflight"
if [[ "$(uname -s)" != "Linux" ]]; then
  finish_skipped "missing_prerequisite:Linux"
fi
if ! command -v uv >/dev/null 2>&1; then
  finish_skipped "missing_prerequisite:uv"
fi
if [[ ! -w "${OUT_DIR}" ]]; then
  finish_failed "output_directory_not_writable"
fi

CURRENT_STAGE="disk_preflight"
AVAILABLE_KB="$(df -Pk "${OUT_DIR}" | awk 'NR == 2 {print $4}')"
REQUIRED_KB="$((MIN_DISK_GB * 1024 * 1024))"
if [[ -z "${AVAILABLE_KB}" || ! "${AVAILABLE_KB}" =~ ^[0-9]+$ ]]; then
  finish_failed "disk_capacity_unavailable"
fi
if [[ "${AVAILABLE_KB}" -lt "${REQUIRED_KB}" ]]; then
  finish_skipped "missing_prerequisite:disk_space"
fi

CURRENT_STAGE="torch_preflight"
if ! uv run python -c 'import torch' >"${PREFLIGHT_LOG}" 2>&1; then
  finish_skipped "missing_prerequisite:torch"
fi
if ! uv run python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
  >>"${PREFLIGHT_LOG}" 2>&1; then
  finish_skipped "missing_prerequisite:CUDA"
fi

CURRENT_STAGE="lerobot_preflight"
if ! uv run python -c \
  'from lerobot.envs.configs import LiberoEnv; from lerobot.envs.factory import make_env; from lerobot.policies.pi05 import PI05Policy' \
  >>"${PREFLIGHT_LOG}" 2>&1; then
  finish_skipped "missing_prerequisite:lerobot"
fi
if ! uv run python -c 'import libero' >>"${PREFLIGHT_LOG}" 2>&1; then
  finish_skipped "missing_prerequisite:libero"
fi
if ! uv run python -c 'import mujoco' >>"${PREFLIGHT_LOG}" 2>&1; then
  finish_skipped "missing_prerequisite:mujoco"
fi

CURRENT_STAGE="config_generation"
export ROBOTPROBE_SMOKE_BASE_CONFIG="${BASE_CONFIG}"
export ROBOTPROBE_SMOKE_PACK_DIR="${PACK_DIR}"
export ROBOTPROBE_SMOKE_RUNTIME_CONFIG="${RUNTIME_CONFIG}"
export ROBOTPROBE_SMOKE_TRACE_PATH="${TRACE_PATH}"
export ROBOTPROBE_SMOKE_OUT_DIR="${OUT_DIR}"
export ROBOTPROBE_SMOKE_SUITE="${SUITE}"
export ROBOTPROBE_SMOKE_TASK_ID="${TASK_ID}"
export ROBOTPROBE_SMOKE_SEED="${SEED}"
export ROBOTPROBE_SMOKE_MAX_STEPS="${MAX_STEPS}"
export ROBOTPROBE_SMOKE_CAPTURE_STORAGE="${CAPTURE_STORAGE}"
export ROBOTPROBE_SMOKE_CAPTURE_MAX_BYTES="${CAPTURE_MAX_BYTES}"
export ROBOTPROBE_SMOKE_LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY}"
uv run python - <<'PY'
import os
from pathlib import Path

import yaml

base = Path(os.environ["ROBOTPROBE_SMOKE_BASE_CONFIG"])
data = yaml.safe_load(base.read_text(encoding="utf-8"))
suite = os.environ["ROBOTPROBE_SMOKE_SUITE"]
task_id = int(os.environ["ROBOTPROBE_SMOKE_TASK_ID"])
seed = int(os.environ["ROBOTPROBE_SMOKE_SEED"])
max_steps = int(os.environ["ROBOTPROBE_SMOKE_MAX_STEPS"])
storage = os.environ["ROBOTPROBE_SMOKE_CAPTURE_STORAGE"]
out_dir = Path(os.environ["ROBOTPROBE_SMOKE_OUT_DIR"]).resolve()

data["pack"] = str(Path(os.environ["ROBOTPROBE_SMOKE_PACK_DIR"]).resolve())
data["components"]["policy"]["options"]["device"] = "cuda"
data["components"]["policy"]["options"]["local_files_only"] = (
    os.environ["ROBOTPROBE_SMOKE_LOCAL_FILES_ONLY"] == "1"
)
data["components"]["backend"]["options"].update(
    {
        "suite": suite,
        "task_id": task_id,
        "episode_length": max_steps,
    }
)
data["components"]["scenario"]["options"].update(
    {
        "suite": suite,
        "task_id": task_id,
        "seed": seed,
        "max_steps": max_steps,
    }
)
data["capture"]["profile"] = str(
    Path(os.environ["ROBOTPROBE_SMOKE_PACK_DIR"])
    / "capture"
    / "pi05-semantic.yaml"
)
data["capture"]["storage"] = storage
data["capture"]["max_bytes_per_run"] = int(
    os.environ["ROBOTPROBE_SMOKE_CAPTURE_MAX_BYTES"]
)
if storage == "summary":
    data["capture"].pop("output_dir", None)
else:
    data["capture"]["output_dir"] = str(out_dir / "activations")
data["trace"].update(
    {
        "out": str(Path(os.environ["ROBOTPROBE_SMOKE_TRACE_PATH"]).resolve()),
        "run_id": f"runtime:pi05-libero-smoke:{suite}:{task_id}:{seed}",
        "scenario": f"{suite}:task:{task_id}",
    }
)
runtime_path = Path(os.environ["ROBOTPROBE_SMOKE_RUNTIME_CONFIG"])
runtime_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

CURRENT_STAGE="runtime_inspection"
if ! uv run robotprobe runtime inspect --config "${RUNTIME_CONFIG}" \
  >"${OUT_DIR}/runtime-inspection.json" 2>"${OUT_DIR}/runtime-inspection.stderr"; then
  finish_failed "runtime_inspection_failed"
fi

CURRENT_STAGE="runtime_run"
if ! uv run robotprobe runtime run --config "${RUNTIME_CONFIG}" \
  >"${RUNTIME_LOG}" 2>&1; then
  finish_failed "runtime_run_failed"
fi

CURRENT_STAGE="trace_validation"
if ! uv run robotprobe validate-trace "${TRACE_PATH}" >"${TRACE_VALIDATION}" 2>&1; then
  finish_failed "trace_validation_failed"
fi
if ! uv run robotprobe pack validate "${PACK_DIR}" >"${PACK_VALIDATION}" 2>&1; then
  finish_failed "pack_validation_failed"
fi

CURRENT_STAGE="runtime_evidence"
if ! uv run robotprobe runtime evidence \
  --trace "${TRACE_PATH}" \
  --config "${RUNTIME_CONFIG}" \
  --out "${RUNTIME_EVIDENCE_DIR}" \
  >"${OUT_DIR}/runtime-evidence-output.json" \
  2>"${OUT_DIR}/runtime-evidence.stderr"; then
  finish_failed "runtime_evidence_failed"
fi
if [[ ! -f "${RUNTIME_EVIDENCE_DIR}/manifest.json" ]]; then
  finish_failed "runtime_evidence_manifest_missing"
fi

CURRENT_STAGE="trace_evidence_check"
if ! uv run python - "${TRACE_PATH}" "${SMOKE_SUMMARY}" "${RUNTIME_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

trace_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
runtime_config_path = Path(sys.argv[3])
runtime_config = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))
policy_options = runtime_config["components"]["policy"]["options"]
checkpoint = policy_options["checkpoint"]
checkpoint_revision = policy_options["revision"]
tokenizer_checkpoint = policy_options["tokenizer_checkpoint"]
tokenizer_revision = policy_options["tokenizer_revision"]
events = [
    json.loads(line)
    for line in trace_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
counts = {
    event_type: sum(event["event_type"] == event_type for event in events)
    for event_type in (
        "model_activation",
        "policy_inference",
        "action_chunk",
        "execution_receipt",
        "outcome",
    )
}
required = (
    "model_activation",
    "policy_inference",
    "action_chunk",
    "execution_receipt",
    "outcome",
)
missing = [event_type for event_type in required if counts[event_type] == 0]
completed_inference = sum(
    event["event_type"] == "policy_inference"
    and event.get("status") == "completed"
    for event in events
)
if completed_inference == 0:
    missing.append("completed_policy_inference")
activation_provenance_matches = any(
    event["event_type"] == "model_activation"
    and event.get("model_ref", {}).get("model_name") == checkpoint
    and event.get("model_ref", {}).get("revision") == checkpoint_revision
    and event.get("model_ref", {}).get("tokenizer_name") == tokenizer_checkpoint
    and event.get("model_ref", {}).get("tokenizer_revision")
    == tokenizer_revision
    for event in events
)
action_provenance_matches = any(
    event["event_type"] == "action_chunk"
    and event.get("metadata", {}).get("checkpoint") == checkpoint
    and event.get("metadata", {}).get("revision") == checkpoint_revision
    and event.get("metadata", {}).get("tokenizer_checkpoint")
    == tokenizer_checkpoint
    and event.get("metadata", {}).get("tokenizer_revision")
    == tokenizer_revision
    for event in events
)
if not activation_provenance_matches:
    missing.append("activation_checkpoint_provenance")
if not action_provenance_matches:
    missing.append("action_checkpoint_provenance")
outcome = next(
    (event for event in reversed(events) if event["event_type"] == "outcome"),
    {},
)
summary = {
    "format": "robotprobe.pi05_libero_smoke.summary.v0",
    "trace": str(trace_path),
    "event_count": len(events),
    "event_counts": counts,
    "completed_policy_inference_count": completed_inference,
    "checkpoint": checkpoint,
    "checkpoint_revision": checkpoint_revision,
    "tokenizer_checkpoint": tokenizer_checkpoint,
    "tokenizer_revision": tokenizer_revision,
    "activation_provenance_matches": activation_provenance_matches,
    "action_provenance_matches": action_provenance_matches,
    "task_success": bool(outcome.get("success", False)),
    "outcome_reason": outcome.get("reason"),
    "runtime_evidence_manifest": "runtime-evidence/manifest.json",
    "missing_required_evidence": missing,
}
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
raise SystemExit(1 if missing else 0)
PY
then
  finish_failed "required_trace_evidence_missing"
fi

CURRENT_STAGE="complete"
write_state "completed" "runtime_and_evidence_validation_completed"
FINAL_STATE_WRITTEN=1
echo "PI0.5/LIBERO smoke completed."
echo "Evidence state: ${STATE_PATH}"
echo "Summary: ${SMOKE_SUMMARY}"
