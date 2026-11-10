from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder


ACTION_KEYS = ("action", "actions", "policy_action", "applied_action")
STATE_KEYS = ("state", "sim_state", "observation", "obs")


def ingest_isaac_lab_arena_trace(
    input_path: str | Path,
    out_path: str | Path,
    *,
    run_id: str,
    target: str,
    scenario: str,
    replay_level: ReplayLevel = ReplayLevel.TRACE_PLAYBACK,
) -> dict[str, Any]:
    source = Path(input_path)
    payload = _load_rows(source)
    rows = payload["rows"]
    success = bool(payload.get("success", _last_success(rows)))

    with TraceRecorder(
        out_path,
        run_id=run_id,
        target=target,
        scenario=scenario,
        replay_level=replay_level,
    ) as recorder:
        for index, row in enumerate(rows):
            step = _step(row, index)
            state = _state(row)
            if state is not None:
                recorder.event(step, "sim_state", "simulator", value=state)
            action = _first_present(row, ACTION_KEYS)
            if action is not None:
                recorder.event(step, "postprocessed_action", "sim_action", value=action)
        recorder.event(len(rows), "outcome", "outcome", success=success, steps=len(rows))

    return {
        "format": "robotprobe.trace.v0",
        "source_format": "isaac-lab-arena.trace",
        "input": str(source),
        "out": str(out_path),
        "step_count": len(rows),
        "success": success,
    }


def _load_rows(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return {
            "rows": [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        }
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return {"rows": [{key: _parse_cell(value) for key, value in row.items()} for row in csv.DictReader(f)]}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"rows": data}
    if isinstance(data, dict):
        rows = data.get("steps") or data.get("trajectory") or data.get("events") or []
        if not isinstance(rows, list):
            raise ValueError("Isaac Lab-Arena steps must be a list")
        return {"rows": [row for row in rows if isinstance(row, dict)], "success": data.get("success")}
    raise ValueError("Isaac Lab-Arena input must be JSON, JSONL, or CSV rows")


def _step(row: dict[str, Any], fallback: int) -> int:
    value = row.get("step", fallback)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _state(row: dict[str, Any]) -> Any | None:
    explicit = _first_present(row, STATE_KEYS)
    if explicit is not None:
        return explicit
    state = {
        key: value
        for key, value in row.items()
        if key not in set(ACTION_KEYS) | {"step", "success", "done", "terminated", "truncated", "reward"}
    }
    return state or None


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _last_success(rows: list[dict[str, Any]]) -> bool:
    for row in reversed(rows):
        for key in ("success", "done", "terminated"):
            if key in row:
                return _as_bool(row[key])
    return False


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_cell(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped
