from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder


def ingest_vla_eval_episode(
    input_path: str | Path,
    out_path: str | Path,
    *,
    run_id: str,
    target: str,
    scenario: str,
    replay_level: ReplayLevel = ReplayLevel.TRACE_PLAYBACK,
) -> dict[str, Any]:
    source = Path(input_path)
    episode = _load_episode(source)
    steps = _extract_steps(episode)
    task_text = _extract_task_text(episode)
    success = bool(episode.get("success", episode.get("episode_success", False)))

    with TraceRecorder(
        out_path,
        run_id=run_id,
        target=target,
        scenario=scenario,
        replay_level=replay_level,
    ) as recorder:
        if task_text:
            recorder.event(0, "instruction", "instruction", value=task_text)
        for index, step in enumerate(steps):
            observation = _first_present(step, ("observation", "obs", "state"))
            if observation is not None:
                recorder.event(index, "observation", "observation", value=observation)
            raw_action = step.get("raw_action")
            action = _first_present(step, ("action", "actions"))
            postprocessed_action = _first_present(step, ("postprocessed_action", "post_action"))
            if raw_action is not None:
                recorder.event(index, "raw_action", "policy_forward", value=raw_action)
            elif action is not None:
                recorder.event(index, "raw_action", "policy_forward", value=action)
            if postprocessed_action is not None:
                recorder.event(index, "postprocessed_action", "action_postprocess", value=postprocessed_action)
            elif action is not None:
                recorder.event(index, "postprocessed_action", "action_postprocess", value=action)
        recorder.event(len(steps), "outcome", "outcome", success=success, steps=len(steps))

    return {
        "format": "robotprobe.trace.v0",
        "source_format": "vla-eval.episode",
        "input": str(source),
        "out": str(out_path),
        "episode_id": episode.get("episode_id") or episode.get("id"),
        "step_count": len(steps),
        "success": success,
    }


def _load_episode(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {"steps": rows, "success": _last_success(rows)}
    data = json.loads(text)
    if isinstance(data, list):
        return {"steps": data, "success": _last_success(data)}
    if not isinstance(data, dict):
        raise ValueError("vla-eval episode input must be a JSON object, list, or JSONL rows")
    return data


def _extract_steps(episode: dict[str, Any]) -> list[dict[str, Any]]:
    steps = episode.get("steps") or episode.get("trajectory") or episode.get("events") or []
    if not isinstance(steps, list):
        raise ValueError("vla-eval episode steps must be a list")
    return [step for step in steps if isinstance(step, dict)]


def _extract_task_text(episode: dict[str, Any]) -> str:
    for key in ("task_description", "instruction", "language_instruction", "prompt"):
        value = episode.get(key)
        if isinstance(value, str):
            return value
    task = episode.get("task")
    if isinstance(task, dict):
        for key in ("language_instruction", "task_description", "instruction", "name"):
            value = task.get(key)
            if isinstance(value, str):
                return value
    return ""


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _last_success(rows: list[dict[str, Any]]) -> bool:
    for row in reversed(rows):
        if "success" in row:
            return bool(row["success"])
        if "episode_success" in row:
            return bool(row["episode_success"])
    return False
