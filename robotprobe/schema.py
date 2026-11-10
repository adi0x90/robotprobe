from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robotprobe.core import ReplayLevel
from robotprobe.runtime.types import ExecutionMode


BASE_REQUIRED_FIELDS = ("run_id", "target", "scenario", "replay_level", "step", "event_type", "layer")
EVENT_REQUIRED_FIELDS = {
    "raw_action": ("value",),
    "postprocessed_action": ("value",),
    "subgoal_request": ("request_ref",),
    "world_model_output": ("generated_image_ref", "request_ref"),
    "model_activation": ("module", "module_kind", "activation_ref", "activation_shape", "activation_dtype", "model_ref", "capture"),
    "activation_capture_dropped": ("reason",),
    "policy_inference": ("status", "policy_call_id", "action_chunk_id", "observation_id"),
    "action_chunk": ("chunk_id", "policy_call_id", "source_observation_id", "command_space", "shape", "value"),
    "scheduled_action": ("command_id", "chunk_id", "policy_call_id", "action_index", "command_space", "value"),
    "safety_decision": ("command_id", "accepted", "disposition", "reason"),
    "execution_receipt": (
        "command_id",
        "backend_id",
        "execution_mode",
        "command_applied",
        "accepted",
    ),
    "runtime_error": ("phase", "component_id", "error_type"),
    "outcome": ("success", "steps"),
    "contract_check": ("contract", "check", "status", "severity", "evidence_refs"),
}
KNOWN_EVENT_TYPES = {
    "instruction",
    "observation",
    "policy_input",
    "subgoal_request",
    "world_model_output",
    "model_activation",
    "activation_capture_dropped",
    "policy_inference",
    "action_chunk",
    "scheduled_action",
    "safety_decision",
    "execution_receipt",
    "runtime_error",
    "raw_action",
    "postprocessed_action",
    "outcome",
    "contract_check",
    "robot_state",
    "sim_state",
}


@dataclass(frozen=True)
class TraceIssue:
    line: int
    field: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class TraceValidationReport:
    path: str
    valid: bool
    event_count: int
    run_ids: list[str] = field(default_factory=list)
    replay_levels: list[str] = field(default_factory=list)
    issues: list[TraceIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def validate_trace_file(path: str | Path) -> TraceValidationReport:
    trace_path = Path(path)
    issues: list[TraceIssue] = []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(TraceIssue(line_number, "$", f"invalid json: {exc.msg}"))
            continue
        if not isinstance(event, dict):
            issues.append(TraceIssue(line_number, "$", "event must be a JSON object"))
            continue
        events.append(event)
        issues.extend(_validate_event(event, line_number))

    run_ids = sorted({str(event["run_id"]) for event in events if "run_id" in event})
    replay_levels = sorted({str(event["replay_level"]) for event in events if "replay_level" in event})
    if len(run_ids) > 1:
        issues.append(TraceIssue(0, "run_id", f"trace contains multiple run_ids: {run_ids}"))

    return TraceValidationReport(
        path=str(trace_path),
        valid=not any(issue.severity == "error" for issue in issues),
        event_count=len(events),
        run_ids=run_ids,
        replay_levels=replay_levels,
        issues=issues,
    )


def validate_trace_events(events: list[dict[str, Any]], *, path: str = "<memory>") -> TraceValidationReport:
    issues: list[TraceIssue] = []
    for index, event in enumerate(events, start=1):
        issues.extend(_validate_event(event, index))
    run_ids = sorted({str(event["run_id"]) for event in events if "run_id" in event})
    replay_levels = sorted({str(event["replay_level"]) for event in events if "replay_level" in event})
    return TraceValidationReport(
        path=path,
        valid=not any(issue.severity == "error" for issue in issues),
        event_count=len(events),
        run_ids=run_ids,
        replay_levels=replay_levels,
        issues=issues,
    )


def _validate_event(event: dict[str, Any], line_number: int) -> list[TraceIssue]:
    issues: list[TraceIssue] = []
    for field_name in BASE_REQUIRED_FIELDS:
        if field_name not in event:
            issues.append(TraceIssue(line_number, field_name, f"missing required field: {field_name}"))

    event_type = event.get("event_type")
    if event_type in EVENT_REQUIRED_FIELDS:
        for field_name in EVENT_REQUIRED_FIELDS[event_type]:
            if field_name not in event:
                issues.append(TraceIssue(line_number, field_name, f"missing required field: {field_name}"))
    elif event_type is not None and event_type not in KNOWN_EVENT_TYPES:
        issues.append(TraceIssue(line_number, "event_type", f"unknown event_type: {event_type}", severity="warning"))

    if event_type == "instruction" and not any(field in event for field in ("value", "before", "after")):
        issues.append(TraceIssue(line_number, "value", "instruction event needs value, before, or after"))
    if event_type == "observation" and not any(field in event for field in ("value", "image_sha256", "observation_ref")):
        issues.append(TraceIssue(line_number, "observation", "observation event needs value, image_sha256, or observation_ref"))
    if event_type == "model_activation":
        if not isinstance(event.get("activation_ref"), str):
            issues.append(TraceIssue(line_number, "activation_ref", "activation_ref must be a string"))
        activation_shape = event.get("activation_shape")
        if not isinstance(activation_shape, list) or not all(isinstance(value, int) for value in activation_shape):
            issues.append(TraceIssue(line_number, "activation_shape", "activation_shape must be a list of integers"))
        if not isinstance(event.get("model_ref"), dict):
            issues.append(TraceIssue(line_number, "model_ref", "model_ref must be an object"))
        if not isinstance(event.get("capture"), dict):
            issues.append(TraceIssue(line_number, "capture", "capture must be an object"))
    if event_type == "activation_capture_dropped" and not isinstance(event.get("reason"), str):
        issues.append(TraceIssue(line_number, "reason", "reason must be a string"))
    if event_type == "policy_inference":
        if event.get("status") not in {"started", "completed"}:
            issues.append(
                TraceIssue(
                    line_number,
                    "status",
                    "policy inference status must be started or completed",
                )
            )
    if event_type == "action_chunk":
        shape = event.get("shape")
        if not isinstance(shape, list) or not all(isinstance(value, int) for value in shape):
            issues.append(TraceIssue(line_number, "shape", "shape must be a list of integers"))
        if not isinstance(event.get("value"), list):
            issues.append(TraceIssue(line_number, "value", "action chunk value must be a list"))
    if event_type == "scheduled_action" and not isinstance(event.get("action_index"), int):
        issues.append(TraceIssue(line_number, "action_index", "action_index must be an integer"))
    if event_type == "safety_decision" and not isinstance(event.get("accepted"), bool):
        issues.append(TraceIssue(line_number, "accepted", "accepted must be boolean"))
    if event_type == "execution_receipt":
        if event.get("execution_mode") not in {mode.value for mode in ExecutionMode}:
            issues.append(
                TraceIssue(line_number, "execution_mode", "unknown execution mode")
            )
        for field_name in ("command_applied", "accepted"):
            if not isinstance(event.get(field_name), bool):
                issues.append(
                    TraceIssue(line_number, field_name, f"{field_name} must be boolean")
                )
    if event_type == "runtime_error":
        for field_name in ("phase", "component_id", "error_type"):
            if not isinstance(event.get(field_name), str):
                issues.append(
                    TraceIssue(line_number, field_name, f"{field_name} must be a string")
                )

    replay_level = event.get("replay_level")
    if replay_level is not None and replay_level not in {level.value for level in ReplayLevel}:
        issues.append(TraceIssue(line_number, "replay_level", f"unknown replay_level: {replay_level}"))

    step = event.get("step")
    if step is not None and not isinstance(step, int):
        issues.append(TraceIssue(line_number, "step", "step must be an integer"))

    return issues
