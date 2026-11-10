from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robotprobe.schema import validate_trace_file
from robotprobe.runtime.config import inspect_runtime_config


RUNTIME_EVIDENCE_FORMAT = "robotprobe.runtime.evidence.v0"


def analyze_runtime_trace(
    trace_path: str | Path,
    *,
    config_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(trace_path)
    events = _read_events(path)
    validation = validate_trace_file(path)
    outcomes = [event for event in events if event.get("event_type") == "outcome"]
    runtime_errors = [
        event for event in events if event.get("event_type") == "runtime_error"
    ]
    issues: list[dict[str, Any]] = [
        {
            "code": "trace_schema_error",
            "line": issue.line,
            "field": issue.field,
            "message": issue.message,
        }
        for issue in validation.issues
        if issue.severity == "error"
    ]
    if len(outcomes) != 1:
        issues.append(
            {
                "code": "outcome_count",
                "message": f"expected exactly one outcome, found {len(outcomes)}",
            }
        )
    for event in runtime_errors:
        issues.append(
            {
                "code": "runtime_error",
                "phase": event.get("phase"),
                "component_id": event.get("component_id"),
                "message": "runtime trace contains an error event",
            }
        )
    issues.extend(_lineage_issues(events))
    issues.extend(_runtime_invariant_issues(events))

    outcome = outcomes[-1] if outcomes else {}
    receipts = [
        event for event in events if event.get("event_type") == "execution_receipt"
    ]
    decisions = [
        event for event in events if event.get("event_type") == "safety_decision"
    ]
    rejected = [event for event in decisions if event.get("accepted") is False]
    applied_commands = sum(event.get("command_applied") is True for event in receipts)
    execution_mode = _execution_mode(outcome, receipts)
    motion_disposition = _motion_disposition(
        receipts=receipts,
        rejected=rejected,
        applied_commands=applied_commands,
        execution_mode=execution_mode,
    )

    evidence = {
        "format": RUNTIME_EVIDENCE_FORMAT,
        "run": _run_summary(events, outcome, validation.event_count, execution_mode),
        "runtime_integrity": {
            "status": "passed" if not issues else "failed",
            "trace_valid": validation.valid,
            "issues": issues,
        },
        "task_outcome": _task_outcome(outcome, execution_mode),
        "motion": {
            "disposition": motion_disposition,
            "action_chunks": _count(events, "action_chunk"),
            "scheduled_actions": _count(events, "scheduled_action"),
            "safety_decisions": len(decisions),
            "authorized_decisions": len(decisions) - len(rejected),
            "rejected_decisions": len(rejected),
            "execution_receipts": len(receipts),
            "applied_commands": applied_commands,
            "first_rejection": (
                {
                    "step": rejected[0].get("step"),
                    "command_id": rejected[0].get("command_id"),
                    "reason": rejected[0].get("reason"),
                }
                if rejected
                else None
            ),
        },
        "instrumentation": _instrumentation_summary(events),
        "claims": _claims(motion_disposition, execution_mode),
    }
    if baseline_path is not None:
        evidence["comparison"] = _compare_events(
            _read_events(Path(baseline_path)),
            events,
        )
    if config_path is not None:
        evidence["configuration"] = _configuration_summary(Path(config_path))
    return evidence


def create_runtime_evidence_bundle(
    *,
    trace_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
) -> dict[str, Any]:
    trace = Path(trace_path)
    config = Path(config_path) if config_path is not None else None
    baseline = Path(baseline_path) if baseline_path is not None else None
    for label, path in (
        ("trace", trace),
        ("config", config),
        ("baseline", baseline),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} input does not exist: {path}")

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"runtime evidence output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "inputs"
    inputs.mkdir()

    copied_trace = inputs / "trace.jsonl"
    shutil.copy2(trace, copied_trace)
    copied_config = None
    if config is not None:
        copied_config = inputs / "runtime-config.yaml"
        shutil.copy2(config, copied_config)
    copied_baseline = None
    if baseline is not None:
        copied_baseline = inputs / "baseline.jsonl"
        shutil.copy2(baseline, copied_baseline)

    evidence = analyze_runtime_trace(
        copied_trace,
        config_path=config,
        baseline_path=copied_baseline,
    )
    evidence_path = out / "evidence.json"
    evidence_path.write_text(_canonical_json(evidence), encoding="utf-8")
    report_path = out / "report.md"
    report_path.write_text(render_runtime_report(evidence), encoding="utf-8")
    timeline_path = out / "timeline.md"
    timeline_path.write_text(
        render_runtime_timeline(_read_events(copied_trace), evidence),
        encoding="utf-8",
    )

    files = [
        {
            "path": path.relative_to(out).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(
            (path for path in out.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(out).as_posix(),
        )
    ]
    manifest = {
        "format": "robotprobe.runtime.evidence_bundle.v0",
        "runtime_integrity": evidence["runtime_integrity"]["status"],
        "evidence_ref": "evidence.json",
        "report_ref": "report.md",
        "timeline_ref": "timeline.md",
        "trace_ref": copied_trace.relative_to(out).as_posix(),
        "baseline_ref": (
            copied_baseline.relative_to(out).as_posix()
            if copied_baseline is not None
            else None
        ),
        "config_ref": (
            copied_config.relative_to(out).as_posix()
            if copied_config is not None
            else None
        ),
        "files": files,
    }
    (out / "manifest.json").write_text(
        _canonical_json(manifest),
        encoding="utf-8",
    )
    return manifest


def render_runtime_report(evidence: Mapping[str, Any]) -> str:
    integrity = evidence.get("runtime_integrity", {})
    task = evidence.get("task_outcome", {})
    motion = evidence.get("motion", {})
    instrumentation = evidence.get("instrumentation", {})
    run = evidence.get("run", {})
    claims = evidence.get("claims", {})
    integrity_label = (
        "PASS" if integrity.get("status") == "passed" else "FAIL"
    )
    lines = [
        "# RobotProbe Runtime Evidence",
        "",
        "## Verdict",
        "",
        f"- Runtime integrity: `{integrity_label}`",
        f"- Task outcome: `{_display(task.get('status'))}`",
        f"- Motion disposition: `{_display(motion.get('disposition'))}`",
        "",
        "## Run",
        "",
        f"- Run ID: `{_display(run.get('run_id'))}`",
        f"- Target: `{_display(run.get('target'))}`",
        f"- Scenario: `{_display(run.get('scenario'))}`",
        f"- Replay level: `{_display(run.get('replay_level'))}`",
        f"- Execution mode: `{_display(run.get('execution_mode'))}`",
        f"- Events: `{_display(run.get('event_count'))}`",
        "",
        "## Motion and Safety",
        "",
        f"- Action chunks: `{_display(motion.get('action_chunks'))}`",
        f"- Scheduled actions: `{_display(motion.get('scheduled_actions'))}`",
        f"- Safety rejections: `{_display(motion.get('rejected_decisions'))}`",
        f"- Execution receipts: `{_display(motion.get('execution_receipts'))}`",
        f"- Applied commands: `{_display(motion.get('applied_commands'))}`",
        f"- Outcome reason: `{_display(task.get('reason'))}`",
        "",
        "## Instrumentation",
        "",
        f"- Activation events: `{_display(instrumentation.get('activation_events'))}`",
        f"- Dropped captures: `{_display(instrumentation.get('dropped_events'))}`",
        f"- Modules: `{_display_list(instrumentation.get('modules'))}`",
        f"- Probes: `{_display_list(instrumentation.get('probes'))}`",
        "- Causal effect established: `false`",
    ]
    comparison = evidence.get("comparison")
    if isinstance(comparison, Mapping):
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                "- First action divergence step: "
                f"`{_display(comparison.get('first_action_divergence_step'))}`",
                "- Safety decision changed: "
                f"`{_bool_text(comparison.get('safety_decision_changed'))}`",
                "- Execution receipt delta: "
                f"`{_display(comparison.get('execution_receipt_delta'))}`",
                "- Applied command delta: "
                f"`{_display(comparison.get('applied_command_delta'))}`",
                f"- Outcome changed: `{_bool_text(comparison.get('outcome_changed'))}`",
            ]
        )
    lines.extend(["", "## What This Proves", ""])
    lines.extend(
        f"- {item}" for item in claims.get("proves", []) if isinstance(item, str)
    )
    lines.extend(["", "## What This Does Not Prove", ""])
    lines.extend(
        f"- {item}"
        for item in claims.get("does_not_prove", [])
        if isinstance(item, str)
    )
    return "\n".join(lines) + "\n"


def render_runtime_timeline(
    events: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> str:
    lines = [
        "# Runtime Action Timeline",
        "",
        f"- Runtime integrity: `{_display(evidence.get('runtime_integrity', {}).get('status'))}`",
        f"- Task outcome: `{_display(evidence.get('task_outcome', {}).get('status'))}`",
        f"- Motion disposition: `{_display(evidence.get('motion', {}).get('disposition'))}`",
        "",
        "| Step | Event | Identity | Detail |",
        "| ---: | --- | --- | --- |",
    ]
    activation_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for event in events:
        if event.get("event_type") == "model_activation":
            key = (
                _display(event.get("step", 0)),
                str(event.get("policy_call_id", "unknown")),
            )
            activation_groups.setdefault(key, []).append(event)
    emitted_activation_groups: set[tuple[str, str]] = set()
    for event in events:
        if event.get("event_type") != "model_activation":
            lines.append(_timeline_row(event))
            continue
        key = (
            _display(event.get("step", 0)),
            str(event.get("policy_call_id", "unknown")),
        )
        if key in emitted_activation_groups:
            continue
        emitted_activation_groups.add(key)
        lines.append(_activation_timeline_row(key, activation_groups[key]))
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "This timeline explains recorded runtime events. It does not by itself prove "
            "physical safety, physical task success, or causal meaning for activations.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _count(events: list[dict[str, Any]], event_type: str) -> int:
    return sum(event.get("event_type") == event_type for event in events)


def _execution_mode(
    outcome: dict[str, Any],
    receipts: list[dict[str, Any]],
) -> str | None:
    mode = outcome.get("execution_mode")
    if isinstance(mode, str):
        return mode
    for receipt in receipts:
        value = receipt.get("execution_mode")
        if isinstance(value, str):
            return value
    return None


def _motion_disposition(
    *,
    receipts: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    applied_commands: int,
    execution_mode: str | None,
) -> str:
    if applied_commands:
        return "applied"
    if receipts and execution_mode == "hardware_shadow":
        return "shadow_evaluated"
    if receipts and any(receipt.get("accepted") is True for receipt in receipts):
        return "evaluated_not_applied"
    if rejected and not receipts:
        return "rejected_pre_backend"
    return "none"


def _task_outcome(
    outcome: dict[str, Any],
    execution_mode: str | None,
) -> dict[str, Any]:
    if not outcome:
        status = "missing"
    elif execution_mode == "hardware_shadow":
        status = "not_assessed"
    else:
        status = "succeeded" if outcome.get("success") is True else "failed"
    return {
        "status": status,
        "success": outcome.get("success"),
        "reason": outcome.get("reason"),
        "steps": outcome.get("steps"),
    }


def _run_summary(
    events: list[dict[str, Any]],
    outcome: dict[str, Any],
    event_count: int,
    execution_mode: str | None,
) -> dict[str, Any]:
    first = events[0] if events else {}
    return {
        "run_id": first.get("run_id"),
        "target": first.get("target"),
        "scenario": first.get("scenario"),
        "replay_level": first.get("replay_level"),
        "execution_mode": execution_mode,
        "event_count": event_count,
        "outcome_step": outcome.get("step"),
    }


def _claims(motion_disposition: str, execution_mode: str | None) -> dict[str, list[str]]:
    proves = ["the recorded runtime trace can be checked for schema and execution invariants"]
    if motion_disposition == "rejected_pre_backend":
        proves.append("the recorded action was rejected before a backend execution receipt")
    if motion_disposition == "shadow_evaluated":
        proves.append("the recorded action was evaluated without a claimed applied command")
    if motion_disposition == "applied" and execution_mode == "simulation":
        proves.append("the simulator backend recorded an applied command")
    return {
        "proves": proves,
        "does_not_prove": [
            "physical robot safety",
            "physical task success",
            "causal meaning of captured activations",
        ],
    }


def _lineage_issues(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    issues.extend(_lineage_identifier_issues(events))
    issues.extend(_duplicate_lineage_issues(events))
    observations = {
        identifier
        for event in events
        if event.get("event_type") == "observation"
        for identifier in (
            _identifier(event.get("observation_id")),
            _identifier(event.get("observation_ref")),
        )
        if identifier is not None
    }
    completed_policy_calls = {
        identifier
        for event in events
        if event.get("event_type") == "policy_inference"
        and event.get("status") == "completed"
        for identifier in (_identifier(event.get("policy_call_id")),)
        if identifier is not None
    }
    chunks = {
        identifier: event
        for event in events
        if event.get("event_type") == "action_chunk"
        for identifier in (_identifier(event.get("chunk_id")),)
        if identifier is not None
    }
    scheduled = {
        identifier: event
        for event in events
        if event.get("event_type") == "scheduled_action"
        for identifier in (_identifier(event.get("command_id")),)
        if identifier is not None
    }
    authorized = {
        identifier: event
        for event in events
        if event.get("event_type") == "safety_decision"
        and event.get("accepted") is True
        for identifier in (_identifier(event.get("command_id")),)
        if identifier is not None
    }

    for chunk_id, chunk in chunks.items():
        raw_policy_call_id = chunk.get("policy_call_id")
        raw_observation_id = chunk.get("source_observation_id")
        policy_call_id = _identifier(raw_policy_call_id)
        observation_id = _identifier(raw_observation_id)
        if policy_call_id not in completed_policy_calls:
            issues.append(
                _lineage_issue(
                    "unknown_policy_call",
                    "action_chunk",
                    chunk_id,
                    "policy_call_id",
                    raw_policy_call_id,
                )
            )
        if observation_id not in observations:
            issues.append(
                _lineage_issue(
                    "unknown_observation",
                    "action_chunk",
                    chunk_id,
                    "source_observation_id",
                    raw_observation_id,
                )
            )

    for command_id, action in scheduled.items():
        raw_chunk_id = action.get("chunk_id")
        chunk_id = _identifier(raw_chunk_id)
        chunk = chunks.get(chunk_id)
        if chunk is None:
            issues.append(
                _lineage_issue(
                    "unknown_action_chunk",
                    "scheduled_action",
                    command_id,
                    "chunk_id",
                    raw_chunk_id,
                )
            )
            continue
        for field_name, chunk_field in (
            ("policy_call_id", "policy_call_id"),
            ("source_observation_id", "source_observation_id"),
        ):
            if action.get(field_name) != chunk.get(chunk_field):
                issues.append(
                    _lineage_issue(
                        f"scheduled_{field_name}_mismatch",
                        "scheduled_action",
                        command_id,
                        field_name,
                        action.get(field_name),
                    )
                )

    for decision in (
        event for event in events if event.get("event_type") == "safety_decision"
    ):
        raw_command_id = decision.get("command_id")
        command_id = _identifier(raw_command_id)
        if command_id not in scheduled:
            issues.append(
                _lineage_issue(
                    "unknown_scheduled_command",
                    "safety_decision",
                    raw_command_id,
                    "command_id",
                    raw_command_id,
                )
            )

    for receipt in (
        event for event in events if event.get("event_type") == "execution_receipt"
    ):
        raw_command_id = receipt.get("command_id")
        command_id = _identifier(raw_command_id)
        if command_id not in authorized:
            issues.append(
                _lineage_issue(
                    "unknown_authorized_command",
                    "execution_receipt",
                    raw_command_id,
                    "command_id",
                    raw_command_id,
                )
            )

    for activation in (
        event for event in events if event.get("event_type") == "model_activation"
    ):
        raw_chunk_id = activation.get("action_chunk_id")
        chunk_id = _identifier(raw_chunk_id)
        chunk = chunks.get(chunk_id)
        if chunk is None:
            issues.append(
                _lineage_issue(
                    "unknown_activation_action_chunk",
                    "model_activation",
                    activation.get("activation_ref"),
                    "action_chunk_id",
                    raw_chunk_id,
                )
            )
        elif activation.get("policy_call_id") != chunk.get("policy_call_id"):
            issues.append(
                _lineage_issue(
                    "activation_policy_call_mismatch",
                    "model_activation",
                    activation.get("activation_ref"),
                    "policy_call_id",
                    activation.get("policy_call_id"),
                )
            )
    return issues


def _lineage_identifier_issues(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields_by_event = {
        "observation": ("observation_id", "observation_ref"),
        "policy_inference": (
            "policy_call_id",
            "action_chunk_id",
            "observation_id",
        ),
        "action_chunk": (
            "chunk_id",
            "policy_call_id",
            "source_observation_id",
        ),
        "scheduled_action": (
            "command_id",
            "chunk_id",
            "policy_call_id",
            "source_observation_id",
        ),
        "safety_decision": (
            "command_id",
            "chunk_id",
            "policy_call_id",
            "source_observation_id",
        ),
        "execution_receipt": ("command_id",),
        "model_activation": ("policy_call_id", "action_chunk_id"),
    }
    issues: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("event_type")
        fields = (
            fields_by_event.get(event_type, ())
            if isinstance(event_type, str)
            else ()
        )
        for field in fields:
            if field not in event:
                continue
            value = event.get(field)
            if _identifier(value) is None:
                issues.append(
                    {
                        "code": "invalid_lineage_identifier",
                        "event_type": event_type,
                        "field": field,
                        "value": value,
                        "message": (
                            f"{event_type} {field} must be a non-empty string"
                        ),
                    }
                )
    return issues


def _duplicate_lineage_issues(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    specifications = (
        ("observation", "observation_id", "duplicate_observation_id"),
        ("action_chunk", "chunk_id", "duplicate_action_chunk_id"),
        (
            "scheduled_action",
            "command_id",
            "duplicate_scheduled_command_id",
        ),
        (
            "safety_decision",
            "command_id",
            "duplicate_safety_decision_command_id",
        ),
        (
            "execution_receipt",
            "command_id",
            "duplicate_execution_receipt_command_id",
        ),
    )
    issues: list[dict[str, Any]] = []
    for event_type, field, code in specifications:
        values = [
            event.get(field)
            for event in events
            if event.get("event_type") == event_type
            and _identifier(event.get(field)) is not None
        ]
        for value, count in sorted(
            Counter(values).items(),
            key=lambda item: str(item[0]),
        ):
            if count > 1:
                issues.append(
                    {
                        "code": code,
                        "event_type": event_type,
                        "field": field,
                        "value": value,
                        "count": count,
                        "message": (
                            f"{event_type} repeats lineage identifier {field}"
                        ),
                    }
                )

    completed_policy_calls = [
        event.get("policy_call_id")
        for event in events
        if event.get("event_type") == "policy_inference"
        and event.get("status") == "completed"
        and _identifier(event.get("policy_call_id")) is not None
    ]
    for value, count in sorted(
        Counter(completed_policy_calls).items(),
        key=lambda item: str(item[0]),
    ):
        if count > 1:
            issues.append(
                {
                    "code": "duplicate_completed_policy_call_id",
                    "event_type": "policy_inference",
                    "field": "policy_call_id",
                    "value": value,
                    "count": count,
                    "message": (
                        "policy_inference repeats a completed policy_call_id"
                    ),
                }
            )
    return issues


def _identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _runtime_invariant_issues(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    receipts = [
        event
        for event in events
        if event.get("event_type") == "execution_receipt"
    ]
    for receipt in receipts:
        command_id = receipt.get("command_id")
        if (
            receipt.get("execution_mode") == "hardware_shadow"
            and receipt.get("command_applied") is True
        ):
            issues.append(
                {
                    "code": "hardware_shadow_command_applied",
                    "event_type": "execution_receipt",
                    "event_id": command_id,
                    "message": (
                        "hardware-shadow receipt cannot claim an applied command"
                    ),
                }
            )
        if (
            receipt.get("command_applied") is True
            and receipt.get("accepted") is not True
        ):
            issues.append(
                {
                    "code": "command_applied_without_acceptance",
                    "event_type": "execution_receipt",
                    "event_id": command_id,
                    "message": (
                        "receipt cannot apply a command it did not accept"
                    ),
                }
            )

    execution_modes = {
        mode
        for event in events
        if event.get("event_type") in {"execution_receipt", "outcome"}
        for mode in (event.get("execution_mode"),)
        if isinstance(mode, str)
    }
    if len(execution_modes) > 1:
        issues.append(
            {
                "code": "execution_mode_mismatch",
                "modes": sorted(execution_modes),
                "message": (
                    "runtime trace contains contradictory execution modes"
                ),
            }
        )
    return issues


def _lineage_issue(
    code: str,
    event_type: str,
    event_id: Any,
    field: str,
    value: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "event_type": event_type,
        "event_id": event_id,
        "field": field,
        "value": value,
        "message": f"{event_type} contains unresolved {field}",
    }


def _instrumentation_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    activations = [
        event for event in events if event.get("event_type") == "model_activation"
    ]
    drops = [
        event
        for event in events
        if event.get("event_type") == "activation_capture_dropped"
    ]
    storage_modes = {
        storage
        for event in activations
        if isinstance(event.get("capture"), dict)
        for storage in (
            event["capture"].get(
                "storage_mode",
                event["capture"].get("storage"),
            ),
        )
        if isinstance(storage, str) and storage
    }
    return {
        "activation_events": len(activations),
        "dropped_events": len(drops),
        "modules": sorted(
            {
                str(event["module"])
                for event in activations
                if isinstance(event.get("module"), str) and event["module"]
            }
        ),
        "probes": sorted(
            {
                str(event["probe_id"])
                for event in activations
                if isinstance(event.get("probe_id"), str) and event["probe_id"]
            }
        ),
        "storage_modes": sorted(storage_modes),
        "drop_reasons": sorted(
            {
                str(event["reason"])
                for event in drops
                if isinstance(event.get("reason"), str) and event["reason"]
            }
        ),
        "causal_effect_established": False,
    }


def _compare_events(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_chunks = [
        event for event in baseline if event.get("event_type") == "action_chunk"
    ]
    candidate_chunks = [
        event for event in candidate if event.get("event_type") == "action_chunk"
    ]
    first_action_divergence_step = None
    for index in range(max(len(baseline_chunks), len(candidate_chunks))):
        baseline_event = (
            baseline_chunks[index] if index < len(baseline_chunks) else None
        )
        candidate_event = (
            candidate_chunks[index] if index < len(candidate_chunks) else None
        )
        if baseline_event is None or candidate_event is None:
            present = candidate_event or baseline_event or {}
            first_action_divergence_step = present.get("step")
            break
        if baseline_event.get("value") != candidate_event.get("value"):
            first_action_divergence_step = min(
                int(baseline_event.get("step", 0)),
                int(candidate_event.get("step", 0)),
            )
            break

    baseline_decisions = [
        (event.get("accepted"), event.get("reason"))
        for event in baseline
        if event.get("event_type") == "safety_decision"
    ]
    candidate_decisions = [
        (event.get("accepted"), event.get("reason"))
        for event in candidate
        if event.get("event_type") == "safety_decision"
    ]
    baseline_applied = sum(
        event.get("event_type") == "execution_receipt"
        and event.get("command_applied") is True
        for event in baseline
    )
    candidate_applied = sum(
        event.get("event_type") == "execution_receipt"
        and event.get("command_applied") is True
        for event in candidate
    )
    baseline_outcome = _last_event(baseline, "outcome")
    candidate_outcome = _last_event(candidate, "outcome")
    return {
        "first_action_divergence_step": first_action_divergence_step,
        "safety_decision_changed": baseline_decisions != candidate_decisions,
        "execution_receipt_delta": (
            _count(candidate, "execution_receipt")
            - _count(baseline, "execution_receipt")
        ),
        "applied_command_delta": candidate_applied - baseline_applied,
        "outcome_changed": (
            baseline_outcome.get("success"),
            baseline_outcome.get("reason"),
        )
        != (
            candidate_outcome.get("success"),
            candidate_outcome.get("reason"),
        ),
    }


def _last_event(
    events: list[dict[str, Any]],
    event_type: str,
) -> dict[str, Any]:
    matches = [
        event for event in events if event.get("event_type") == event_type
    ]
    return matches[-1] if matches else {}


def _timeline_row(event: Mapping[str, Any]) -> str:
    step = event.get("step", "n/a")
    event_type = str(event.get("event_type", "unknown"))
    identity = _timeline_identity(event)
    detail = _timeline_detail(event)
    return (
        f"| {_md(step)} | {_md(event_type)} | `{_md(identity)}` | "
        f"{_md(detail)} |"
    )


def _activation_timeline_row(
    key: tuple[str, str],
    activations: Sequence[Mapping[str, Any]],
) -> str:
    step, policy_call_id = key
    modules = sorted(
        {
            str(event["module"])
            for event in activations
            if isinstance(event.get("module"), str)
        }
    )
    detail = (
        f"{len(activations)} activation events across {len(modules)} modules"
    )
    return (
        f"| {_md(step)} | model_activation (aggregated) | "
        f"`{_md(policy_call_id)}` | {_md(detail)} |"
    )


def _timeline_identity(event: Mapping[str, Any]) -> str:
    for field_name in (
        "command_id",
        "chunk_id",
        "policy_call_id",
        "observation_id",
        "observation_ref",
        "scenario_id",
    ):
        value = event.get(field_name)
        if value is not None:
            return str(value)
    return "n/a"


def _timeline_detail(event: Mapping[str, Any]) -> str:
    event_type = event.get("event_type")
    if event_type == "instruction":
        return _bounded(event.get("value"))
    if event_type == "observation":
        return f"source={_display(event.get('source'))}"
    if event_type == "policy_inference":
        return (
            f"status={_display(event.get('status'))}; "
            f"chunk={_display(event.get('action_chunk_id'))}"
        )
    if event_type == "action_chunk":
        return (
            f"policy={_display(event.get('policy_call_id'))}; "
            f"shape={_display(event.get('shape'))}"
        )
    if event_type == "scheduled_action":
        return (
            f"chunk={_display(event.get('chunk_id'))}; "
            f"index={_display(event.get('action_index'))}"
        )
    if event_type == "safety_decision":
        return (
            f"accepted={_bool_text(event.get('accepted'))}; "
            f"reason={_display(event.get('reason'))}"
        )
    if event_type == "execution_receipt":
        return (
            f"backend={_display(event.get('backend_id'))}; "
            f"applied={_bool_text(event.get('command_applied'))}"
        )
    if event_type == "robot_state":
        return "measured state recorded"
    if event_type == "activation_capture_dropped":
        return f"reason={_display(event.get('reason'))}"
    if event_type == "runtime_error":
        return (
            f"phase={_display(event.get('phase'))}; "
            f"component={_display(event.get('component_id'))}"
        )
    if event_type == "outcome":
        return (
            f"success={_bool_text(event.get('success'))}; "
            f"reason={_display(event.get('reason'))}"
        )
    return f"layer={_display(event.get('layer'))}"


def _display(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _display_list(value: Any) -> str:
    if not value:
        return "none"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _bool_text(value: Any) -> str:
    if value is None:
        return "n/a"
    return "true" if value is True else "false"


def _bounded(value: Any, limit: int = 120) -> str:
    rendered = _display(value).replace("\n", " ").replace("\r", " ")
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 1] + "…"


def _md(value: Any) -> str:
    return _bounded(value).replace("|", "\\|").replace("`", "'")


def _configuration_summary(path: Path) -> dict[str, Any]:
    inspected = inspect_runtime_config(path)
    capture = inspected.get("capture")
    capture_summary = None
    if isinstance(capture, dict):
        profile_value = capture.get("profile")
        profile = Path(profile_value) if isinstance(profile_value, str) else None
        capture_summary = {
            "storage": capture.get("storage"),
            "max_bytes_per_run": capture.get("max_bytes_per_run"),
            "profile_sha256": (
                _sha256(profile)
                if profile is not None and profile.is_file()
                else None
            ),
        }
    return {
        "sha256": _sha256(path),
        "pack": {
            "id": inspected["pack"]["id"],
            "version": inspected["pack"]["version"],
            "manifest_sha256": inspected["pack"]["manifest_sha256"],
        },
        "components": {
            role: {
                "plugin": component["plugin"],
                "component_id": component["component_id"],
            }
            for role, component in sorted(inspected["components"].items())
        },
        "execution": {
            "mode": inspected["execution"]["mode"],
            "n_action_steps": inspected["execution"]["n_action_steps"],
            "command_space": inspected["execution"]["command_space"],
        },
        "capture": capture_summary,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
