from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_incident_report(
    summary: dict[str, Any],
    *,
    check: dict[str, Any] | None = None,
    title: str = "RobotProbe Incident Report",
) -> str:
    hypothesis = summary.get("root_cause_hypothesis", {})
    evidence = summary.get("evidence", {})
    outcome_delta = evidence.get("outcome_delta", {})
    check_passed = check.get("passed") if check else None

    lines = [
        f"# {title}",
        "",
        "## Diagnosis",
        "",
        f"- Layer: `{_value(hypothesis.get('layer'))}`",
        f"- Declared by manifest: `{_value(hypothesis.get('declared_layer'))}`",
        f"- Contract: `{_value(hypothesis.get('contract'))}`",
        f"- Fault: `{_value(hypothesis.get('fault'))}`",
        f"- Layer match score: `{_format_float(hypothesis.get('rule_match_score'))}`",
        f"- RobotSec check: `{_format_check(check_passed)}`",
        "",
        "## Evidence",
        "",
        f"- First divergence step: `{_value(evidence.get('first_divergence_step'))}`",
        f"- Raw policy action changed: `{_format_bool(evidence.get('raw_policy_action_changed'))}`",
        f"- Postprocessed action changed: `{_format_bool(evidence.get('postprocessed_action_changed'))}`",
        f"- Instruction changed: `{_format_bool(evidence.get('instruction_changed'))}`",
        f"- Observation replayed: `{_format_bool(evidence.get('observation_replayed'))}`",
        f"- Contract violated: `{_format_bool(evidence.get('contract_violated'))}`",
        f"- Outcome violation changed: `{_format_bool(evidence.get('outcome_violation_changed'))}`",
        f"- Clean outcome violations: `{_format_list(evidence.get('clean_outcome_violations'))}`",
        f"- Perturbed outcome violations: `{_format_list(evidence.get('perturbed_outcome_violations'))}`",
        f"- Hidden success safety violation: `{_format_bool(evidence.get('hidden_success_safety_violation'))}`",
        f"- Success changed: `{_format_bool(outcome_delta.get('success_changed'))}`",
        f"- Step inflation: `{_format_ratio(outcome_delta.get('step_inflation'))}`",
        f"- Replay level: `{_value(summary.get('replay_level'))}`",
    ]
    if check and check.get("failures"):
        lines.extend(["", "## Check Failures", ""])
        lines.extend(f"- `{failure}`" for failure in check["failures"])
    metrics = summary.get("metrics", {})
    if metrics:
        lines.extend(["", "## Metrics", ""])
        for group_name in ("clean", "perturbed", "delta"):
            group = metrics.get(group_name, {})
            for key in sorted(group):
                lines.append(f"- {group_name}.{key}: `{_format_metric(group[key])}`")
    return "\n".join(lines) + "\n"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_incident_report(
    summary_path: str | Path,
    *,
    check_path: str | Path | None = None,
    out_path: str | Path | None = None,
    title: str = "RobotProbe Incident Report",
) -> str:
    report = build_incident_report(
        load_json(summary_path),
        check=load_json(check_path) if check_path else None,
        title=title,
    )
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    return report


def _value(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _format_bool(value: Any) -> str:
    if value is None:
        return "n/a"
    return "true" if bool(value) else "false"


def _format_check(value: Any) -> str:
    if value is None:
        return "n/a"
    return "PASS" if bool(value) else "FAIL"


def _format_list(value: Any) -> str:
    if value is None:
        return "n/a"
    if not value:
        return "none"
    return ", ".join(str(item) for item in value)


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _format_ratio(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}x"


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
