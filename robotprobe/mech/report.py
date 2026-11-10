from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_mech_report(
    *,
    summary: dict[str, Any],
    activation_diff: dict[str, Any],
    feature_diff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(summary, dict) or not isinstance(activation_diff, dict):
        raise ValueError("summary and activation diff must be objects")
    external_hypothesis = summary.get("root_cause_hypothesis", {})
    evidence = summary.get("evidence", {})
    if not isinstance(external_hypothesis, dict) or not isinstance(evidence, dict):
        raise ValueError("summary root_cause_hypothesis and evidence must be objects")
    hypothesis = activation_diff.get("hypothesis", {})
    layers = activation_diff.get("layers", [])
    if not isinstance(hypothesis, dict) or not isinstance(layers, list) or not all(isinstance(layer, dict) for layer in layers):
        raise ValueError("activation diff requires an object hypothesis and a list of layers")
    if feature_diff is not None and (
        not isinstance(feature_diff, dict) or not isinstance(feature_diff.get("features", []), list)
        or not all(isinstance(feature, dict) for feature in feature_diff.get("features", []))
    ):
        raise ValueError("feature diff must be an object with a features list")
    synthetic_value = activation_diff.get("synthetic")
    synthetic = synthetic_value if isinstance(synthetic_value, bool) else None
    comparison_complete = activation_diff.get("comparison_complete") is True and bool(layers) and all(
        layer.get("comparison_complete") is True and layer.get("paired_step_count", 0) > 0
        for layer in layers
    )
    evidence_eligible = synthetic is False and comparison_complete and activation_diff.get("evidence_eligible") is True
    any_significant = any(layer.get("activation_diff_significant") for layer in layers)
    negative_control_stable = comparison_complete and all(layer.get("negative_control_stable") is True for layer in layers)
    external_layer = external_hypothesis.get("layer")
    internal_precedes = bool(hypothesis.get("internal_divergence_precedes_action_divergence"))
    evidence_level = 1 if evidence_eligible else 0
    if synthetic is True:
        claim_boundary = "synthetic fixture only; not evidence about a model"
    elif synthetic is None:
        claim_boundary = "capture provenance unknown; not eligible as model evidence"
    elif not comparison_complete:
        claim_boundary = "insufficient paired capture coverage; stability and attribution are undetermined"
    elif not evidence_eligible:
        claim_boundary = "capture evidence is ineligible; no model finding established"
    elif negative_control_stable and external_layer == "action_postprocess":
        claim_boundary = "negative control stable; do not blame model internals without intervention evidence"
    elif internal_precedes:
        claim_boundary = "observed internal change precedes action change; prediction and causation are untested"
    elif any_significant:
        claim_boundary = "correlational activation divergence only"
    else:
        claim_boundary = "no significant internal divergence in paired captures; no model root-cause claim"
    action_first = activation_diff.get("first_action_divergence_step")
    if action_first is None:
        action_steps = [layer.get("first_action_divergence_step", layer.get("external_first_divergence_step")) for layer in layers]
        present = [step for step in action_steps if step is not None]
        action_first = min(present) if present else None
    return {
        "format": "robotprobe.mech.report.v0",
        "incident_id": activation_diff.get("incident_id"),
        "synthetic": synthetic,
        "evidence_eligible": evidence_eligible,
        "comparison_complete": comparison_complete,
        "negative_control_stable": negative_control_stable,
        "external_root_cause_layer": external_layer,
        "first_external_divergence_step": evidence.get("first_divergence_step"),
        "first_action_divergence_step": action_first,
        "activation_layers": layers,
        "feature_diff": feature_diff,
        "evidence_level": evidence_level,
        "model_root_cause_claim_allowed": False,
        "claim_boundary": claim_boundary,
        "recommended_regression": _recommended_regression(external_layer, internal_precedes),
    }


def write_mech_report(
    *,
    summary_path: str | Path,
    activation_diff_path: str | Path,
    out_path: str | Path,
    feature_diff_path: str | Path | None = None,
) -> dict[str, Any]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    activation_diff = json.loads(Path(activation_diff_path).read_text(encoding="utf-8"))
    feature_diff = json.loads(Path(feature_diff_path).read_text(encoding="utf-8")) if feature_diff_path else None
    report = build_mech_report(summary=summary, activation_diff=activation_diff, feature_diff=feature_diff)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        out.write_text(render_mech_report_markdown(report), encoding="utf-8")
    return report


def render_mech_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MechProbe Incident Report",
        "",
        "## External Incident Summary",
        "",
        f"- External root-cause layer: `{_value(report.get('external_root_cause_layer'))}`",
        f"- First external incident divergence step: `{_value(report.get('first_external_divergence_step'))}`",
        f"- First action divergence step: `{_value(report.get('first_action_divergence_step'))}`",
        "",
        "## Activation Evidence",
        "",
        f"- Capture provenance: `{_provenance_label(report.get('synthetic'))}`",
        f"- Evidence eligible: `{_bool(report.get('evidence_eligible'))}`",
        f"- Paired comparison complete: `{_bool(report.get('comparison_complete'))}`",
        "",
    ]
    for layer in report.get("activation_layers", []):
        lines.extend(
            [
                f"- Module: `{layer.get('module')}`",
                f"  - Significant activation diff: `{_bool(layer.get('activation_diff_significant'))}`",
                f"  - First internal divergence step: `{_value(layer.get('first_significant_divergence_step'))}`",
                f"  - Lead time before action divergence: `{_value(layer.get('lead_time_steps'))}` steps",
                f"  - Paired steps: `{_value(layer.get('paired_step_count'))}`",
                f"  - Comparison complete: `{_bool(layer.get('comparison_complete'))}`",
                f"  - Negative-control stable: `{_bool(layer.get('negative_control_stable'))}`",
            ]
        )
    features = (report.get("feature_diff") or {}).get("features", [])
    if features:
        lines.extend(["", "## Candidate Features", ""])
        for feature in features:
            lines.extend([
                f"- {feature.get('label')} (`{feature.get('feature_id')}`)",
                f"  - Evidence level: `{_value(feature.get('evidence_level'))}`",
                f"  - Evidence eligible: `{_bool(feature.get('evidence_eligible'))}`",
                f"  - First feature divergence step: `{_value(feature.get('first_feature_divergence_step'))}`",
                f"  - Lead time before action divergence: `{_value(feature.get('lead_time_steps'))}` steps",
                f"  - {feature.get('reason', 'Candidate label only; no causal claim.')}",
            ])
    lines.extend(
        [
            "",
            "## Evidence Level",
            "",
            f"- Level: `{report.get('evidence_level')}`",
            f"- Model root-cause claim allowed: `{_bool(report.get('model_root_cause_claim_allowed'))}`",
            "",
            "## What This Does Not Prove",
            "",
            f"- {report.get('claim_boundary')}",
            "- It does not prove an SAE feature label by inspection.",
            "- It does not prove hardware safety or real-world exploitability.",
            "",
            "## Next Mitigation / Regression",
            "",
            f"- {report.get('recommended_regression')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _recommended_regression(external_layer: str | None, internal_precedes: bool) -> str:
    if external_layer == "action_postprocess":
        return "keep external action-contract regression; use MechProbe as a negative control"
    if internal_precedes:
        return "add activation-diff timing check before promoting any feature-level claim"
    return "collect more paired traces before adding mechanistic claims"


def _value(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _provenance_label(synthetic: bool | None) -> str:
    if synthetic is True:
        return "synthetic fixture"
    return "model capture" if synthetic is False else "unknown"
