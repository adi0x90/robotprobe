from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from robotprobe.core import load_intervention_manifest
from robotprobe.metrics import compute_metric_summary


def summarize_trace_pair(
    clean_trace: str | Path,
    perturbed_trace: str | Path,
    *,
    intervention_manifest_ref: str | Path | None = None,
    metrics: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    clean = read_trace(clean_trace)
    perturbed = read_trace(perturbed_trace)
    manifest = load_intervention_manifest(intervention_manifest_ref) if intervention_manifest_ref else None

    raw_policy_drift, raw_drift_first = _changed(clean, perturbed, "raw_action")
    post_drift, post_drift_first = _changed(clean, perturbed, "postprocessed_action")
    instruction_changed, instruction_first = _instruction_changed_within_trace(perturbed)
    observation_replayed, observation_first = _observation_replayed_within_trace(perturbed)
    action_mutated, action_mutated_first = _changed_within_trace(
        perturbed,
        "raw_action",
        "postprocessed_action",
    )

    # A post-policy intervention cannot change raw policy output. Under that manifest the raw
    # action comes from a stochastic policy on a different rollout, so cross-run raw drift is
    # run-to-run noise; the intervention's own effect is the raw-vs-post mutation inside the
    # perturbed trace. Outside that case both flags are plain clean-vs-perturbed comparisons.
    if _is_action_postprocess_manifest(manifest):
        raw_changed = False
        post_changed = action_mutated
        raw_divergence_first = None
        post_divergence_first = action_mutated_first
    else:
        raw_changed = raw_policy_drift
        post_changed = post_drift
        raw_divergence_first = raw_drift_first
        post_divergence_first = post_drift_first

    declared_layer = manifest.target_layers[0] if manifest and manifest.target_layers else None
    root_layer = _observed_layer(
        instruction_changed=instruction_changed,
        observation_replayed=observation_replayed,
        raw_changed=raw_changed,
        post_changed=post_changed,
    )
    first_divergence = _first_divergence_step(
        instruction_changed=instruction_changed,
        instruction_first=instruction_first,
        observation_replayed=observation_replayed,
        observation_first=observation_first,
        raw_first=raw_divergence_first,
        post_first=post_divergence_first,
    )
    clean_outcome = _last_event(clean, "outcome")
    perturbed_outcome = _last_event(perturbed, "outcome")
    step_inflation = _step_inflation(clean_outcome, perturbed_outcome)
    contract_violations = _contract_violations(perturbed)
    clean_outcome_violations = _outcome_violations(clean_outcome)
    perturbed_outcome_violations = _outcome_violations(perturbed_outcome)
    outcome_violation_changed = clean_outcome_violations != perturbed_outcome_violations

    summary = {
        "root_cause_hypothesis": {
            "layer": root_layer,
            "declared_layer": declared_layer,
            "contract": manifest.contract if manifest else None,
            "fault": manifest.name if manifest else _scenario_name(perturbed),
            "rule_match_score": _rule_match_score(root_layer, declared_layer),
        },
        "evidence": {
            # "changed" answers "did the intervention alter this layer?"; "_drift_vs_clean" is the
            # raw clean-vs-perturbed comparison, which also moves when the policy is stochastic.
            "raw_policy_action_changed": raw_changed,
            "postprocessed_action_changed": post_changed,
            "raw_policy_drift_vs_clean": raw_policy_drift,
            "postprocessed_action_drift_vs_clean": post_drift,
            "raw_action_mutated_after_policy": action_mutated,
            "instruction_changed": instruction_changed,
            "observation_replayed": observation_replayed,
            "contract_violated": bool(contract_violations),
            "violated_contracts": sorted({violation.get("contract") for violation in contract_violations if violation.get("contract")}),
            "max_contract_severity": _max_contract_severity(contract_violations),
            "clean_outcome_violations": clean_outcome_violations,
            "perturbed_outcome_violations": perturbed_outcome_violations,
            "outcome_violation_changed": outcome_violation_changed,
            "clean_keepout_violated": bool(_field(clean_outcome, "keepout_violated")),
            "perturbed_keepout_violated": bool(_field(perturbed_outcome, "keepout_violated")),
            "hidden_success_safety_violation": (
                _field(clean_outcome, "success") is True
                and _field(perturbed_outcome, "success") is True
                and outcome_violation_changed
            ),
            "first_divergence_step": first_divergence,
            "outcome_delta": {
                "success_changed": _field(clean_outcome, "success") != _field(perturbed_outcome, "success"),
                "step_inflation": step_inflation,
                "outcome_violation_changed": outcome_violation_changed,
                "keepout_violation_changed": bool(_field(clean_outcome, "keepout_violated")) != bool(_field(perturbed_outcome, "keepout_violated")),
            },
        },
        "replay_level": _field(perturbed[0] if perturbed else {}, "replay_level"),
    }
    if metrics:
        summary["metrics"] = compute_metric_summary(
            clean,
            perturbed,
            metric_names=metrics,
            first_divergence_step=first_divergence,
            manifest=manifest,
        )
    return summary


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _events_by_step(events: list[dict[str, Any]], event_type: str) -> dict[int, dict[str, Any]]:
    return {int(e["step"]): e for e in events if e.get("event_type") == event_type}


def _changed(
    clean: list[dict[str, Any]],
    perturbed: list[dict[str, Any]],
    event_type: str,
) -> tuple[bool, int | None]:
    clean_by_step = _events_by_step(clean, event_type)
    perturbed_by_step = _events_by_step(perturbed, event_type)
    for step in sorted(set(clean_by_step) & set(perturbed_by_step)):
        if not _same_value(clean_by_step[step].get("value"), perturbed_by_step[step].get("value")):
            return True, step
    return False, None


def _changed_within_trace(
    events: list[dict[str, Any]],
    before_event_type: str,
    after_event_type: str,
) -> tuple[bool, int | None]:
    """Did the value change between two layers inside the same run?"""
    before_by_step = _events_by_step(events, before_event_type)
    after_by_step = _events_by_step(events, after_event_type)
    for step in sorted(set(before_by_step) & set(after_by_step)):
        if not _same_value(before_by_step[step].get("value"), after_by_step[step].get("value")):
            return True, step
    return False, None


def _instruction_changed_within_trace(events: list[dict[str, Any]]) -> tuple[bool, int | None]:
    for event in events:
        if event.get("event_type") != "instruction":
            continue
        before = event.get("before")
        after = event.get("after", event.get("value"))
        # A declaration or an unknown endpoint does not establish a mutation.
        if before is None or after is None:
            continue
        if before != after:
            return True, int(event["step"])
    return False, None


def _observation_replayed_within_trace(events: list[dict[str, Any]]) -> tuple[bool, int | None]:
    previous_hash: str | None = None
    for event in events:
        if event.get("event_type") != "observation":
            continue
        image_hash = event.get("image_sha256")
        if image_hash is not None and image_hash == previous_hash:
            return True, int(event["step"])
        previous_hash = image_hash
    return False, None


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
    return left == right


def _min_present(*values: int | None) -> int | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def _last_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    matches = [e for e in events if e.get("event_type") == event_type]
    return matches[-1] if matches else {}


def _contract_violations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("event_type") == "contract_check" and event.get("status") == "violated"
    ]


def _max_contract_severity(events: list[dict[str, Any]]) -> float | None:
    severities = [event.get("severity") for event in events if isinstance(event.get("severity"), (int, float))]
    return float(max(severities)) if severities else None


def _outcome_violations(outcome: dict[str, Any]) -> list[str]:
    return sorted(key for key, value in outcome.items() if key.endswith("_violated") and value is True)


def _field(event: dict[str, Any], key: str) -> Any:
    return event.get(key)


def _step_inflation(clean_outcome: dict[str, Any], perturbed_outcome: dict[str, Any]) -> float | None:
    clean_steps = clean_outcome.get("steps")
    perturbed_steps = perturbed_outcome.get("steps")
    if not clean_steps or not perturbed_steps:
        return None
    return float(perturbed_steps) / float(clean_steps)


def _is_action_postprocess_manifest(manifest) -> bool:
    return bool(manifest and manifest.target_layers and manifest.target_layers[0] == "action_postprocess")


def _observed_layer(
    *,
    instruction_changed: bool,
    observation_replayed: bool,
    raw_changed: bool,
    post_changed: bool,
) -> str:
    """Layer where the recorded data first differs. Measured from the traces, not declared."""
    if instruction_changed:
        return "instruction"
    if observation_replayed:
        return "observation"
    if post_changed and not raw_changed:
        return "action_postprocess"
    if raw_changed:
        return "policy_forward"
    return "unknown"


def _first_divergence_step(
    *,
    instruction_changed: bool,
    instruction_first: int | None,
    observation_replayed: bool,
    observation_first: int | None,
    raw_first: int | None,
    post_first: int | None,
) -> int | None:
    if instruction_changed:
        return instruction_first
    if observation_replayed:
        return observation_first
    return _min_present(raw_first, post_first)


def _rule_match_score(root_layer: str, declared_layer: str | None) -> float:
    """Fixed score from the layer heuristic.

    This is a rule constant, not a calibrated probability. It is kept so existing
    challenge packs and golden summaries stay comparable; it will be replaced by a
    measured score once there is a labelled incident corpus to calibrate against.
    """
    if declared_layer is not None:
        return 0.87 if root_layer == declared_layer else 0.62
    return 0.62 if root_layer == "unknown" else 0.87


def _scenario_name(events: list[dict[str, Any]]) -> str | None:
    return events[0].get("scenario") if events else None
