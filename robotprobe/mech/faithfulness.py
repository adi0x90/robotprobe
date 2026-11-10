from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robotprobe.mech.cache_view import ActivationCacheView


def compute_consequence_faithfulness(
    *,
    mech_dir: str | Path,
    summary_path: str | Path,
    target_metric: str,
    out_path: str | Path | None = None,
    max_k: int | None = None,
) -> dict[str, Any]:
    if max_k is not None and (type(max_k) is not int or max_k <= 0):
        raise ValueError("max_k must be a positive integer")
    view = ActivationCacheView(mech_dir)
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    consequence_score = _target_consequence_score(summary, target_metric)
    if not np.isfinite(consequence_score):
        raise ValueError("consequence score must be finite")
    components = []
    for module in view.component_names:
        component = _component_faithfulness(
            view,
            module,
            consequence_score=consequence_score,
            max_k=max_k,
        )
        components.append(component)
    any_activation_delta = any(component["max_activation_delta"] > 0 for component in components)
    comparison_complete = view.comparison_complete
    eligible = view.synthetic is False and comparison_complete
    evidence_level = 1 if eligible and consequence_score > 0 and any_activation_delta else 0
    result = {
        "format": "robotprobe.mech.consequence_aopc.v0",
        "mode": "offline_proxy",
        "incident_id": view.incident_id,
        "synthetic": view.synthetic,
        "evidence_eligible": eligible,
        "comparison_complete": comparison_complete,
        "target_metric": target_metric,
        "consequence_score": consequence_score,
        "intervention_executed": False,
        "promotion_allowed": False,
        "evidence_level": evidence_level,
        "replay_level": summary.get("replay_level"),
        "components": components,
        "claim_boundary": {
            "offline_proxy_only": True,
            "causal_without_intervention": False,
            "requires_replay_or_patch_for_level_3": True,
        },
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _component_faithfulness(
    view: ActivationCacheView,
    module: str,
    *,
    consequence_score: float,
    max_k: int | None,
) -> dict[str, Any]:
    deltas = []
    original_shape = None
    for step in view.paired_steps(module):
        clean, perturbed = view.paired_arrays(module, step)
        if original_shape is not None and clean.shape != original_shape:
            raise ValueError(f"activation shape changes across steps for {module}")
        original_shape = clean.shape
        delta = np.abs(perturbed.reshape(-1).astype(np.float64) - clean.reshape(-1).astype(np.float64))
        if not np.all(np.isfinite(delta)):
            raise ValueError(f"activation delta must be finite for {module} step {step}")
        deltas.append(delta)
    if not deltas:
        return {
            "module": module,
            **view.comparison_coverage(module),
            "top_dimensions": [],
            "k_values": [],
            "metric_deltas": [],
            "aopc_score": 0.0,
            "max_activation_delta": 0.0,
        }
    importance = np.mean(np.stack(deltas, axis=0), axis=0)
    dimension_count = int(importance.shape[0])
    limit = min(dimension_count, max_k if max_k is not None else dimension_count)
    ranked = sorted(range(dimension_count), key=lambda idx: (-float(importance[idx]), idx))
    top_dimensions = ranked[:limit]
    total_importance = float(np.sum(importance))
    k_values = list(range(1, limit + 1))
    cumulative = np.cumsum(importance[top_dimensions], dtype=np.float64)
    fractions = cumulative / total_importance if total_importance > 0 else np.zeros(limit)
    metric_deltas = (consequence_score * fractions).tolist()
    return {
        "module": module,
        **view.comparison_coverage(module),
        "top_dimensions": top_dimensions,
        "k_values": k_values,
        "metric_deltas": metric_deltas,
        "aopc_score": _integrate_area(k_values, metric_deltas),
        "max_activation_delta": float(np.max(importance)) if importance.size else 0.0,
    }


def _target_consequence_score(summary: dict[str, Any], target_metric: str) -> float:
    evidence = summary.get("evidence", {})
    metrics = summary.get("metrics", {})
    if not isinstance(evidence, dict) or not isinstance(metrics, dict):
        raise ValueError("summary evidence and metrics must be objects")
    outcome_delta = evidence.get("outcome_delta", {})
    metric_delta = metrics.get("delta", {})
    if not isinstance(outcome_delta, dict) or not isinstance(metric_delta, dict):
        raise ValueError("summary outcome_delta and metrics.delta must be objects")
    if target_metric == "raw_action_delta":
        return 1.0 if evidence.get("raw_policy_action_changed") else 0.0
    if target_metric == "postprocessed_action_delta":
        return 1.0 if evidence.get("postprocessed_action_changed") else 0.0
    if target_metric == "contract_violation_delta":
        if evidence.get("contract_violated"):
            return float(evidence.get("max_contract_severity") or 1.0)
        return 0.0
    if target_metric == "near_miss_delta":
        return float(metric_delta.get("near_miss_score") or 0.0)
    if target_metric == "trajectory_deviation":
        return float(metric_delta.get("trajectory_deviation") or 0.0)
    if target_metric == "action_energy_delta":
        return abs(float(metric_delta.get("action_energy") or 0.0))
    if target_metric == "step_inflation":
        step_inflation = outcome_delta.get("step_inflation")
        return max(0.0, float(step_inflation) - 1.0) if step_inflation is not None else 0.0
    if target_metric == "outcome_delta":
        return 1.0 if outcome_delta.get("success_changed") or outcome_delta.get("outcome_violation_changed") else 0.0
    raise ValueError(f"unsupported target_metric: {target_metric}")


def _integrate_area(x_values: list[int], y_values: list[float]) -> float:
    if not y_values:
        return 0.0
    if len(y_values) == 1:
        return float(y_values[0])
    area = 0.0
    for index in range(len(y_values) - 1):
        width = x_values[index + 1] - x_values[index]
        area += width * (y_values[index] + y_values[index + 1]) / 2.0
    return float(area)
