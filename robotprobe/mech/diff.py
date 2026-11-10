from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from robotprobe.mech.cache_view import ActivationCacheView


def diff_activations(
    *,
    index_path: str | Path | None = None,
    mech_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    threshold: float = 1e-6,
) -> dict[str, Any]:
    if index_path is None:
        if mech_dir is None:
            raise ValueError("index_path or mech_dir is required")
        index_path = Path(mech_dir) / "activation_index.json"
    index_file = Path(index_path)
    root = Path(mech_dir) if mech_dir is not None else index_file.parent
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    view = ActivationCacheView(root, index_path=index_file)
    index = view.index
    external_first = index.get("external_first_divergence_step")
    layers = []
    for module in view.component_names:
        coverage = view.comparison_coverage(module)
        step_deltas = []
        for step in view.paired_steps(module):
            clean, perturbed = view.paired_arrays(module, step)
            clean, perturbed = clean.reshape(-1), perturbed.reshape(-1)
            delta = perturbed.astype(np.float64) - clean.astype(np.float64)
            l2_delta = float(np.linalg.norm(delta))
            if not math.isfinite(l2_delta):
                raise ValueError(f"activation delta must be finite for {module} step {step}")
            step_deltas.append(
                {
                    "step": int(step),
                    "l2_delta": l2_delta,
                    "cosine_distance": _cosine_distance(clean, perturbed),
                }
            )
        significant_steps = [item["step"] for item in step_deltas if item["l2_delta"] > threshold]
        first_internal = min(significant_steps) if significant_steps else None
        mean_l2 = float(np.mean([item["l2_delta"] for item in step_deltas])) if step_deltas else None
        max_l2 = float(max(item["l2_delta"] for item in step_deltas)) if step_deltas else None
        lead_time = (
            int(external_first) - int(first_internal)
            if external_first is not None and first_internal is not None
            else None
        )
        significant = first_internal is not None
        layers.append(
            {
                "module": module,
                "mean_l2_delta": mean_l2,
                "max_l2_delta": max_l2,
                "first_significant_divergence_step": first_internal,
                "external_first_divergence_step": external_first,
                "first_action_divergence_step": external_first,
                "lead_time_steps": lead_time,
                "stability_score": 1.0 / (1.0 + mean_l2) if mean_l2 is not None else None,
                "activation_diff_significant": significant,
                "negative_control_stable": bool(index.get("negative_control")) and coverage["comparison_complete"] and not significant,
                **coverage,
                "step_deltas": step_deltas,
            }
        )
    first_internal_any = _min_present(layer["first_significant_divergence_step"] for layer in layers)
    internal_precedes = (
        first_internal_any is not None
        and external_first is not None
        and int(first_internal_any) < int(external_first)
    )
    comparison_complete = bool(layers) and all(layer["comparison_complete"] for layer in layers)
    evidence_eligible = view.synthetic is False and comparison_complete
    result = {
        "format": "robotprobe.mech.activation_diff.v0",
        "incident_id": index.get("incident_id"),
        "run_pair": index.get("trace_pair", {}),
        "model": index.get("model", {}),
        "synthetic": view.synthetic,
        "evidence_eligible": evidence_eligible,
        "evidence_level": 1 if evidence_eligible else 0,
        "comparison_complete": comparison_complete,
        "negative_control": bool(index.get("negative_control")),
        "first_action_divergence_step": external_first,
        "layers": layers,
        "hypothesis": {
            "internal_divergence_precedes_action_divergence": internal_precedes,
            "rule_match_score": (0.68 if internal_precedes else 0.42) if evidence_eligible else 0.0,
            "recommended_attribution": "model_internal_candidate" if internal_precedes else "external_or_downstream",
            "evidence_eligible": evidence_eligible,
        },
    }
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _cosine_distance(clean: np.ndarray, perturbed: np.ndarray) -> float:
    left = clean.astype(np.float64)
    right = perturbed.astype(np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if math.isclose(denom, 0.0):
        return 0.0 if np.allclose(left, right) else 1.0
    cosine = float(np.dot(left, right) / denom)
    return 1.0 - max(-1.0, min(1.0, cosine))


def _min_present(values) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None
