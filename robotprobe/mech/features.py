from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robotprobe.mech.cache_view import ActivationCacheView


def diff_features(
    *,
    mech_dir: str | Path,
    dictionary_path: str | Path,
    out_path: str | Path | None = None,
    threshold: float = 1e-6,
) -> dict[str, Any]:
    root = Path(mech_dir)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    view = ActivationCacheView(root)
    index = view.index
    dictionary = _load_dictionary(dictionary_path)
    external_first = index.get("external_first_divergence_step")
    features = []
    for spec in dictionary.get("features", []):
        module = str(spec["layer"])
        coverage = view.comparison_coverage(module)
        direction = np.asarray(spec["direction"], dtype=np.float64).reshape(-1)
        if not direction.size or not np.all(np.isfinite(direction)):
            raise ValueError("feature directions must contain finite values")
        step_deltas = []
        steps = view.paired_steps(module) if module in view.component_names else []
        for step in steps:
            clean, perturbed = view.paired_arrays(module, step)
            clean = clean.reshape(-1).astype(np.float64)
            perturbed = perturbed.reshape(-1).astype(np.float64)
            if clean.shape != direction.shape:
                raise ValueError(f"feature {spec['feature_id']} direction shape {direction.shape} does not match activation {clean.shape}")
            clean_score = float(np.dot(clean, direction))
            perturbed_score = float(np.dot(perturbed, direction))
            if not np.all(np.isfinite([clean_score, perturbed_score, perturbed_score - clean_score])):
                raise ValueError("feature projections must be finite")
            step_deltas.append(
                {
                    "step": int(step),
                    "clean_activation": clean_score,
                    "perturbed_activation": perturbed_score,
                    "delta": perturbed_score - clean_score,
                }
            )
        first_feature = _first_significant_step(step_deltas, threshold=threshold)
        lead_time = (
            int(external_first) - int(first_feature)
            if external_first is not None and first_feature is not None
            else None
        )
        eligible = view.synthetic is False and view.comparison_complete and coverage["comparison_complete"]
        evidence_level = 1 if eligible and first_feature is not None else 0
        features.append(
            {
                "feature_id": str(spec["feature_id"]),
                "label": _candidate_label(str(spec.get("label", spec["feature_id"]))),
                "label_confidence": float(spec.get("label_confidence", 0.0)),
                "evidence_level": evidence_level,
                "synthetic": view.synthetic,
                "evidence_eligible": eligible,
                **coverage,
                "promotion_allowed": False,
                "clean_activation_mean": _mean([item["clean_activation"] for item in step_deltas]),
                "perturbed_activation_mean": _mean([item["perturbed_activation"] for item in step_deltas]),
                "first_feature_divergence_step": first_feature,
                "first_action_divergence_step": external_first,
                "lead_time_steps": lead_time,
                "negative_control_stable": bool(index.get("negative_control")) and coverage["comparison_complete"] and first_feature is None,
                "reason": _reason(evidence_level),
                "step_deltas": step_deltas,
            }
        )
    comparison_complete = bool(features) and view.comparison_complete and all(
        feature["comparison_complete"] for feature in features
    )
    eligible = view.synthetic is False and comparison_complete
    result = {
        "format": "robotprobe.mech.feature_diff.v0",
        "incident_id": index.get("incident_id"),
        "dictionary_id": dictionary.get("id"),
        "dictionary_type": dictionary.get("type"),
        "features": features,
        "synthetic": view.synthetic,
        "evidence_eligible": eligible,
        "evidence_level": max((feature["evidence_level"] for feature in features), default=0) if eligible else 0,
        "comparison_complete": comparison_complete,
        "claim_boundary": {
            "labels_are_candidates": True,
            "causal_without_intervention": False,
        },
    }
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _load_dictionary(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError("invalid feature dictionary") from exc
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        raise ValueError("dictionary must be an object with a features list")
    for spec in data["features"]:
        if not isinstance(spec, dict) or not all(key in spec for key in ("feature_id", "layer", "direction")):
            raise ValueError("feature needs feature_id, layer, and direction")
        if not isinstance(spec["direction"], list):
            raise ValueError("feature direction must be a list")
    return data


def _first_significant_step(step_deltas: list[dict[str, Any]], *, threshold: float) -> int | None:
    for item in step_deltas:
        if abs(float(item["delta"])) > threshold:
            return int(item["step"])
    return None


def _candidate_label(label: str) -> str:
    return label if label.startswith("candidate: ") else f"candidate: {label}"


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _reason(evidence_level: int) -> str:
    if evidence_level == 1:
        return "observed activation difference only; prediction and causation are untested"
    return "no eligible activation-difference evidence; fixture, unknown provenance, or insufficient comparison"
