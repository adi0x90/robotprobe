from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any


class MechEvidenceLevel(IntEnum):
    SEMANTIC_LOOKS_RELEVANT = 0
    CORRELATES_WITH_INCIDENT = 1
    PREDICTS_ACTION_DIVERGENCE = 2
    INTERVENTION_CHANGES_ACTION = 3
    REPRODUCES_OR_PREVENTS_FAILURE = 4
    GENERALIZES = 5


@dataclass(frozen=True)
class ModelActivationEvent:
    run_id: str
    target: str
    scenario: str
    replay_level: str
    step: int
    module: str
    module_kind: str
    activation_ref: str
    activation_shape: list[int]
    activation_dtype: str
    model_ref: dict[str, Any]
    baseline_run_id: str | None = None
    perturbation_group_id: str | None = None
    episode_id: str | None = None
    capture: dict[str, Any] = field(
        default_factory=lambda: {
            "capture_mode": "forward_hook",
            "hook_version": "mechprobe.activation_hook.v0",
            "detached": True,
            "requires_grad": False,
        }
    )

    def to_trace_event(self) -> dict[str, Any]:
        event = dataclasses.asdict(self)
        event["event_type"] = "model_activation"
        event["layer"] = "model_activation"
        return {key: value for key, value in event.items() if value is not None}


@dataclass(frozen=True)
class FeatureFinding:
    feature_id: str
    label: str
    evidence_level: MechEvidenceLevel = MechEvidenceLevel.SEMANTIC_LOOKS_RELEVANT
    label_confidence: float = 0.0
    reason: str = "semantic label only; no incident evidence yet"

    def __post_init__(self) -> None:
        if not self.label.startswith("candidate: "):
            object.__setattr__(self, "label", f"candidate: {self.label}")

    @property
    def promotion_allowed(self) -> bool:
        return int(self.evidence_level) >= int(MechEvidenceLevel.INTERVENTION_CHANGES_ACTION)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "label": self.label,
            "label_confidence": self.label_confidence,
            "evidence_level": int(self.evidence_level),
            "promotion_allowed": self.promotion_allowed,
            "reason": self.reason,
        }


def load_capture_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ValueError("capture config is not JSON and PyYAML is not installed") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid capture config YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("capture config must be a mapping")
    return data


def validate_capture_config(config: Any) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(config, dict):
        return _validation_result(["capture config must be a mapping"], warnings)
    if config.get("format") != "robotprobe.mech.capture_config.v0":
        errors.append("format")
    if not isinstance(config.get("id"), str) or not config["id"].strip():
        errors.append("id")
    if config.get("model_access") != "white_box":
        errors.append("model_access")
    layers = config.get("layers")
    if not isinstance(layers, list) or not layers:
        errors.append("layers")
    else:
        names = set()
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict) or not isinstance(layer.get("name"), str) or not layer["name"]:
                errors.append(f"layers[{index}].name")
                continue
            if layer["name"] in names:
                errors.append(f"layers[{index}].name duplicate")
            names.add(layer["name"])
            if layer.get("capture", "output") != "output":
                errors.append(f"layers[{index}].capture")
    storage = config.get("storage")
    if not isinstance(storage, dict):
        errors.append("storage")
    else:
        if storage.get("format") not in ("safetensors", "npy"):
            errors.append("storage.format")
        if storage.get("dtype") not in (None, "float16", "float32", "float64"):
            errors.append("storage.dtype")
        budget = storage.get("max_bytes_per_run")
        if type(budget) is not int or budget <= 0:
            errors.append("storage.max_bytes_per_run")
    if "steps" in config:
        steps = config["steps"]
        if not isinstance(steps, dict):
            errors.append("steps")
        else:
            if steps.get("mode", "every_n") != "every_n":
                errors.append("steps.mode")
            if type(steps.get("n", 1)) is not int or steps.get("n", 1) <= 0:
                errors.append("steps.n")
            maximum = steps.get("max_steps")
            if maximum is not None and (type(maximum) is not int or maximum < 0):
                errors.append("steps.max_steps")
    claim_boundary = config.get("claim_boundary")
    if not isinstance(claim_boundary, dict):
        errors.append("claim_boundary")
    else:
        if claim_boundary.get("no_feature_labels") is not True:
            warnings.append("claim_boundary.no_feature_labels should be true for v0 capture")
        if claim_boundary.get("no_causal_claims") is not True:
            warnings.append("claim_boundary.no_causal_claims should be true for v0 capture")
    return _validation_result(errors, warnings)


def _validation_result(errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "format": "robotprobe.mech.capture_config.validation.v0",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }
