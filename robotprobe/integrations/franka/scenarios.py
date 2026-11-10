from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.types import ScenarioSpec


@dataclass(frozen=True)
class DeterministicReachScenario:
    scenario_id: str = "deterministic-franka-reach"
    instruction: str = "Move the Franka end effector to the target."
    seed: int = 7
    max_steps: int = 20
    initial_position: tuple[float, float, float] = (0.4, 0.0, 0.3)
    target_position: tuple[float, float, float] = (0.5, 0.0, 0.3)
    tolerance: float = 0.005
    workspace_min: tuple[float, float, float] = (0.2, -0.5, 0.1)
    workspace_max: tuple[float, float, float] = (0.7, 0.5, 0.8)
    initial_position_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0)

    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "scenario",
            "component_id": "franka.scenario.deterministic_reach.v0",
            "command_spaces": [],
            "observation_streams": [],
            "runtime_family": "franka_deterministic",
            "is_hardware": False,
            "supports_reset": False,
            "supports_ground_truth": False,
            "supports_shadow": False,
            "can_apply_hardware_commands": False,
            "requires_reset": True,
            "optional_dependencies": [],
        }
    )

    def __post_init__(self) -> None:
        initial = _vector(self.initial_position, "initial_position")
        target = _vector(self.target_position, "target_position")
        lower = _vector(self.workspace_min, "workspace_min")
        upper = _vector(self.workspace_max, "workspace_max")
        jitter = _vector(self.initial_position_jitter, "initial_position_jitter")
        if not np.all(lower < upper):
            raise ValueError("workspace_min must be strictly below workspace_max")
        if np.any(initial < lower) or np.any(initial > upper):
            raise ValueError("initial_position must lie inside the workspace")
        if np.any(target < lower) or np.any(target > upper):
            raise ValueError("target_position must lie inside the workspace")
        if np.any(jitter < 0):
            raise ValueError("initial_position_jitter must be non-negative")
        if self.tolerance <= 0 or not np.isfinite(self.tolerance):
            raise ValueError("tolerance must be positive and finite")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    def build(self) -> ScenarioSpec:
        return ScenarioSpec(
            scenario_id=self.scenario_id,
            instruction=self.instruction,
            seed=self.seed,
            max_steps=self.max_steps,
            options={
                "initial_position": list(self.initial_position),
                "target_position": list(self.target_position),
                "tolerance": self.tolerance,
                "workspace_min": list(self.workspace_min),
                "workspace_max": list(self.workspace_max),
                "initial_position_jitter": list(self.initial_position_jitter),
                "simulator_kind": "deterministic_kinematic_contract",
            },
        )


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite three-dimensional vector")
    return array
