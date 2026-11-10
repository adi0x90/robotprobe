from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from robotprobe.integrations.franka.embodiment import (
    FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
)
from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.types import (
    AuthorizedAction,
    BackendHealth,
    ExecutionMode,
    ExecutionReceipt,
    RawObservation,
    ScenarioSpec,
)


class FrankaRecordedReplayBackend:
    """Evaluate actions against synthetic recorded-format snapshots without motion."""

    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "backend",
            "component_id": "franka.recorded_snapshot_replay.v0",
            "command_spaces": [FRANKA_CARTESIAN_DELTA_COMMAND_SPACE],
            "observation_streams": [
                "synthetic_rgb",
                "eef_pose",
                "gripper",
                "joint_state",
            ],
            "runtime_family": "franka_recorded_replay",
            "is_hardware": False,
            "supports_reset": True,
            "supports_ground_truth": False,
            "supports_shadow": False,
            "can_apply_hardware_commands": False,
            "requires_reset": False,
            "optional_dependencies": [],
        }
    )

    def __init__(
        self,
        *,
        snapshots: Sequence[Mapping[str, Any]],
        fixture_provenance: str = "synthetic_recorded_format",
        clock_ns: Callable[[], int] = time.time_ns,
    ):
        if fixture_provenance != "synthetic_recorded_format":
            raise ValueError(
                "fixture_provenance must be synthetic_recorded_format"
            )
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._snapshots = [
            _validated_snapshot(snapshot, index)
            for index, snapshot in enumerate(snapshots)
        ]
        if not self._snapshots:
            raise ValueError("recorded replay requires at least one snapshot")
        self.fixture_provenance = fixture_provenance
        self.clock_ns = clock_ns
        self._connected = False
        self._scenario: ScenarioSpec | None = None
        self._last_raw: RawObservation | None = None
        self._sequence = 0
        self._snapshot_index = 0

    def connect(self) -> None:
        self._connected = True

    def reset(self, scenario: ScenarioSpec) -> RawObservation:
        if not self._connected:
            raise RuntimeError("Franka recorded replay backend is disconnected")
        if not isinstance(scenario, ScenarioSpec):
            raise TypeError("Franka recorded replay reset requires ScenarioSpec")
        self._scenario = scenario
        self._sequence = 0
        self._snapshot_index = 0
        self._last_raw = self._raw_observation(self._snapshots[0], sequence=0)
        return self._last_raw

    def observe(self) -> RawObservation:
        self._require_ready()
        self._sequence += 1
        self._snapshot_index = min(
            self._snapshot_index + 1,
            len(self._snapshots) - 1,
        )
        self._last_raw = self._raw_observation(
            self._snapshots[self._snapshot_index],
            sequence=self._sequence,
        )
        return self._last_raw

    def execute(self, action: AuthorizedAction) -> ExecutionReceipt:
        self._require_ready()
        if not isinstance(action, AuthorizedAction):
            raise TypeError("Franka recorded replay requires AuthorizedAction")
        if not action.decision.accepted:
            raise ValueError(
                "Franka recorded replay requires an authorized action"
            )
        if action.scheduled.command_space != FRANKA_CARTESIAN_DELTA_COMMAND_SPACE:
            raise ValueError(
                "Franka recorded replay received an unsupported command space"
            )
        assert self._last_raw is not None
        if (
            action.scheduled.source_observation_id
            != self._last_raw.observation_id
        ):
            raise ValueError(
                "Franka recorded replay action source observation is stale"
            )
        value = np.asarray(action.value, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(
                "Franka recorded replay action must be a finite seven-value vector"
            )
        return ExecutionReceipt(
            command_id=action.command_id,
            backend_id=self.capabilities.component_id,
            execution_mode=ExecutionMode.SIMULATION,
            command_applied=False,
            accepted=True,
            timestamp_ns=self.clock_ns(),
            measured_state={
                "observation_id": self._last_raw.observation_id,
                "eef_position": self._last_raw.payload["eef_position"].tolist(),
                "eef_quaternion": self._last_raw.payload[
                    "eef_quaternion"
                ].tolist(),
                "gripper_qpos": self._last_raw.payload["gripper_qpos"].tolist(),
                "joint_positions": self._last_raw.payload[
                    "joint_positions"
                ].tolist(),
            },
            done=False,
            success=False,
            metadata={
                "reason": "recorded_snapshot_evaluated_not_applied",
                "fixture_provenance": self.fixture_provenance,
                "read_only": True,
            },
        )

    def health(self) -> BackendHealth:
        return BackendHealth(
            backend_id=self.capabilities.component_id,
            connected=self._connected,
            healthy=self._connected,
            timestamp_ns=self.clock_ns(),
            details={
                "read_only": True,
                "fixture_provenance": self.fixture_provenance,
                "snapshots": len(self._snapshots),
                "reset": self._scenario is not None,
            },
        )

    def close(self) -> None:
        self._connected = False
        self._scenario = None
        self._last_raw = None
        self._sequence = 0
        self._snapshot_index = 0

    def _raw_observation(
        self,
        snapshot: Mapping[str, Any],
        *,
        sequence: int,
    ) -> RawObservation:
        image_specs = snapshot["image_specs"]
        images = {
            str(role): np.full(
                tuple(spec["shape"]),
                spec["fill"],
                dtype=np.uint8,
            )
            for role, spec in image_specs.items()
        }
        return RawObservation(
            observation_id=f"franka-replay:{sequence:06d}",
            sequence=sequence,
            captured_at_ns=self.clock_ns(),
            payload={
                "images": images,
                "eef_position": np.asarray(
                    snapshot["eef_position"],
                    dtype=np.float64,
                ),
                "eef_quaternion": np.asarray(
                    snapshot["eef_quaternion"],
                    dtype=np.float64,
                ),
                "gripper_qpos": np.asarray(
                    snapshot["gripper_qpos"],
                    dtype=np.float64,
                ),
                "joint_positions": np.asarray(
                    snapshot["joint_positions"],
                    dtype=np.float64,
                ),
                "joint_velocities": np.asarray(
                    snapshot["joint_velocities"],
                    dtype=np.float64,
                ),
                "joint_efforts": np.asarray(
                    snapshot["joint_efforts"],
                    dtype=np.float64,
                ),
                "instruction": self._scenario.instruction,
                "frame_id": "franka_link0",
                "calibration_id": "synthetic-recorded-format-v0",
                "quaternion_convention": "xyzw",
            },
            source=self.capabilities.component_id,
            metadata={
                "fixture_provenance": self.fixture_provenance,
                "read_only": True,
                "snapshot_index": self._snapshot_index,
            },
        )

    def _require_ready(self) -> None:
        if (
            not self._connected
            or self._scenario is None
            or self._last_raw is None
        ):
            raise RuntimeError(
                "Franka recorded replay backend must connect and reset before use"
            )


def _validated_snapshot(
    value: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"snapshot {index} must be a mapping")
    image_specs = value.get("image_specs")
    if not isinstance(image_specs, Mapping) or not image_specs:
        raise ValueError(f"snapshot {index} image_specs must be a non-empty mapping")
    normalized_specs: dict[str, dict[str, Any]] = {}
    for role, spec in image_specs.items():
        if not isinstance(role, str) or not role:
            raise ValueError(f"snapshot {index} image role must be non-empty")
        if not isinstance(spec, Mapping):
            raise TypeError(f"snapshot {index} image spec {role!r} must be a mapping")
        shape = spec.get("shape")
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 3
            or not all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in shape
            )
        ):
            raise ValueError(
                f"snapshot {index} image {role!r} must have a positive "
                "three-dimensional shape"
            )
        if shape[2] != 3:
            raise ValueError(
                f"snapshot {index} image {role!r} must have three channels"
            )
        fill = spec.get("fill")
        if (
            not isinstance(fill, int)
            or isinstance(fill, bool)
            or not 0 <= fill <= 255
        ):
            raise ValueError(
                f"snapshot {index} image {role!r} fill must be an integer "
                "within [0, 255]"
            )
        normalized_specs[role] = {"shape": tuple(shape), "fill": fill}

    normalized: dict[str, Any] = {"image_specs": normalized_specs}
    for field_name, length in (
        ("eef_position", 3),
        ("eef_quaternion", 4),
        ("gripper_qpos", 2),
        ("joint_positions", 7),
        ("joint_velocities", 7),
        ("joint_efforts", 7),
    ):
        normalized[field_name] = _vector(
            value.get(field_name),
            f"snapshot {index} {field_name}",
            length,
        )
    return normalized


def _vector(value: Any, field_name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(
            f"{field_name} must be a finite vector with length {length}"
        )
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy
