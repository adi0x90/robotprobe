from __future__ import annotations

import copy
import dataclasses
import hashlib
import operator
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


class ExecutionMode(str, Enum):
    SIMULATION = "simulation"
    HARDWARE_SHADOW = "hardware_shadow"
    HARDWARE_EXECUTABLE = "hardware_executable"


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    instruction: str
    seed: int
    max_steps: int
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.scenario_id, "scenario_id")
        _require_non_empty(self.instruction, "instruction")
        object.__setattr__(
            self,
            "seed",
            _nonnegative_integer(self.seed, "seed"),
        )
        object.__setattr__(
            self,
            "max_steps",
            _positive_integer(self.max_steps, "max_steps"),
        )
        object.__setattr__(self, "options", _freeze_mapping(self.options))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class RawObservation:
    observation_id: str
    sequence: int
    captured_at_ns: int
    payload: Mapping[str, Any]
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.observation_id, "observation_id")
        _require_non_empty(self.source, "source")
        object.__setattr__(
            self,
            "sequence",
            _nonnegative_integer(self.sequence, "sequence"),
        )
        object.__setattr__(
            self,
            "captured_at_ns",
            _nonnegative_integer(self.captured_at_ns, "captured_at_ns"),
        )
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class CanonicalObservation:
    observation_id: str
    sequence: int
    captured_at_ns: int
    images: Mapping[str, Any]
    eef_position: Any
    eef_quaternion: Any
    gripper_qpos: Any
    instruction: str
    source: str
    frame_id: str
    calibration_id: str
    joint_positions: Any | None = None
    joint_velocities: Any | None = None
    joint_efforts: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "observation_id",
            "instruction",
            "source",
            "frame_id",
            "calibration_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "sequence",
            _nonnegative_integer(self.sequence, "sequence"),
        )
        object.__setattr__(
            self,
            "captured_at_ns",
            _nonnegative_integer(self.captured_at_ns, "captured_at_ns"),
        )
        object.__setattr__(self, "images", _freeze_images(self.images))
        object.__setattr__(self, "eef_position", _vector(self.eef_position, "eef_position", 3))
        object.__setattr__(self, "eef_quaternion", _vector(self.eef_quaternion, "eef_quaternion", 4))
        object.__setattr__(self, "gripper_qpos", _vector(self.gripper_qpos, "gripper_qpos", 2))
        object.__setattr__(
            self,
            "joint_positions",
            _optional_vector(self.joint_positions, "joint_positions", 7),
        )
        object.__setattr__(
            self,
            "joint_velocities",
            _optional_vector(self.joint_velocities, "joint_velocities", 7),
        )
        object.__setattr__(
            self,
            "joint_efforts",
            _optional_vector(self.joint_efforts, "joint_efforts", 7),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_trace_fields(self) -> dict[str, Any]:
        images = {
            name: {
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "sha256": f"sha256:{hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()}",
            }
            for name, image in self.images.items()
        }
        return {
            "observation_id": self.observation_id,
            "sequence": self.sequence,
            "captured_at_ns": self.captured_at_ns,
            "images": images,
            "eef_position": self.eef_position.tolist(),
            "eef_quaternion": self.eef_quaternion.tolist(),
            "gripper_qpos": self.gripper_qpos.tolist(),
            "joint_positions": _optional_list(self.joint_positions),
            "joint_velocities": _optional_list(self.joint_velocities),
            "joint_efforts": _optional_list(self.joint_efforts),
            "instruction": self.instruction,
            "source": self.source,
            "frame_id": self.frame_id,
            "calibration_id": self.calibration_id,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class InferenceContext:
    env_step: int
    policy_call_id: str
    action_chunk_id: str
    observation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "env_step",
            _nonnegative_integer(self.env_step, "env_step"),
        )
        for field_name in (
            "policy_call_id",
            "action_chunk_id",
            "observation_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ProposedActionChunk:
    chunk_id: str
    policy_call_id: str
    source_observation_id: str
    source_observation_captured_at_ns: int
    command_space: str
    values: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "chunk_id",
            "policy_call_id",
            "source_observation_id",
            "command_space",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "source_observation_captured_at_ns",
            _nonnegative_integer(
                self.source_observation_captured_at_ns,
                "source_observation_captured_at_ns",
            ),
        )
        values = np.array(self.values, dtype=float, copy=True)
        if values.ndim != 2:
            raise ValueError("action chunk values must have rank two")
        if values.shape[0] == 0:
            raise ValueError("action chunk must contain at least one action")
        if values.shape[1] == 0:
            raise ValueError("action chunk action dimension must be positive")
        if not np.isfinite(values).all():
            raise ValueError("action chunk values must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_trace_fields(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "policy_call_id": self.policy_call_id,
            "source_observation_id": self.source_observation_id,
            "source_observation_captured_at_ns": self.source_observation_captured_at_ns,
            "command_space": self.command_space,
            "shape": list(self.values.shape),
            "value": self.values.tolist(),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class ScheduledAction:
    command_id: str
    chunk_id: str
    policy_call_id: str
    source_observation_id: str
    source_observation_captured_at_ns: int
    action_index: int
    command_space: str
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "command_id",
            "chunk_id",
            "policy_call_id",
            "source_observation_id",
            "command_space",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "source_observation_captured_at_ns",
            _nonnegative_integer(
                self.source_observation_captured_at_ns,
                "source_observation_captured_at_ns",
            ),
        )
        object.__setattr__(
            self,
            "action_index",
            _nonnegative_integer(self.action_index, "action_index"),
        )
        object.__setattr__(self, "value", _vector(self.value, "scheduled action", None))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_trace_fields(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "chunk_id": self.chunk_id,
            "policy_call_id": self.policy_call_id,
            "source_observation_id": self.source_observation_id,
            "source_observation_captured_at_ns": self.source_observation_captured_at_ns,
            "action_index": self.action_index,
            "command_space": self.command_space,
            "value": self.value.tolist(),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class SafetyDecision:
    accepted: bool
    disposition: str
    reason: str
    checked_at_ns: int
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be boolean")
        _require_non_empty(self.disposition, "disposition")
        _require_non_empty(self.reason, "reason")
        object.__setattr__(
            self,
            "checked_at_ns",
            _nonnegative_integer(self.checked_at_ns, "checked_at_ns"),
        )
        object.__setattr__(self, "details", _freeze_mapping(self.details))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class AuthorizedAction:
    scheduled: ScheduledAction
    decision: SafetyDecision

    @property
    def value(self) -> np.ndarray:
        return self.scheduled.value

    @property
    def command_id(self) -> str:
        return self.scheduled.command_id

    def to_trace_fields(self) -> dict[str, Any]:
        return {
            **self.scheduled.to_trace_fields(),
            "decision": self.decision.to_dict(),
        }


@dataclass(frozen=True)
class ExecutionReceipt:
    command_id: str
    backend_id: str
    execution_mode: ExecutionMode
    command_applied: bool
    accepted: bool
    timestamp_ns: int
    measured_state: Mapping[str, Any] = field(default_factory=dict)
    done: bool = False
    success: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.command_id, "command_id")
        _require_non_empty(self.backend_id, "backend_id")
        mode = ExecutionMode(self.execution_mode)
        object.__setattr__(self, "execution_mode", mode)
        for field_name in (
            "command_applied",
            "accepted",
            "done",
            "success",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        object.__setattr__(
            self,
            "timestamp_ns",
            _nonnegative_integer(self.timestamp_ns, "timestamp_ns"),
        )
        if mode is ExecutionMode.HARDWARE_SHADOW and self.command_applied:
            raise ValueError("hardware_shadow receipts cannot claim command_applied=true")
        if self.command_applied and not self.accepted:
            raise ValueError("accepted=false receipts cannot claim command_applied=true")
        if self.success and not self.done:
            raise ValueError("success=true receipts must also set done=true")
        object.__setattr__(self, "measured_state", _freeze_mapping(self.measured_state))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_trace_fields(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "backend_id": self.backend_id,
            "execution_mode": self.execution_mode.value,
            "command_applied": self.command_applied,
            "accepted": self.accepted,
            "timestamp_ns": self.timestamp_ns,
            "measured_state": _json_safe(self.measured_state),
            "done": self.done,
            "success": self.success,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class BackendHealth:
    backend_id: str
    connected: bool
    healthy: bool
    timestamp_ns: int
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.backend_id, "backend_id")
        if not isinstance(self.connected, bool) or not isinstance(self.healthy, bool):
            raise ValueError("backend health connected and healthy must be boolean")
        object.__setattr__(
            self,
            "timestamp_ns",
            _nonnegative_integer(self.timestamp_ns, "timestamp_ns"),
        )
        object.__setattr__(self, "details", _freeze_mapping(self.details))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class EpisodeResult:
    run_id: str
    success: bool
    steps: int
    reason: str
    trace_path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


class RuntimePhaseError(RuntimeError):
    def __init__(self, phase: str, component_id: str, error: BaseException):
        self.phase = phase
        self.component_id = component_id
        self.original_type = type(error).__name__
        super().__init__(f"{phase} failed in {component_id}: {self.original_type}")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(copy.deepcopy(dict(value)))


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        rendered = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a non-negative integer") from error
    if rendered < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(rendered)


def _positive_integer(value: Any, field_name: str) -> int:
    rendered = _nonnegative_integer(value, field_name)
    if rendered == 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return rendered


def _freeze_images(value: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for name, image in value.items():
        array = np.array(image, copy=True)
        if array.ndim < 2:
            raise ValueError(f"image {name!r} must have at least two dimensions")
        array.setflags(write=False)
        images[str(name)] = array
    return MappingProxyType(images)


def _vector(value: Any, name: str, size: int | None) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if size is not None and array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} values must be finite")
    array.setflags(write=False)
    return array


def _optional_vector(value: Any | None, name: str, size: int) -> np.ndarray | None:
    return None if value is None else _vector(value, name, size)


def _optional_list(value: np.ndarray | None) -> list[float] | None:
    return None if value is None else value.tolist()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
