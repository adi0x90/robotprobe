from __future__ import annotations

import operator
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from robotprobe.integrations.franka.embodiment import FRANKA_CARTESIAN_DELTA_COMMAND_SPACE
from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.types import (
    AuthorizedAction,
    BackendHealth,
    ExecutionMode,
    ExecutionReceipt,
    RawObservation,
    ScenarioSpec,
)


class FrankaShadowOptionalDependencyError(RuntimeError):
    pass


class FrankaShadowStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrankaTransportHealth:
    connected: bool
    healthy: bool
    captured_at_ns: int
    robot_mode: str
    controller_authority: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.connected, bool) or not isinstance(
            self.healthy,
            bool,
        ):
            raise ValueError(
                "transport health connected and healthy must be boolean"
            )
        object.__setattr__(
            self,
            "captured_at_ns",
            _timestamp_ns(self.captured_at_ns, "transport health timestamp"),
        )
        for field_name in ("robot_mode", "controller_authority"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"transport health {field_name} must be a non-empty string"
                )
        object.__setattr__(
            self,
            "details",
            MappingProxyType(dict(self.details)),
        )


@dataclass(frozen=True)
class FrankaShadowSnapshot:
    captured_at_ns: int
    images: Mapping[str, Any]
    eef_position: Any
    eef_quaternion: Any
    gripper_qpos: Any
    joint_positions: Any
    joint_velocities: Any
    joint_efforts: Any
    frame_id: str
    calibration_id: str
    robot_mode: str
    controller_authority: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "captured_at_ns",
            _timestamp_ns(self.captured_at_ns, "Franka shadow timestamp"),
        )
        images: dict[str, np.ndarray] = {}
        for role, value in self.images.items():
            if not role:
                raise ValueError("Franka shadow image roles must be non-empty")
            image = np.array(value, copy=True)
            if image.ndim != 3:
                raise ValueError(
                    f"Franka shadow image {role!r} must have three dimensions"
                )
            image.setflags(write=False)
            images[str(role)] = image
        if not images:
            raise ValueError("Franka shadow snapshot must contain camera images")
        object.__setattr__(self, "images", MappingProxyType(images))
        object.__setattr__(
            self,
            "eef_position",
            _readonly_vector(self.eef_position, "eef_position", 3),
        )
        object.__setattr__(
            self,
            "eef_quaternion",
            _readonly_vector(self.eef_quaternion, "eef_quaternion", 4),
        )
        object.__setattr__(
            self,
            "gripper_qpos",
            _readonly_vector(self.gripper_qpos, "gripper_qpos", 2),
        )
        object.__setattr__(
            self,
            "joint_positions",
            _readonly_vector(self.joint_positions, "joint_positions", 7),
        )
        object.__setattr__(
            self,
            "joint_velocities",
            _readonly_vector(self.joint_velocities, "joint_velocities", 7),
        )
        object.__setattr__(
            self,
            "joint_efforts",
            _readonly_vector(self.joint_efforts, "joint_efforts", 7),
        )
        for name in (
            "frame_id",
            "calibration_id",
            "robot_mode",
            "controller_authority",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Franka shadow {name} must be a non-empty string"
                )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@runtime_checkable
class FrankaShadowTransport(Protocol):
    """Read-only transport surface; command publication is intentionally absent."""

    def connect(self) -> None: ...

    def health(self) -> FrankaTransportHealth: ...

    def read_snapshot(self) -> FrankaShadowSnapshot: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class FrankaShadowScenario:
    scenario_id: str = "franka-shadow-observation"
    instruction: str = "Observe the Franka while evaluating a proposed manipulation."
    seed: int = 7
    max_steps: int = 20
    options: Mapping[str, Any] = field(default_factory=dict)

    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "scenario",
            "component_id": "franka.scenario.hardware_shadow.v0",
            "command_spaces": [],
            "observation_streams": [],
            "runtime_family": "franka_hardware_shadow",
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
        for field_name in ("scenario_id", "instruction"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Franka shadow {field_name} must be a non-empty string"
                )
        object.__setattr__(
            self,
            "seed",
            _timestamp_ns(self.seed, "Franka shadow seed"),
        )
        max_steps = _timestamp_ns(self.max_steps, "Franka shadow max_steps")
        if max_steps == 0:
            raise ValueError("Franka shadow max_steps must be a positive integer")
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def build(self) -> ScenarioSpec:
        return ScenarioSpec(
            scenario_id=self.scenario_id,
            instruction=self.instruction,
            seed=self.seed,
            max_steps=self.max_steps,
            options={
                **dict(self.options),
                "observation_only": True,
                "reset_semantics": "logical_session_only",
            },
        )


class FrankaRos2ShadowBackend:
    """Read physical Franka state and evaluate actions without a command path."""

    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "backend",
            "component_id": "franka.ros2_read_only_shadow.v0",
            "command_spaces": [FRANKA_CARTESIAN_DELTA_COMMAND_SPACE],
            "observation_streams": [
                "agent_rgb",
                "wrist_rgb",
                "eef_pose",
                "gripper",
                "joint_state",
                "robot_mode",
                "controller_authority",
            ],
            "runtime_family": "franka_hardware_shadow",
            "is_hardware": True,
            "supports_reset": True,
            "supports_ground_truth": False,
            "supports_shadow": True,
            "can_apply_hardware_commands": False,
            "requires_reset": False,
            "optional_dependencies": ["robotprobe-franka-ros2"],
        }
    )

    def __init__(
        self,
        *,
        transport: FrankaShadowTransport | None = None,
        transport_options: Mapping[str, Any] | None = None,
        transport_loader: Callable[
            [Mapping[str, Any]], FrankaShadowTransport
        ]
        | None = None,
        max_state_age_ms: float = 250.0,
        required_image_roles: tuple[str, ...] = ("agent", "wrist"),
        clock_ns: Callable[[], int] = time.time_ns,
    ):
        if max_state_age_ms <= 0 or not np.isfinite(max_state_age_ms):
            raise ValueError("max_state_age_ms must be positive and finite")
        if not required_image_roles or not all(required_image_roles):
            raise ValueError("required_image_roles must contain non-empty names")
        if len(set(required_image_roles)) != len(required_image_roles):
            raise ValueError("required_image_roles must not contain duplicates")
        self._provided_transport = transport
        self._transport_options = dict(transport_options or {})
        self._transport_loader = transport_loader or _default_transport_loader
        self.max_state_age_ms = float(max_state_age_ms)
        self.required_image_roles = tuple(required_image_roles)
        self.clock_ns = clock_ns
        self._transport: FrankaShadowTransport | None = None
        self._connected = False
        self._scenario: ScenarioSpec | None = None
        self._last_raw: RawObservation | None = None
        self._observation_sequence = 0

    def connect(self) -> None:
        if self._connected:
            return
        try:
            transport = (
                self._provided_transport
                if self._provided_transport is not None
                else self._transport_loader(self._transport_options)
            )
        except ImportError as error:
            raise FrankaShadowOptionalDependencyError(
                "The read-only Franka ROS 2 transport is unavailable at connect(); "
                "install robotprobe-franka-ros2 or inject a read-only transport."
            ) from error
        if not isinstance(transport, FrankaShadowTransport):
            raise TypeError(
                "Franka shadow transport must implement connect, health, "
                "read_snapshot, and close"
            )

        self._transport = transport
        try:
            transport.connect()
            health = self._transport_health()
            if not health.connected:
                raise FrankaShadowStateError("Franka shadow transport is disconnected")
            if not health.healthy:
                raise FrankaShadowStateError("Franka shadow transport is unhealthy")
        except Exception:
            self._close_transport()
            raise
        self._connected = True

    def reset(self, scenario: ScenarioSpec) -> RawObservation:
        self._require_connected()
        if not isinstance(scenario, ScenarioSpec):
            raise TypeError("Franka shadow reset requires ScenarioSpec")
        raw = self._read_raw(scenario, sequence=0)
        self._scenario = scenario
        self._last_raw = raw
        self._observation_sequence = 0
        return raw

    def observe(self) -> RawObservation:
        self._require_ready()
        assert self._scenario is not None
        next_sequence = self._observation_sequence + 1
        raw = self._read_raw(self._scenario, sequence=next_sequence)
        self._observation_sequence = next_sequence
        self._last_raw = raw
        return raw

    def execute(self, action: AuthorizedAction) -> ExecutionReceipt:
        self._require_ready()
        if not isinstance(action, AuthorizedAction):
            raise TypeError("Franka shadow backend requires AuthorizedAction")
        if not action.decision.accepted:
            raise ValueError("Franka shadow backend refuses rejected actions")
        if action.decision.disposition != "shadow_only":
            raise ValueError(
                "Franka shadow backend requires a shadow_only authorization"
            )
        if action.scheduled.command_space != FRANKA_CARTESIAN_DELTA_COMMAND_SPACE:
            raise ValueError("Franka shadow backend received an unsupported command space")
        value = np.asarray(action.value, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("Franka shadow action must be a finite seven-value vector")

        assert self._last_raw is not None
        measured_state = _measured_state(self._last_raw)
        return ExecutionReceipt(
            command_id=action.command_id,
            backend_id=self.capabilities.component_id,
            execution_mode=ExecutionMode.HARDWARE_SHADOW,
            command_applied=False,
            accepted=True,
            timestamp_ns=self.clock_ns(),
            measured_state=measured_state,
            done=False,
            success=False,
            metadata={
                "reason": "shadow_evaluated_not_applied",
                "robot_mode": self._last_raw.payload["robot_mode"],
                "controller_authority": self._last_raw.payload[
                    "controller_authority"
                ],
                "transport_command_surface": "absent",
            },
        )

    def health(self) -> BackendHealth:
        if not self._connected or self._transport is None:
            return BackendHealth(
                backend_id=self.capabilities.component_id,
                connected=False,
                healthy=False,
                timestamp_ns=self.clock_ns(),
                details={
                    "read_only": True,
                    "transport_command_surface": "absent",
                },
            )
        transport_health = self._transport_health()
        return BackendHealth(
            backend_id=self.capabilities.component_id,
            connected=transport_health.connected,
            healthy=transport_health.healthy,
            timestamp_ns=transport_health.captured_at_ns,
            details={
                **dict(transport_health.details),
                "robot_mode": transport_health.robot_mode,
                "controller_authority": transport_health.controller_authority,
                "read_only": True,
                "transport_command_surface": "absent",
                "reset": self._scenario is not None,
            },
        )

    def close(self) -> None:
        try:
            self._close_transport()
        finally:
            self._scenario = None
            self._last_raw = None
            self._observation_sequence = 0

    def _read_raw(self, scenario: ScenarioSpec, *, sequence: int) -> RawObservation:
        health = self._transport_health()
        if not health.connected:
            raise FrankaShadowStateError("Franka shadow transport is disconnected")
        if not health.healthy:
            raise FrankaShadowStateError("Franka shadow transport is unhealthy")
        assert self._transport is not None
        snapshot = self._transport.read_snapshot()
        if not isinstance(snapshot, FrankaShadowSnapshot):
            raise TypeError(
                "Franka shadow transport must return FrankaShadowSnapshot"
            )
        missing_roles = [
            role for role in self.required_image_roles if role not in snapshot.images
        ]
        if missing_roles:
            raise FrankaShadowStateError(
                f"Franka shadow snapshot is missing image roles: {missing_roles}"
            )
        state_age_ns = self._validate_freshness(snapshot.captured_at_ns)
        return RawObservation(
            observation_id=f"franka-shadow:{sequence:06d}",
            sequence=sequence,
            captured_at_ns=snapshot.captured_at_ns,
            payload={
                "images": dict(snapshot.images),
                "eef_position": snapshot.eef_position,
                "eef_quaternion": snapshot.eef_quaternion,
                "gripper_qpos": snapshot.gripper_qpos,
                "joint_positions": snapshot.joint_positions,
                "joint_velocities": snapshot.joint_velocities,
                "joint_efforts": snapshot.joint_efforts,
                "instruction": scenario.instruction,
                "frame_id": snapshot.frame_id,
                "calibration_id": snapshot.calibration_id,
                "quaternion_convention": "xyzw",
                "robot_mode": snapshot.robot_mode,
                "controller_authority": snapshot.controller_authority,
            },
            source=self.capabilities.component_id,
            metadata={
                **dict(snapshot.metadata),
                "state_age_ns": state_age_ns,
                "read_only": True,
                "reset_semantics": "logical_session_only",
                "transport_command_surface": "absent",
                "robot_mode": snapshot.robot_mode,
                "controller_authority": snapshot.controller_authority,
            },
        )

    def _validate_freshness(self, captured_at_ns: int) -> int:
        try:
            now_ns = _timestamp_ns(self.clock_ns(), "Franka shadow clock")
        except ValueError as error:
            raise FrankaShadowStateError(
                "Franka shadow clock must return a non-negative integer timestamp"
            ) from error
        age_ns = now_ns - captured_at_ns
        if age_ns < 0:
            raise FrankaShadowStateError(
                "Franka shadow state has clock skew from a future timestamp"
            )
        if age_ns > self.max_state_age_ms * 1_000_000:
            raise FrankaShadowStateError("Franka shadow state is stale")
        return age_ns

    def _transport_health(self) -> FrankaTransportHealth:
        if self._transport is None:
            raise FrankaShadowStateError("Franka shadow transport is disconnected")
        health = self._transport.health()
        if not isinstance(health, FrankaTransportHealth):
            raise TypeError(
                "Franka shadow transport health() must return FrankaTransportHealth"
            )
        return health

    def _require_connected(self) -> None:
        if not self._connected or self._transport is None:
            raise FrankaShadowStateError(
                "Franka shadow backend must connect before reset"
            )

    def _require_ready(self) -> None:
        if (
            not self._connected
            or self._transport is None
            or self._scenario is None
            or self._last_raw is None
        ):
            raise FrankaShadowStateError(
                "Franka shadow backend must connect and reset before use"
            )

    def _close_transport(self) -> None:
        transport = self._transport
        self._transport = None
        self._connected = False
        if transport is not None:
            transport.close()


def _default_transport_loader(
    options: Mapping[str, Any],
) -> FrankaShadowTransport:
    from robotprobe_franka_ros2 import create_shadow_transport

    return create_shadow_transport(**dict(options))


def _readonly_vector(value: Any, field_name: str, length: int) -> np.ndarray:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(
            f"Franka shadow {field_name} must be a finite {length}-value vector"
        )
    vector.setflags(write=False)
    return vector


def _timestamp_ns(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        rendered = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a non-negative integer") from error
    if rendered < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(rendered)


def _measured_state(raw: RawObservation) -> dict[str, Any]:
    payload = raw.payload
    return {
        "observation_id": raw.observation_id,
        "captured_at_ns": raw.captured_at_ns,
        "eef_position": np.asarray(payload["eef_position"]).tolist(),
        "eef_quaternion": np.asarray(payload["eef_quaternion"]).tolist(),
        "gripper_qpos": np.asarray(payload["gripper_qpos"]).tolist(),
        "joint_positions": np.asarray(payload["joint_positions"]).tolist(),
        "joint_velocities": np.asarray(payload["joint_velocities"]).tolist(),
        "joint_efforts": np.asarray(payload["joint_efforts"]).tolist(),
        "robot_mode": payload["robot_mode"],
        "controller_authority": payload["controller_authority"],
    }
