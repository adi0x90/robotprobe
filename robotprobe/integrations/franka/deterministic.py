from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from robotprobe.integrations.franka.embodiment import FRANKA_CARTESIAN_DELTA_COMMAND_SPACE
from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.types import (
    AuthorizedAction,
    BackendHealth,
    CanonicalObservation,
    ExecutionMode,
    ExecutionReceipt,
    InferenceContext,
    ProposedActionChunk,
    RawObservation,
    ScenarioSpec,
)


class DeterministicReachPolicy:
    """Small deterministic policy used to prove scheduling and backend contracts."""

    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "policy",
            "component_id": "franka.policy.deterministic_reach.v0",
            "command_spaces": [FRANKA_CARTESIAN_DELTA_COMMAND_SPACE],
            "observation_streams": ["rgb", "eef_pose", "gripper", "joint_state"],
            "runtime_family": "franka_deterministic",
            "is_hardware": False,
            "supports_reset": False,
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
        step_size: float = 0.02,
        action_horizon: int = 2,
        gripper_command: float = 1.0,
    ):
        if step_size <= 0 or not np.isfinite(step_size):
            raise ValueError("step_size must be positive and finite")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if not -1.0 <= gripper_command <= 1.0:
            raise ValueError("gripper_command must be within [-1, 1]")
        self.step_size = float(step_size)
        self.action_horizon = action_horizon
        self.gripper_command = float(gripper_command)
        self._loaded = False
        self._target: np.ndarray | None = None

    def load(self) -> None:
        self._loaded = True

    def reset(self, scenario: ScenarioSpec) -> None:
        if not self._loaded:
            raise RuntimeError("deterministic policy must be loaded before reset")
        self._target = _option_vector(scenario, "target_position")

    def predict(
        self,
        observation: CanonicalObservation,
        context: InferenceContext,
    ) -> ProposedActionChunk:
        if not self._loaded or self._target is None:
            raise RuntimeError("deterministic policy must be loaded and reset before prediction")
        remaining = np.array(self._target - observation.eef_position, copy=True)
        actions = np.zeros((self.action_horizon, 7), dtype=np.float64)
        for index in range(self.action_horizon):
            distance = float(np.linalg.norm(remaining))
            if distance > 0:
                translation = remaining if distance <= self.step_size else remaining / distance * self.step_size
                actions[index, :3] = translation
                remaining -= translation
            actions[index, 6] = self.gripper_command
        return ProposedActionChunk(
            chunk_id=context.action_chunk_id,
            policy_call_id=context.policy_call_id,
            source_observation_id=context.observation_id,
            source_observation_captured_at_ns=observation.captured_at_ns,
            command_space=FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
            values=actions,
            metadata={
                "policy": self.capabilities.component_id,
                "policy_kind": "deterministic_contract_test",
                "target_position": self._target.tolist(),
            },
        )

    def close(self) -> None:
        self._target = None
        self._loaded = False


class DeterministicFrankaBackend:
    """Deterministic kinematic contract backend; this is not a physics simulator."""

    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "backend",
            "component_id": "franka.deterministic_kinematic.v0",
            "command_spaces": [FRANKA_CARTESIAN_DELTA_COMMAND_SPACE],
            "observation_streams": ["synthetic_rgb", "eef_pose", "gripper", "joint_state"],
            "runtime_family": "franka_deterministic",
            "is_hardware": False,
            "supports_reset": True,
            "supports_ground_truth": True,
            "supports_shadow": False,
            "can_apply_hardware_commands": False,
            "requires_reset": False,
            "optional_dependencies": [],
        }
    )

    def __init__(self, *, clock_ns: Callable[[], int] = time.time_ns):
        self.clock_ns = clock_ns
        self._connected = False
        self._scenario: ScenarioSpec | None = None
        self._observation_sequence = 0
        self._applied_steps = 0
        self._position = np.zeros(3, dtype=np.float64)
        self._quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._gripper_qpos = np.array([0.02, 0.02], dtype=np.float64)
        self._joint_positions = np.zeros(7, dtype=np.float64)
        self._joint_velocities = np.zeros(7, dtype=np.float64)
        self._joint_efforts = np.zeros(7, dtype=np.float64)
        self._target = np.zeros(3, dtype=np.float64)
        self._workspace_min = np.zeros(3, dtype=np.float64)
        self._workspace_max = np.zeros(3, dtype=np.float64)
        self._tolerance = 0.0

    def connect(self) -> None:
        self._connected = True

    def reset(self, scenario: ScenarioSpec) -> RawObservation:
        if not self._connected:
            raise RuntimeError("deterministic Franka backend must connect before reset")
        initial = _option_vector(scenario, "initial_position")
        jitter = _option_vector(scenario, "initial_position_jitter")
        rng = np.random.default_rng(scenario.seed)
        self._position = initial + rng.uniform(-jitter, jitter)
        self._target = _option_vector(scenario, "target_position")
        self._workspace_min = _option_vector(scenario, "workspace_min")
        self._workspace_max = _option_vector(scenario, "workspace_max")
        if np.any(self._position < self._workspace_min) or np.any(
            self._position > self._workspace_max
        ):
            raise ValueError("seeded initial Franka position is outside the workspace")
        self._tolerance = float(scenario.options["tolerance"])
        self._scenario = scenario
        self._observation_sequence = 0
        self._applied_steps = 0
        self._quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._gripper_qpos = np.array([0.02, 0.02], dtype=np.float64)
        self._joint_positions.fill(0.0)
        self._joint_velocities.fill(0.0)
        self._joint_efforts.fill(0.0)
        return self._raw_observation()

    def observe(self) -> RawObservation:
        self._require_ready("observe")
        self._observation_sequence += 1
        return self._raw_observation()

    def execute(self, action: AuthorizedAction) -> ExecutionReceipt:
        self._require_ready("execute")
        if not isinstance(action, AuthorizedAction):
            raise TypeError("deterministic Franka backend requires AuthorizedAction")
        if not action.decision.accepted:
            raise ValueError("deterministic Franka backend refuses unauthorized actions")
        if action.scheduled.command_space != FRANKA_CARTESIAN_DELTA_COMMAND_SPACE:
            raise ValueError("deterministic Franka backend received an unsupported command space")

        value = np.asarray(action.value, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("deterministic Franka action must be a finite seven-value vector")
        proposed_position = self._position + value[:3]
        if np.any(proposed_position < self._workspace_min) or np.any(
            proposed_position > self._workspace_max
        ):
            return self._receipt(
                action,
                accepted=False,
                command_applied=False,
                done=True,
                success=False,
                reason="backend_workspace_violation",
            )

        self._position = proposed_position
        self._quaternion = _apply_axis_angle_delta(self._quaternion, value[3:6])
        finger_position = 0.04 * (float(np.clip(value[6], -1.0, 1.0)) + 1.0) / 2.0
        self._gripper_qpos[:] = finger_position
        self._applied_steps += 1
        distance = float(np.linalg.norm(self._target - self._position))
        success = distance <= self._tolerance
        assert self._scenario is not None
        exhausted = self._applied_steps >= self._scenario.max_steps
        return self._receipt(
            action,
            accepted=True,
            command_applied=True,
            done=success or exhausted,
            success=success,
            reason="target_reached" if success else ("backend_step_limit" if exhausted else "applied"),
        )

    def health(self) -> BackendHealth:
        return BackendHealth(
            backend_id=self.capabilities.component_id,
            connected=self._connected,
            healthy=self._connected,
            timestamp_ns=self.clock_ns(),
            details={
                "runtime_kind": "deterministic_kinematic_contract",
                "reset": self._scenario is not None,
            },
        )

    def close(self) -> None:
        self._connected = False
        self._scenario = None

    def _require_ready(self, operation: str) -> None:
        if not self._connected or self._scenario is None:
            raise RuntimeError(
                f"deterministic Franka backend must connect and reset before {operation}"
            )

    def _raw_observation(self) -> RawObservation:
        assert self._scenario is not None
        return RawObservation(
            observation_id=f"franka-sim:{self._observation_sequence:06d}",
            sequence=self._observation_sequence,
            captured_at_ns=self.clock_ns(),
            payload={
                "images": self._synthetic_images(),
                "eef_position": self._position,
                "eef_quaternion": self._quaternion,
                "gripper_qpos": self._gripper_qpos,
                "joint_positions": self._joint_positions,
                "joint_velocities": self._joint_velocities,
                "joint_efforts": self._joint_efforts,
                "instruction": self._scenario.instruction,
                "frame_id": "franka_link0",
                "calibration_id": "deterministic-franka-v0",
                "quaternion_convention": "xyzw",
            },
            source=self.capabilities.component_id,
            metadata={
                "ground_truth": True,
                "simulator_kind": "deterministic_kinematic_contract",
            },
        )

    def _synthetic_images(self) -> dict[str, np.ndarray]:
        normalized = np.clip(
            (self._position - np.array([0.2, -0.5, 0.1]))
            / np.array([0.5, 1.0, 0.7]),
            0.0,
            1.0,
        )
        color = np.rint(normalized * 255).astype(np.uint8)
        agent = np.broadcast_to(color, (8, 8, 3)).copy()
        wrist = np.flip(agent, axis=1).copy()
        wrist[:, :, 2] = np.uint8(self._observation_sequence % 256)
        return {"agent": agent, "wrist": wrist}

    def _measured_state(self) -> dict[str, Any]:
        return {
            "eef_position": self._position.tolist(),
            "eef_quaternion": self._quaternion.tolist(),
            "gripper_qpos": self._gripper_qpos.tolist(),
            "joint_positions": self._joint_positions.tolist(),
            "distance_to_target": float(np.linalg.norm(self._target - self._position)),
        }

    def _receipt(
        self,
        action: AuthorizedAction,
        *,
        accepted: bool,
        command_applied: bool,
        done: bool,
        success: bool,
        reason: str,
    ) -> ExecutionReceipt:
        return ExecutionReceipt(
            command_id=action.command_id,
            backend_id=self.capabilities.component_id,
            execution_mode=ExecutionMode.SIMULATION,
            command_applied=command_applied,
            accepted=accepted,
            timestamp_ns=self.clock_ns(),
            measured_state=self._measured_state(),
            done=done,
            success=success,
            metadata={
                "reason": reason,
                "runtime_kind": "deterministic_kinematic_contract",
                "applied_steps": self._applied_steps,
            },
        )


def _option_vector(scenario: ScenarioSpec, field_name: str) -> np.ndarray:
    if field_name not in scenario.options:
        raise ValueError(f"scenario is missing required option {field_name!r}")
    array = np.asarray(scenario.options[field_name], dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"scenario option {field_name!r} must be a finite three-vector")
    return np.array(array, copy=True)


def _apply_axis_angle_delta(
    quaternion_xyzw: np.ndarray,
    axis_angle: np.ndarray,
) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle == 0:
        return np.array(quaternion_xyzw, copy=True)
    axis = axis_angle / angle
    half = angle / 2.0
    delta = np.concatenate((axis * np.sin(half), [np.cos(half)]))
    x1, y1, z1, w1 = quaternion_xyzw
    x2, y2, z2, w2 = delta
    composed = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    return composed / np.linalg.norm(composed)
