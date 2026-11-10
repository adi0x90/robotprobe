from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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


_LIBERO_DEFAULT_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class LiberoOptionalDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiberoEnvironmentSettings:
    suite: str
    task_id: int
    episode_length: int | None
    observation_height: int
    observation_width: int
    control_mode: str
    obs_type: str = "pixels_agent_pos"
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    init_states: bool = True


class LiberoScenarioProvider:
    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "scenario",
            "component_id": "libero.scenario.task.v0",
            "command_spaces": [],
            "observation_streams": [],
            "runtime_family": "lerobot_libero",
            "is_hardware": False,
            "supports_reset": False,
            "supports_ground_truth": False,
            "supports_shadow": False,
            "can_apply_hardware_commands": False,
            "requires_reset": True,
            "optional_dependencies": ["libero"],
        }
    )

    def __init__(
        self,
        *,
        suite: str = "libero_spatial",
        task_id: int = 0,
        seed: int = 0,
        max_steps: int | None = None,
        instruction: str | None = None,
        metadata_loader: Callable[[str, int], str] | None = None,
    ):
        if not suite.strip():
            raise ValueError("LIBERO suite must be non-empty")
        if task_id < 0:
            raise ValueError("LIBERO task_id must be non-negative")
        resolved_steps = (
            int(max_steps)
            if max_steps is not None
            else _LIBERO_DEFAULT_MAX_STEPS.get(suite, 500)
        )
        if resolved_steps <= 0:
            raise ValueError("LIBERO max_steps must be positive")
        if instruction is not None and not instruction.strip():
            raise ValueError("LIBERO instruction must be non-empty when configured")
        self.suite = suite
        self.task_id = int(task_id)
        self.seed = int(seed)
        self.max_steps = resolved_steps
        self.instruction = instruction
        self.metadata_loader = metadata_loader or _default_task_metadata

    def build(self) -> ScenarioSpec:
        instruction = self.instruction
        if instruction is None:
            try:
                instruction = self.metadata_loader(self.suite, self.task_id)
            except ImportError as error:
                raise LiberoOptionalDependencyError(
                    "LIBERO task metadata is unavailable at scenario build; install LIBERO."
                ) from error
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("LIBERO task metadata did not provide a valid instruction")
        return ScenarioSpec(
            scenario_id=f"{self.suite}:task:{self.task_id}",
            instruction=instruction,
            seed=self.seed,
            max_steps=self.max_steps,
            options={
                "suite": self.suite,
                "task_id": self.task_id,
                "benchmark": "LIBERO",
                "control_mode": "relative",
            },
        )


class LiberoFrankaBackend:
    """Lazy, instance-owned LeRobot LIBERO backend for simulated Franka episodes."""

    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "backend",
            "component_id": "lerobot.libero_franka.v0",
            "command_spaces": [FRANKA_CARTESIAN_DELTA_COMMAND_SPACE],
            "observation_streams": [
                "agent_rgb",
                "wrist_rgb",
                "eef_pose",
                "gripper",
                "joint_state",
            ],
            "runtime_family": "lerobot_libero",
            "is_hardware": False,
            "supports_reset": True,
            "supports_ground_truth": True,
            "supports_shadow": False,
            "can_apply_hardware_commands": False,
            "requires_reset": False,
            "optional_dependencies": ["lerobot", "libero", "mujoco"],
        }
    )

    def __init__(
        self,
        *,
        suite: str = "libero_spatial",
        task_id: int = 0,
        episode_length: int | None = None,
        observation_height: int = 360,
        observation_width: int = 360,
        control_mode: str = "relative",
        init_states: bool = True,
        environment_factory: Callable[[LiberoEnvironmentSettings], Any] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ):
        if not suite.strip():
            raise ValueError("LIBERO suite must be non-empty")
        if task_id < 0:
            raise ValueError("LIBERO task_id must be non-negative")
        if episode_length is not None and episode_length <= 0:
            raise ValueError("LIBERO episode_length must be positive")
        if observation_height <= 0 or observation_width <= 0:
            raise ValueError("LIBERO observation dimensions must be positive")
        if control_mode != "relative":
            raise ValueError(
                "LIBERO Franka backend requires relative control for the declared command space"
            )
        self.settings = LiberoEnvironmentSettings(
            suite=suite,
            task_id=int(task_id),
            episode_length=episode_length,
            observation_height=observation_height,
            observation_width=observation_width,
            control_mode=control_mode,
            init_states=init_states,
        )
        self.environment_factory = environment_factory or _default_environment_factory
        self.clock_ns = clock_ns
        self._environment: Any | None = None
        self._scenario: ScenarioSpec | None = None
        self._last_observation: Mapping[str, Any] | None = None
        self._last_raw: RawObservation | None = None
        self._task_description: str | None = None
        self._step_count = 0
        self._observation_sequence = 0

    def connect(self) -> None:
        if self._environment is not None:
            return
        try:
            environment = self.environment_factory(self.settings)
        except ImportError as error:
            raise LiberoOptionalDependencyError(
                "LeRobot/LIBERO/MuJoCo dependencies are unavailable at connect(); "
                "install the optional LIBERO runtime before running this backend."
            ) from error
        if environment is None:
            raise RuntimeError("LIBERO environment factory returned no environment")
        self._environment = environment
        self._task_description = _environment_task_description(environment)

    def reset(self, scenario: ScenarioSpec) -> RawObservation:
        self._require_connected()
        self._validate_scenario(scenario)
        result = self._environment.reset(seed=scenario.seed)
        observation, _info = _normalize_reset_result(result)
        environment_instruction = (
            _environment_task_description(self._environment)
            or self._task_description
        )
        if (
            environment_instruction
            and _normalize_instruction(environment_instruction)
            != _normalize_instruction(scenario.instruction)
        ):
            raise ValueError(
                "LIBERO environment instruction does not match scenario instruction"
            )
        self._scenario = scenario
        self._last_observation = observation
        self._step_count = 0
        self._observation_sequence = 0
        self._task_description = scenario.instruction
        self._last_raw = self._make_raw(observation)
        return self._last_raw

    def observe(self) -> RawObservation:
        self._require_ready()
        assert self._last_raw is not None
        return self._last_raw

    def execute(self, action: AuthorizedAction) -> ExecutionReceipt:
        self._require_ready()
        if not isinstance(action, AuthorizedAction):
            raise TypeError("LIBERO backend requires AuthorizedAction")
        if not action.decision.accepted:
            raise ValueError("LIBERO backend refuses unauthorized actions")
        if action.scheduled.command_space != FRANKA_CARTESIAN_DELTA_COMMAND_SPACE:
            raise ValueError("LIBERO backend received an unsupported command space")
        value = np.asarray(action.value, dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("LIBERO action must be a finite seven-value vector")

        env_action = (
            value[np.newaxis, :]
            if int(getattr(self._environment, "num_envs", 0) or 0) > 0
            else value
        )
        result = self._environment.step(env_action)
        observation, reward, terminated, truncated, info = _normalize_step_result(
            result
        )
        terminal_observation = _terminal_observation(info) or observation
        self._last_observation = terminal_observation
        self._step_count += 1
        self._observation_sequence += 1
        self._last_raw = self._make_raw(terminal_observation)

        success = _extract_success(info)
        environment_terminated = _bool_any(terminated)
        environment_truncated = _bool_any(truncated)
        assert self._scenario is not None
        max_steps_exhausted = self._step_count >= self._scenario.max_steps
        done = (
            success
            or environment_terminated
            or environment_truncated
            or max_steps_exhausted
        )
        reason = (
            "task_success"
            if success
            else (
                "environment_terminated"
                if environment_terminated
                else (
                    "environment_truncated"
                    if environment_truncated
                    else (
                        "max_steps_exhausted"
                        if max_steps_exhausted
                        else "applied"
                    )
                )
            )
        )
        return ExecutionReceipt(
            command_id=action.command_id,
            backend_id=self.capabilities.component_id,
            execution_mode=ExecutionMode.SIMULATION,
            command_applied=True,
            accepted=True,
            timestamp_ns=self.clock_ns(),
            measured_state=_measured_state(self._last_raw),
            done=done,
            success=success,
            metadata={
                "reason": reason,
                "reward": _scalar(reward),
                "environment_terminated": environment_terminated,
                "environment_truncated": environment_truncated,
                "max_steps_exhausted": max_steps_exhausted,
                "suite": self.settings.suite,
                "task_id": self.settings.task_id,
                "step": self._step_count,
            },
        )

    def health(self) -> BackendHealth:
        connected = self._environment is not None
        return BackendHealth(
            backend_id=self.capabilities.component_id,
            connected=connected,
            healthy=connected,
            timestamp_ns=self.clock_ns(),
            details={
                "suite": self.settings.suite,
                "task_id": self.settings.task_id,
                "environment_owned_by_backend": True,
                "reset": self._scenario is not None,
            },
        )

    def close(self) -> None:
        environment = self._environment
        self._environment = None
        self._scenario = None
        self._last_observation = None
        self._last_raw = None
        self._task_description = None
        self._step_count = 0
        self._observation_sequence = 0
        if environment is not None:
            environment.close()

    def _validate_scenario(self, scenario: ScenarioSpec) -> None:
        suite = scenario.options.get("suite")
        task_id = scenario.options.get("task_id")
        if suite is not None and suite != self.settings.suite:
            raise ValueError(
                f"scenario suite {suite!r} does not match backend suite {self.settings.suite!r}"
            )
        if task_id is not None and int(task_id) != self.settings.task_id:
            raise ValueError("scenario task_id does not match backend task_id")
        if (
            self.settings.episode_length is not None
            and scenario.max_steps > self.settings.episode_length
        ):
            raise ValueError("scenario max_steps exceeds backend episode_length")

    def _make_raw(self, observation: Mapping[str, Any]) -> RawObservation:
        state = _extract_franka_observation(observation)
        instruction = self._task_description
        if not instruction and self._scenario is not None:
            instruction = self._scenario.instruction
        if not instruction:
            raise ValueError("LIBERO environment did not expose a task instruction")
        return RawObservation(
            observation_id=(
                f"libero:{self.settings.suite}:{self.settings.task_id}:"
                f"{self._observation_sequence:06d}"
            ),
            sequence=self._observation_sequence,
            captured_at_ns=self.clock_ns(),
            payload={
                **state,
                "instruction": instruction,
                "frame_id": "libero_franka_base",
                "calibration_id": (
                    f"libero:{self.settings.observation_width}x"
                    f"{self.settings.observation_height}:v0"
                ),
                "quaternion_convention": "xyzw",
            },
            source=self.capabilities.component_id,
            metadata={
                "suite": self.settings.suite,
                "task_id": self.settings.task_id,
                "control_mode": self.settings.control_mode,
                "ground_truth": True,
                "joint_efforts_available": False,
                "environment_owned_by_backend": True,
            },
        )

    def _require_connected(self) -> None:
        if self._environment is None:
            raise RuntimeError("LIBERO backend must connect before reset")

    def _require_ready(self) -> None:
        if (
            self._environment is None
            or self._scenario is None
            or self._last_observation is None
            or self._last_raw is None
        ):
            raise RuntimeError("LIBERO backend must connect and reset before use")


def _default_environment_factory(settings: LiberoEnvironmentSettings) -> Any:
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env

    config = LiberoEnvConfig(
        task=settings.suite,
        task_ids=[settings.task_id],
        episode_length=settings.episode_length,
        obs_type=settings.obs_type,
        camera_name=settings.camera_name,
        init_states=settings.init_states,
        observation_height=settings.observation_height,
        observation_width=settings.observation_width,
        control_mode=settings.control_mode,
        camera_name_mapping={
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "image2",
        },
    )
    environments = make_env(config, n_envs=1, use_async_envs=False)
    return environments[settings.suite][settings.task_id]


def _default_task_metadata(suite_name: str, task_id: int) -> str:
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if suite_name not in suites:
        raise ValueError(f"unknown LIBERO suite: {suite_name}")
    suite = suites[suite_name]()
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(f"LIBERO task_id out of range: {task_id}")
    return str(suite.get_task(task_id).language)


def _environment_task_description(environment: Any) -> str | None:
    call = getattr(environment, "call", None)
    if callable(call):
        try:
            values = call("task_description")
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                if values and isinstance(values[0], str):
                    return values[0]
            if isinstance(values, str):
                return values
        except (AttributeError, KeyError, NotImplementedError):
            pass
    value = getattr(environment, "task_description", None)
    return str(value) if isinstance(value, str) and value else None


def _normalize_instruction(value: str) -> str:
    return " ".join(value.split())


def _normalize_reset_result(result: Any) -> tuple[Mapping[str, Any], Any]:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("LIBERO reset must return observation or (observation, info)")
        observation, info = result
    else:
        observation, info = result, {}
    if not isinstance(observation, Mapping):
        raise TypeError("LIBERO reset observation must be a mapping")
    return observation, info


def _normalize_step_result(
    result: Any,
) -> tuple[Mapping[str, Any], Any, Any, Any, Any]:
    if not isinstance(result, tuple):
        raise TypeError("LIBERO step result must be a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:
        observation, reward, terminated, info = result
        truncated = False
    else:
        raise ValueError("LIBERO step must return four or five values")
    if not isinstance(observation, Mapping):
        raise TypeError("LIBERO step observation must be a mapping")
    return observation, reward, terminated, truncated, info


def _extract_franka_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    pixels = observation.get("pixels")
    robot_state = observation.get("robot_state")
    if not isinstance(pixels, Mapping) or not isinstance(robot_state, Mapping):
        raise ValueError("LIBERO observation requires pixels and robot_state mappings")
    agent_image = pixels.get("image", pixels.get("agent"))
    wrist_image = pixels.get("image2", pixels.get("wrist"))
    if agent_image is None or wrist_image is None:
        raise ValueError("LIBERO observation requires main and wrist images")
    eef = robot_state.get("eef")
    gripper = robot_state.get("gripper")
    joints = robot_state.get("joints", robot_state.get("joint"))
    if not all(isinstance(value, Mapping) for value in (eef, gripper, joints)):
        raise ValueError("LIBERO robot_state is missing eef, gripper, or joints")
    return {
        "images": {
            "agent": _unbatch(agent_image, expected_rank=3, field_name="agent image"),
            "wrist": _unbatch(wrist_image, expected_rank=3, field_name="wrist image"),
        },
        "eef_position": _unbatch(eef.get("pos"), expected_rank=1, field_name="eef position"),
        "eef_quaternion": _unbatch(
            eef.get("quat"),
            expected_rank=1,
            field_name="eef quaternion",
        ),
        "gripper_qpos": _unbatch(
            gripper.get("qpos"),
            expected_rank=1,
            field_name="gripper qpos",
        ),
        "joint_positions": _unbatch(
            joints.get("pos"),
            expected_rank=1,
            field_name="joint positions",
        ),
        "joint_velocities": _unbatch(
            joints.get("vel"),
            expected_rank=1,
            field_name="joint velocities",
        ),
        "joint_efforts": np.zeros(7, dtype=np.float64),
    }


def _unbatch(value: Any, *, expected_rank: int, field_name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"LIBERO observation is missing {field_name}")
    array = np.asarray(value)
    if array.ndim == expected_rank + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != expected_rank:
        raise ValueError(
            f"LIBERO {field_name} must have rank {expected_rank} after unbatching"
        )
    return np.array(array, copy=True)


def _extract_success(info: Any) -> bool:
    if isinstance(info, Mapping):
        for key in ("is_success", "success", "task_success"):
            if key in info and _bool_any(info[key]):
                return True
        for key in ("final_info", "final_infos"):
            if key in info and _extract_success(info[key]):
                return True
        return False
    if isinstance(info, Sequence) and not isinstance(info, (str, bytes)):
        return any(_extract_success(item) for item in info)
    return False


def _terminal_observation(info: Any) -> Mapping[str, Any] | None:
    if not isinstance(info, Mapping):
        return None
    for key in ("final_observation", "terminal_observation"):
        value = info.get(key)
        if isinstance(value, Mapping):
            return value
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and value
            and isinstance(value[0], Mapping)
        ):
            return value[0]
    return None


def _bool_any(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(np.asarray(value, dtype=bool).any())
    except (TypeError, ValueError):
        return bool(value)


def _scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    return float(array.reshape(-1)[0]) if array.size else 0.0


def _measured_state(raw: RawObservation) -> dict[str, Any]:
    payload = raw.payload
    return {
        "observation_id": raw.observation_id,
        "eef_position": np.asarray(payload["eef_position"]).tolist(),
        "eef_quaternion": np.asarray(payload["eef_quaternion"]).tolist(),
        "gripper_qpos": np.asarray(payload["gripper_qpos"]).tolist(),
        "joint_positions": np.asarray(payload["joint_positions"]).tolist(),
        "joint_velocities": np.asarray(payload["joint_velocities"]).tolist(),
    }
