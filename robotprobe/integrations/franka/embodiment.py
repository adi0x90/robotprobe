from __future__ import annotations

from typing import Any, Mapping

from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.types import (
    CanonicalObservation,
    ProposedActionChunk,
    RawObservation,
)


FRANKA_CARTESIAN_DELTA_COMMAND_SPACE = "franka.cartesian_delta_gripper.v0"


class FrankaEmbodimentAdapter:
    """Translate backend-specific Franka state into RobotProbe's canonical boundary."""

    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "embodiment",
            "component_id": "franka.canonical.v0",
            "command_spaces": [FRANKA_CARTESIAN_DELTA_COMMAND_SPACE],
            "observation_streams": ["rgb", "eef_pose", "gripper", "joint_state"],
            "runtime_family": "franka",
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
        image_role_map: Mapping[str, str] | None = None,
    ):
        self.image_role_map = dict(image_role_map or {"agent": "agent", "wrist": "wrist"})
        if not self.image_role_map:
            raise ValueError("image_role_map must contain at least one canonical image role")
        if not all(role and source for role, source in self.image_role_map.items()):
            raise ValueError("image role and source names must be non-empty")

    def canonicalize_observation(self, raw: RawObservation) -> CanonicalObservation:
        if not isinstance(raw, RawObservation):
            raise TypeError("Franka embodiment requires RawObservation")
        payload = raw.payload
        convention = str(payload.get("quaternion_convention", ""))
        if convention != "xyzw":
            raise ValueError("Franka observations must declare quaternion_convention='xyzw'")
        raw_images = payload.get("images")
        if not isinstance(raw_images, Mapping):
            raise ValueError("Franka observation images must be a mapping")
        images: dict[str, Any] = {}
        for canonical_role, source_name in self.image_role_map.items():
            if source_name not in raw_images:
                raise ValueError(
                    f"Franka observation is missing configured image source {source_name!r}"
                )
            images[canonical_role] = raw_images[source_name]

        required = (
            "eef_position",
            "eef_quaternion",
            "gripper_qpos",
            "joint_positions",
            "joint_velocities",
            "joint_efforts",
            "instruction",
            "frame_id",
            "calibration_id",
        )
        missing = [field_name for field_name in required if field_name not in payload]
        if missing:
            raise ValueError(f"Franka observation is missing fields: {missing}")

        return CanonicalObservation(
            observation_id=raw.observation_id,
            sequence=raw.sequence,
            captured_at_ns=raw.captured_at_ns,
            images=images,
            eef_position=payload["eef_position"],
            eef_quaternion=payload["eef_quaternion"],
            gripper_qpos=payload["gripper_qpos"],
            joint_positions=payload["joint_positions"],
            joint_velocities=payload["joint_velocities"],
            joint_efforts=payload["joint_efforts"],
            instruction=str(payload["instruction"]),
            source=raw.source,
            frame_id=str(payload["frame_id"]),
            calibration_id=str(payload["calibration_id"]),
            metadata={
                "quaternion_convention": convention,
                "raw_source_metadata": dict(raw.metadata),
            },
        )

    def decode_action_chunk(self, proposed: ProposedActionChunk) -> ProposedActionChunk:
        if not isinstance(proposed, ProposedActionChunk):
            raise TypeError("Franka embodiment requires ProposedActionChunk")
        if proposed.command_space != FRANKA_CARTESIAN_DELTA_COMMAND_SPACE:
            raise ValueError(
                f"undeclared Franka command space: {proposed.command_space!r}"
            )
        if proposed.values.shape[1] != 7:
            raise ValueError(
                "Franka Cartesian delta actions must have seven values: "
                "xyz delta, axis-angle delta, and gripper command"
            )
        metadata = dict(proposed.metadata)
        metadata.update(
            {
                "embodiment_adapter": self.capabilities.component_id,
                "rotation_delta_convention": "axis_angle_xyz",
                "gripper_command_range": [-1.0, 1.0],
            }
        )
        return ProposedActionChunk(
            chunk_id=proposed.chunk_id,
            policy_call_id=proposed.policy_call_id,
            source_observation_id=proposed.source_observation_id,
            source_observation_captured_at_ns=proposed.source_observation_captured_at_ns,
            command_space=proposed.command_space,
            values=proposed.values,
            metadata=metadata,
        )
