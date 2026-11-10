import unittest

import numpy as np

from robotprobe.integrations.franka.deterministic import DeterministicFrankaBackend
from robotprobe.integrations.franka.embodiment import (
    FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
    FrankaEmbodimentAdapter,
)
from robotprobe.integrations.franka.scenarios import DeterministicReachScenario
from robotprobe.runtime.conformance import assert_backend_conforms
from robotprobe.runtime.types import (
    AuthorizedAction,
    ExecutionMode,
    ProposedActionChunk,
    RawObservation,
    SafetyDecision,
    ScheduledAction,
)


class FrankaEmbodimentTests(unittest.TestCase):
    def test_canonicalizes_franka_state_images_and_xyzw_quaternion(self):
        front = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        wrist = np.full((2, 4, 3), 7, dtype=np.uint8)
        raw = RawObservation(
            observation_id="franka:000004",
            sequence=4,
            captured_at_ns=5_000,
            payload={
                "images": {"front_camera": front, "hand_camera": wrist},
                "eef_position": [0.4, -0.1, 0.3],
                "eef_quaternion": [0.1, 0.2, 0.3, 0.9],
                "gripper_qpos": [0.02, 0.021],
                "joint_positions": np.arange(7, dtype=float),
                "joint_velocities": np.arange(7, dtype=float) / 10,
                "joint_efforts": np.arange(7, dtype=float) / 100,
                "instruction": "Reach the target.",
                "frame_id": "franka_link0",
                "calibration_id": "calibration:test",
                "quaternion_convention": "xyzw",
            },
            source="franka_test",
            metadata={"transport": "test"},
        )
        adapter = FrankaEmbodimentAdapter(
            image_role_map={"agent": "front_camera", "wrist": "hand_camera"}
        )

        observation = adapter.canonicalize_observation(raw)

        self.assertEqual(observation.observation_id, raw.observation_id)
        self.assertEqual(observation.eef_quaternion.tolist(), [0.1, 0.2, 0.3, 0.9])
        self.assertEqual(observation.joint_positions.tolist(), list(range(7)))
        self.assertEqual(observation.joint_velocities.tolist(), [value / 10 for value in range(7)])
        self.assertEqual(observation.joint_efforts.tolist(), [value / 100 for value in range(7)])
        np.testing.assert_array_equal(observation.images["agent"], front)
        np.testing.assert_array_equal(observation.images["wrist"], wrist)
        self.assertEqual(observation.metadata["quaternion_convention"], "xyzw")
        self.assertEqual(observation.metadata["raw_source_metadata"]["transport"], "test")

    def test_rejects_non_xyzw_observations_and_missing_image_roles(self):
        adapter = FrankaEmbodimentAdapter(image_role_map={"agent": "front"})
        payload = canonical_raw_payload()
        payload["quaternion_convention"] = "wxyz"

        with self.assertRaisesRegex(ValueError, "xyzw"):
            adapter.canonicalize_observation(raw_observation(payload))

        payload = canonical_raw_payload()
        payload["images"] = {"other": np.zeros((2, 2, 3), dtype=np.uint8)}
        with self.assertRaisesRegex(ValueError, "front"):
            adapter.canonicalize_observation(raw_observation(payload))

    def test_decodes_only_declared_seven_dimensional_action_space(self):
        adapter = FrankaEmbodimentAdapter()
        chunk = proposed_chunk(np.zeros((2, 7)))

        decoded = adapter.decode_action_chunk(chunk)

        self.assertEqual(decoded.command_space, FRANKA_CARTESIAN_DELTA_COMMAND_SPACE)
        self.assertEqual(decoded.values.shape, (2, 7))
        self.assertEqual(decoded.metadata["embodiment_adapter"], "franka.canonical.v0")

        with self.assertRaisesRegex(ValueError, "command space"):
            adapter.decode_action_chunk(
                proposed_chunk(np.zeros((1, 7)), command_space="franka.joint_position.v0")
            )
        with self.assertRaisesRegex(ValueError, "seven"):
            adapter.decode_action_chunk(proposed_chunk(np.zeros((1, 6))))


class DeterministicFrankaBackendTests(unittest.TestCase):
    def test_declares_explicit_simulation_capabilities(self):
        capabilities = DeterministicFrankaBackend().capabilities

        self.assertEqual(capabilities.component_id, "franka.deterministic_kinematic.v0")
        self.assertFalse(capabilities.is_hardware)
        self.assertTrue(capabilities.supports_reset)
        self.assertTrue(capabilities.supports_ground_truth)
        self.assertFalse(capabilities.supports_shadow)
        self.assertEqual(
            capabilities.command_spaces,
            (FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,),
        )

    def test_lifecycle_misuse_fails_explicitly_and_close_is_observable(self):
        backend = DeterministicFrankaBackend(clock_ns=lambda: 1_000)
        scenario = DeterministicReachScenario().build()

        with self.assertRaisesRegex(RuntimeError, "connect"):
            backend.reset(scenario)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            backend.observe()

        backend.connect()
        backend.reset(scenario)
        backend.close()

        self.assertFalse(backend.health().connected)
        with self.assertRaisesRegex(RuntimeError, "connect"):
            backend.observe()

    def test_seeded_reset_and_synthetic_images_are_deterministic(self):
        scenario = DeterministicReachScenario(
            seed=19,
            initial_position_jitter=(0.01, 0.02, 0.01),
        ).build()
        first = DeterministicFrankaBackend(clock_ns=lambda: 1_000)
        second = DeterministicFrankaBackend(clock_ns=lambda: 1_000)
        first.connect()
        second.connect()

        first_observation = first.reset(scenario)
        second_observation = second.reset(scenario)

        np.testing.assert_array_equal(
            first_observation.payload["eef_position"],
            second_observation.payload["eef_position"],
        )
        np.testing.assert_array_equal(
            first_observation.payload["images"]["agent"],
            second_observation.payload["images"]["agent"],
        )
        self.assertEqual(first_observation.observation_id, "franka-sim:000000")

    def test_cartesian_delta_updates_state_and_reaches_scenario_target(self):
        backend = DeterministicFrankaBackend(clock_ns=lambda: 2_000)
        scenario = DeterministicReachScenario(
            initial_position=(0.4, 0.0, 0.3),
            target_position=(0.41, 0.0, 0.3),
            tolerance=1e-6,
        ).build()
        backend.connect()
        backend.reset(scenario)

        receipt = backend.execute(authorized_action([0.01, 0, 0, 0, 0, 0, -1.0]))
        observed = backend.observe()

        self.assertTrue(receipt.accepted)
        self.assertTrue(receipt.command_applied)
        self.assertTrue(receipt.done)
        self.assertTrue(receipt.success)
        self.assertEqual(receipt.execution_mode, ExecutionMode.SIMULATION)
        self.assertAlmostEqual(receipt.measured_state["eef_position"][0], 0.41)
        self.assertEqual(receipt.measured_state["gripper_qpos"], [0.0, 0.0])
        self.assertEqual(observed.sequence, 1)
        self.assertEqual(observed.payload["eef_position"].tolist(), [0.41000000000000003, 0.0, 0.3])

    def test_backend_workspace_check_is_fail_closed_without_state_transition(self):
        backend = DeterministicFrankaBackend(clock_ns=lambda: 2_000)
        scenario = DeterministicReachScenario(
            initial_position=(0.4, 0.0, 0.3),
            target_position=(0.5, 0.0, 0.3),
            workspace_min=(0.2, -0.5, 0.1),
            workspace_max=(0.6, 0.5, 0.8),
        ).build()
        backend.connect()
        before = backend.reset(scenario).payload["eef_position"].copy()

        receipt = backend.execute(authorized_action([0.3, 0, 0, 0, 0, 0, 1.0]))
        after = backend.observe().payload["eef_position"]

        self.assertFalse(receipt.accepted)
        self.assertFalse(receipt.command_applied)
        self.assertTrue(receipt.done)
        self.assertFalse(receipt.success)
        self.assertEqual(receipt.metadata["reason"], "backend_workspace_violation")
        np.testing.assert_array_equal(after, before)

    def test_public_backend_conformance_helper_accepts_deterministic_backend(self):
        backend = DeterministicFrankaBackend(clock_ns=lambda: 3_000)
        scenario = DeterministicReachScenario(
            target_position=(0.55, 0.0, 0.3),
        ).build()

        report = assert_backend_conforms(
            backend=backend,
            scenario=scenario,
            action=authorized_action([0.01, 0, 0, 0, 0, 0, 1.0]),
        )

        self.assertTrue(report.passed, report.issues)
        self.assertIn("execution_receipt", report.checks)
        self.assertFalse(backend.health().connected)


def canonical_raw_payload():
    return {
        "images": {"front": np.zeros((2, 2, 3), dtype=np.uint8)},
        "eef_position": [0.4, 0.0, 0.3],
        "eef_quaternion": [0.0, 0.0, 0.0, 1.0],
        "gripper_qpos": [0.02, 0.02],
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_efforts": np.zeros(7),
        "instruction": "Reach.",
        "frame_id": "franka_link0",
        "calibration_id": "test",
        "quaternion_convention": "xyzw",
    }


def raw_observation(payload):
    return RawObservation(
        observation_id="raw:0",
        sequence=0,
        captured_at_ns=1,
        payload=payload,
        source="test",
    )


def proposed_chunk(values, *, command_space=FRANKA_CARTESIAN_DELTA_COMMAND_SPACE):
    return ProposedActionChunk(
        chunk_id="chunk:000000",
        policy_call_id="policy:000000",
        source_observation_id="observation:000000",
        source_observation_captured_at_ns=1_000,
        command_space=command_space,
        values=values,
    )


def authorized_action(value):
    scheduled = ScheduledAction(
        command_id="chunk:000000:action:000",
        chunk_id="chunk:000000",
        policy_call_id="policy:000000",
        source_observation_id="franka-sim:000000",
        source_observation_captured_at_ns=1_000,
        action_index=0,
        command_space=FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
        value=value,
    )
    return AuthorizedAction(
        scheduled=scheduled,
        decision=SafetyDecision(
            accepted=True,
            disposition="authorized",
            reason="within_envelope",
            checked_at_ns=1_000,
        ),
    )


if __name__ == "__main__":
    unittest.main()
