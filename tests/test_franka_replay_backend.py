import unittest

from robotprobe.integrations.franka.replay_backend import FrankaRecordedReplayBackend
from robotprobe.integrations.franka.scenarios import DeterministicReachScenario
from robotprobe.runtime.conformance import assert_backend_conforms
from robotprobe.runtime.types import (
    AuthorizedAction,
    ExecutionMode,
    SafetyDecision,
    ScheduledAction,
)


class FrankaRecordedReplayBackendTests(unittest.TestCase):
    def test_authorized_action_is_evaluated_but_never_applied(self):
        backend = make_backend()
        scenario = make_scenario()
        backend.connect()
        observation = backend.reset(scenario)

        receipt = backend.execute(
            authorized_action(source_observation_id=observation.observation_id)
        )

        self.assertTrue(receipt.accepted)
        self.assertFalse(receipt.command_applied)
        self.assertEqual(receipt.execution_mode, ExecutionMode.SIMULATION)
        self.assertEqual(
            receipt.metadata["fixture_provenance"],
            "synthetic_recorded_format",
        )
        self.assertEqual(
            receipt.metadata["reason"],
            "recorded_snapshot_evaluated_not_applied",
        )

    def test_lifecycle_replays_snapshots_and_reuses_the_last_snapshot(self):
        backend = make_backend(
            snapshots=[snapshot(position_x=0.4), snapshot(position_x=0.41)]
        )
        with self.assertRaisesRegex(RuntimeError, "connect and reset"):
            backend.observe()

        backend.connect()
        self.assertTrue(backend.health().connected)
        first = backend.reset(make_scenario())
        second = backend.observe()
        repeated = backend.observe()
        backend.close()
        backend.close()

        self.assertEqual(first.sequence, 0)
        self.assertEqual(second.sequence, 1)
        self.assertEqual(repeated.sequence, 2)
        self.assertEqual(second.payload["eef_position"].tolist(), [0.41, 0.0, 0.3])
        self.assertEqual(repeated.payload["eef_position"].tolist(), [0.41, 0.0, 0.3])
        self.assertFalse(backend.health().connected)

    def test_public_backend_conformance_helper_accepts_replay_backend(self):
        backend = make_backend(
            snapshots=[snapshot(position_x=0.4), snapshot(position_x=0.41)]
        )

        report = assert_backend_conforms(
            backend=backend,
            scenario=make_scenario(),
            action=authorized_action(
                source_observation_id="franka-replay:000001"
            ),
        )

        self.assertTrue(report.passed, report.issues)
        self.assertFalse(backend.health().connected)

    def test_execute_rejects_unauthorized_wrong_space_and_stale_source(self):
        backend = make_backend()
        backend.connect()
        observation = backend.reset(make_scenario())

        with self.assertRaisesRegex(ValueError, "authorized action"):
            backend.execute(
                authorized_action(
                    source_observation_id=observation.observation_id,
                    accepted=False,
                )
            )
        with self.assertRaisesRegex(ValueError, "command space"):
            backend.execute(
                authorized_action(
                    source_observation_id=observation.observation_id,
                    command_space="other.command.v0",
                )
            )
        with self.assertRaisesRegex(ValueError, "source observation"):
            backend.execute(
                authorized_action(source_observation_id="franka-replay:stale")
            )

    def test_constructor_rejects_untruthful_or_invalid_fixtures(self):
        with self.assertRaisesRegex(ValueError, "at least one snapshot"):
            make_backend(snapshots=[])
        with self.assertRaisesRegex(ValueError, "synthetic_recorded_format"):
            make_backend(fixture_provenance="physical_capture")
        invalid_shape = snapshot()
        invalid_shape["image_specs"]["agent"]["shape"] = [2, 2]
        with self.assertRaisesRegex(ValueError, "three-dimensional"):
            make_backend(snapshots=[invalid_shape])
        invalid_vector = snapshot()
        invalid_vector["joint_positions"] = [0.0] * 6
        with self.assertRaisesRegex(ValueError, "joint_positions"):
            make_backend(snapshots=[invalid_vector])

    def test_backend_has_no_command_publication_surface(self):
        backend = make_backend()

        for method_name in ("publish", "send", "command"):
            with self.subTest(method_name=method_name):
                self.assertFalse(hasattr(backend, method_name))


def make_backend(**overrides) -> FrankaRecordedReplayBackend:
    options = {
        "snapshots": [snapshot()],
        "fixture_provenance": "synthetic_recorded_format",
        "clock_ns": lambda: 1_000,
    }
    options.update(overrides)
    return FrankaRecordedReplayBackend(**options)


def make_scenario():
    return DeterministicReachScenario(
        initial_position=(0.4, 0.0, 0.3),
        target_position=(0.5, 0.0, 0.3),
        max_steps=3,
    ).build()


def snapshot(*, position_x: float = 0.4) -> dict:
    return {
        "image_specs": {
            "agent": {"shape": [2, 2, 3], "fill": 0},
            "wrist": {"shape": [2, 2, 3], "fill": 16},
        },
        "eef_position": [position_x, 0.0, 0.3],
        "eef_quaternion": [0.0, 0.0, 0.0, 1.0],
        "gripper_qpos": [0.02, 0.02],
        "joint_positions": [0.0] * 7,
        "joint_velocities": [0.0] * 7,
        "joint_efforts": [0.0] * 7,
    }


def authorized_action(
    *,
    source_observation_id: str,
    command_space: str = "franka.cartesian_delta_gripper.v0",
    accepted: bool = True,
) -> AuthorizedAction:
    return AuthorizedAction(
        scheduled=ScheduledAction(
            command_id="command:000000:000",
            chunk_id="chunk:000000",
            policy_call_id="policy:000000",
            source_observation_id=source_observation_id,
            source_observation_captured_at_ns=1_000,
            action_index=0,
            command_space=command_space,
            value=[0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
        decision=SafetyDecision(
            accepted=accepted,
            disposition="authorized" if accepted else "rejected",
            reason="within_envelope" if accepted else "test_rejection",
            checked_at_ns=1_000,
        ),
    )


if __name__ == "__main__":
    unittest.main()
