import unittest

import numpy as np

from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.safety import SafetyEnvelope, SafetyGateway
from robotprobe.runtime.scheduler import ActionScheduler
from robotprobe.runtime.types import (
    CanonicalObservation,
    ExecutionMode,
    ProposedActionChunk,
    RuntimePhaseError,
    ScheduledAction,
)


class ActionSchedulerTests(unittest.TestCase):
    def test_schedules_a_bounded_prefix_with_stable_lineage(self):
        scheduler = ActionScheduler(n_action_steps=2)
        chunk = _chunk(
            "chunk-7",
            [
                [0.01, 0, 0, 0, 0, 0, 1],
                [0.02, 0, 0, 0, 0, 0, 1],
                [0.03, 0, 0, 0, 0, 0, 1],
            ],
        )

        count = scheduler.load(chunk)
        first = scheduler.next_action()
        second = scheduler.next_action()

        self.assertEqual(count, 2)
        self.assertEqual(first.command_id, "chunk-7:action:000")
        self.assertEqual(second.command_id, "chunk-7:action:001")
        self.assertEqual([first.action_index, second.action_index], [0, 1])
        self.assertEqual(first.policy_call_id, chunk.policy_call_id)
        self.assertEqual(first.source_observation_id, chunk.source_observation_id)
        self.assertEqual(first.source_observation_captured_at_ns, 1_000)
        self.assertFalse(scheduler.has_pending)

    def test_refreshes_after_exhaustion_and_rejects_overwrite(self):
        scheduler = ActionScheduler(n_action_steps=1)
        scheduler.load(_chunk("chunk-1", [[0, 0, 0, 0, 0, 0, 0]]))

        with self.assertRaisesRegex(RuntimeError, "pending"):
            scheduler.load(_chunk("chunk-2", [[0, 0, 0, 0, 0, 0, 0]]))

        scheduler.next_action()
        scheduler.load(_chunk("chunk-2", [[0.1, 0, 0, 0, 0, 0, 0]]))

        self.assertEqual(scheduler.next_action().chunk_id, "chunk-2")

    def test_reset_drops_queued_actions(self):
        scheduler = ActionScheduler(n_action_steps=2)
        scheduler.load(
            _chunk(
                "chunk-1",
                [[0, 0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0, 0]],
            )
        )

        scheduler.reset()

        self.assertFalse(scheduler.has_pending)
        with self.assertRaisesRegex(RuntimeError, "empty"):
            scheduler.next_action()

    def test_requires_a_positive_execution_horizon(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            ActionScheduler(n_action_steps=0)


class SafetyGatewayTests(unittest.TestCase):
    def test_rejects_non_finite_observation_age_limits_and_invalid_clock_values(self):
        for value in (np.nan, np.inf):
            with self.subTest(max_observation_age_ms=value), self.assertRaisesRegex(
                ValueError,
                "finite",
            ):
                SafetyEnvelope(
                    command_space="franka.cartesian_delta_gripper.v0",
                    max_abs=(0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 1.0),
                    max_observation_age_ms=value,
                )

        with self.assertRaisesRegex(ValueError, "now_ns"):
            _gateway().authorize(
                _scheduled([0.01, 0, 0, 0, 0, 0, 0]),
                _observation(),
                now_ns=np.nan,
            )

    def test_authorizes_finite_simulation_action_within_envelope(self):
        gateway = _gateway()
        scheduled = _scheduled([0.01, 0, 0, 0, 0, 0, 1.0])

        authorized = gateway.authorize(scheduled, _observation(), now_ns=1_500)

        self.assertTrue(authorized.decision.accepted)
        self.assertEqual(authorized.decision.disposition, "authorized")
        self.assertEqual(authorized.decision.reason, "within_envelope")

    def test_rejects_wrong_command_space_dimension_and_bounds_without_clipping(self):
        gateway = _gateway()
        cases = (
            (
                _scheduled([0.01] * 7, command_space="franka.joint_position.v0"),
                "command_space_mismatch",
            ),
            (_scheduled([0.01] * 6), "action_dimension_mismatch"),
            (_scheduled([0.2, 0, 0, 0, 0, 0, 1.0]), "action_limit_exceeded"),
        )

        for scheduled, reason in cases:
            with self.subTest(reason=reason):
                authorized = gateway.authorize(scheduled, _observation(), now_ns=1_500)
                self.assertFalse(authorized.decision.accepted)
                self.assertEqual(authorized.decision.reason, reason)
                self.assertEqual(authorized.value.tolist(), scheduled.value.tolist())

    def test_non_finite_action_is_rejected_at_the_typed_boundary(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            _scheduled([0, 0, 0, 0, 0, 0, np.nan])

    def test_rejects_stale_future_and_out_of_workspace_actions(self):
        stale = _gateway(max_observation_age_ms=0.0001).authorize(
            _scheduled([0.01, 0, 0, 0, 0, 0, 0]),
            _observation(),
            now_ns=2_000,
        )
        future = _gateway().authorize(
            _scheduled([0.01, 0, 0, 0, 0, 0, 0], captured_at_ns=2_000),
            _observation(captured_at_ns=2_000),
            now_ns=1_500,
        )
        workspace = _gateway().authorize(
            _scheduled([0.06, 0, 0, 0, 0, 0, 0]),
            _observation(eef_position=[0.55, 0.0, 0.3]),
            now_ns=1_500,
        )

        self.assertEqual(stale.decision.reason, "observation_stale")
        self.assertEqual(future.decision.reason, "observation_clock_skew")
        self.assertEqual(workspace.decision.reason, "workspace_violation")

    def test_simulation_rejects_hardware_backend(self):
        gateway = _gateway(
            backend=_backend_capabilities(
                is_hardware=True,
                supports_shadow=True,
            )
        )

        result = gateway.authorize(
            _scheduled([0.01, 0, 0, 0, 0, 0, 0]),
            _observation(),
            now_ns=1_500,
        )

        self.assertFalse(result.decision.accepted)
        self.assertEqual(result.decision.reason, "execution_mode_backend_mismatch")

    def test_hardware_shadow_is_authorized_only_as_non_commanding(self):
        gateway = _gateway(
            mode=ExecutionMode.HARDWARE_SHADOW,
            backend=_backend_capabilities(
                is_hardware=True,
                supports_shadow=True,
            ),
        )

        result = gateway.authorize(
            _scheduled([0.01, 0, 0, 0, 0, 0, 0]),
            _observation(),
            now_ns=1_500,
        )

        self.assertTrue(result.decision.accepted)
        self.assertEqual(result.decision.disposition, "shadow_only")

    def test_hardware_executable_mode_is_not_supported_by_initial_gateway(self):
        gateway = _gateway(
            mode=ExecutionMode.HARDWARE_EXECUTABLE,
            backend=_backend_capabilities(
                is_hardware=True,
                supports_shadow=True,
                can_apply_hardware_commands=True,
            ),
        )

        result = gateway.authorize(
            _scheduled([0.01, 0, 0, 0, 0, 0, 0]),
            _observation(),
            now_ns=1_500,
        )

        self.assertFalse(result.decision.accepted)
        self.assertEqual(result.decision.reason, "hardware_executable_not_supported")

    def test_shadow_receipt_invariant_is_checked_even_for_foreign_receipt_objects(self):
        gateway = _gateway(
            mode=ExecutionMode.HARDWARE_SHADOW,
            backend=_backend_capabilities(
                is_hardware=True,
                supports_shadow=True,
            ),
        )
        receipt = ForeignReceipt(command_applied=True, backend_id="foreign")

        with self.assertRaises(RuntimePhaseError) as raised:
            gateway.validate_receipt(receipt)

        self.assertEqual(raised.exception.phase, "execution_receipt")
        self.assertIn("foreign", str(raised.exception))


class ForeignReceipt:
    def __init__(self, *, command_applied, backend_id):
        self.command_applied = command_applied
        self.backend_id = backend_id


def _gateway(
    *,
    mode=ExecutionMode.SIMULATION,
    backend=None,
    max_observation_age_ms=1.0,
):
    return SafetyGateway(
        envelope=SafetyEnvelope(
            command_space="franka.cartesian_delta_gripper.v0",
            max_abs=(0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 1.0),
            workspace_min=(0.2, -0.5, 0.1),
            workspace_max=(0.6, 0.5, 0.8),
            max_observation_age_ms=max_observation_age_ms,
        ),
        execution_mode=mode,
        backend_capabilities=backend or _backend_capabilities(),
    )


def _backend_capabilities(**overrides):
    data = {
        "format": "robotprobe.runtime.component.v0",
        "kind": "backend",
        "component_id": "backend.test",
        "command_spaces": ["franka.cartesian_delta_gripper.v0"],
        "observation_streams": ["state"],
        "runtime_family": "test",
        "is_hardware": False,
        "supports_reset": True,
        "supports_ground_truth": True,
        "supports_shadow": False,
        "can_apply_hardware_commands": False,
        "requires_reset": False,
        "optional_dependencies": [],
    }
    data.update(overrides)
    return RuntimeComponentCapabilities.from_dict(data)


def _chunk(chunk_id, values):
    return ProposedActionChunk(
        chunk_id=chunk_id,
        policy_call_id=f"policy:{chunk_id}",
        source_observation_id="obs-1",
        source_observation_captured_at_ns=1_000,
        command_space="franka.cartesian_delta_gripper.v0",
        values=np.asarray(values, dtype=float),
    )


def _scheduled(
    value,
    *,
    command_space="franka.cartesian_delta_gripper.v0",
    captured_at_ns=1_000,
):
    return ScheduledAction(
        command_id="command-1",
        chunk_id="chunk-1",
        policy_call_id="policy-1",
        source_observation_id="obs-1",
        source_observation_captured_at_ns=captured_at_ns,
        action_index=0,
        command_space=command_space,
        value=np.asarray(value, dtype=float),
    )


def _observation(*, eef_position=(0.4, 0.0, 0.3), captured_at_ns=1_000):
    return CanonicalObservation(
        observation_id="obs-1",
        sequence=1,
        captured_at_ns=captured_at_ns,
        images={},
        eef_position=np.asarray(eef_position),
        eef_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        gripper_qpos=np.array([0.02, 0.02]),
        instruction="reach",
        source="test",
        frame_id="base",
        calibration_id="test",
    )


if __name__ == "__main__":
    unittest.main()
