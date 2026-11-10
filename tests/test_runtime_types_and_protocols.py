import unittest

import numpy as np

from robotprobe.runtime.protocols import (
    ActivationCapture,
    EmbodimentAdapter,
    PolicyAdapter,
    RuntimeBackend,
    ScenarioProvider,
)
from robotprobe.runtime.types import (
    AuthorizedAction,
    BackendHealth,
    CanonicalObservation,
    EpisodeResult,
    ExecutionMode,
    ExecutionReceipt,
    InferenceContext,
    ProposedActionChunk,
    RawObservation,
    RuntimePhaseError,
    SafetyDecision,
    ScheduledAction,
    ScenarioSpec,
)


class RuntimeTypesTests(unittest.TestCase):
    def test_scenario_requires_complete_identity_and_strict_integer_bounds(self):
        base = {
            "scenario_id": "scenario-1",
            "instruction": "Pick up the bowl.",
            "seed": 7,
            "max_steps": 20,
        }
        invalid_values = (
            ("scenario_id", " "),
            ("instruction", " "),
            ("seed", np.nan),
            ("seed", -1),
            ("seed", True),
            ("max_steps", np.nan),
            ("max_steps", 1.5),
            ("max_steps", True),
            ("max_steps", 0),
        )
        for field_name, value in invalid_values:
            invalid = {**base, field_name: value}
            with self.subTest(field_name=field_name, value=value), self.assertRaises(
                ValueError
            ):
                ScenarioSpec(**invalid)

    def test_scenario_and_observation_are_json_safe_and_copy_array_inputs(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        eef_position = np.array([0.4, 0.0, 0.3], dtype=np.float32)
        scenario = ScenarioSpec(
            scenario_id="reach",
            instruction="Move to the target.",
            seed=7,
            max_steps=20,
            options={"target": [0.5, 0.0, 0.3]},
        )
        observation = CanonicalObservation(
            observation_id="obs-000001",
            sequence=1,
            captured_at_ns=1_000,
            images={"agent": image},
            eef_position=eef_position,
            eef_quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            gripper_qpos=np.array([0.02, 0.02]),
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            joint_efforts=np.zeros(7),
            instruction=scenario.instruction,
            source="deterministic_franka",
            frame_id="franka_base",
            calibration_id="sim-v1",
        )

        image[:] = 0
        eef_position[:] = 0
        fields = observation.to_trace_fields()

        self.assertEqual(scenario.to_dict()["options"]["target"], [0.5, 0.0, 0.3])
        self.assertEqual(fields["eef_position"], [0.4000000059604645, 0.0, 0.30000001192092896])
        self.assertEqual(fields["images"]["agent"]["shape"], [2, 2, 3])
        self.assertEqual(int(observation.images["agent"][0, 0, 1]), 1)
        self.assertFalse(observation.eef_position.flags.writeable)
        with self.assertRaises(ValueError):
            observation.eef_position[0] = 1.0

    def test_canonical_observation_rejects_malformed_franka_vectors(self):
        kwargs = _observation_kwargs()
        for field, value, message in (
            ("eef_position", np.zeros(2), "eef_position"),
            ("eef_quaternion", np.zeros(3), "eef_quaternion"),
            ("gripper_qpos", np.zeros(1), "gripper_qpos"),
            ("joint_positions", np.zeros(6), "joint_positions"),
            ("joint_velocities", np.zeros(8), "joint_velocities"),
            ("joint_efforts", np.zeros(1), "joint_efforts"),
        ):
            invalid = dict(kwargs)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                CanonicalObservation(**invalid)

    def test_observation_and_action_timestamps_are_non_negative_integers(self):
        raw_kwargs = {
            "observation_id": "raw-1",
            "sequence": 0,
            "payload": {},
            "source": "backend",
        }
        for captured_at_ns in (np.nan, np.inf, 1.5, -1, True):
            with self.subTest(
                raw_timestamp=captured_at_ns
            ), self.assertRaisesRegex(ValueError, "captured_at_ns"):
                RawObservation(
                    captured_at_ns=captured_at_ns,
                    **raw_kwargs,
                )

        chunk_kwargs = {
            "chunk_id": "chunk-1",
            "policy_call_id": "policy-1",
            "source_observation_id": "obs-1",
            "command_space": "franka.cartesian_delta_gripper.v0",
            "values": np.zeros((1, 7)),
        }
        for captured_at_ns in (np.nan, np.inf, 1.5, -1, True):
            with self.subTest(
                chunk_timestamp=captured_at_ns
            ), self.assertRaisesRegex(
                ValueError,
                "source_observation_captured_at_ns",
            ):
                ProposedActionChunk(
                    source_observation_captured_at_ns=captured_at_ns,
                    **chunk_kwargs,
                )

    def test_canonical_observation_requires_semantic_identity_fields(self):
        for field_name in (
            "observation_id",
            "instruction",
            "source",
            "frame_id",
            "calibration_id",
        ):
            invalid = _observation_kwargs()
            invalid[field_name] = " "
            with self.subTest(field_name=field_name), self.assertRaisesRegex(
                ValueError,
                field_name,
            ):
                CanonicalObservation(**invalid)

    def test_proposed_chunk_rejects_wrong_rank_empty_and_non_finite_values(self):
        kwargs = {
            "chunk_id": "chunk-1",
            "policy_call_id": "policy-1",
            "source_observation_id": "obs-1",
            "source_observation_captured_at_ns": 100,
            "command_space": "franka.cartesian_delta_gripper.v0",
        }
        with self.assertRaisesRegex(ValueError, "rank two"):
            ProposedActionChunk(values=np.zeros(7), **kwargs)
        with self.assertRaisesRegex(ValueError, "at least one action"):
            ProposedActionChunk(values=np.empty((0, 7)), **kwargs)
        with self.assertRaisesRegex(ValueError, "finite"):
            ProposedActionChunk(values=np.array([[0, 0, 0, 0, 0, 0, np.nan]]), **kwargs)

    def test_action_lineage_and_receipts_are_serializable(self):
        chunk = ProposedActionChunk(
            chunk_id="chunk-1",
            policy_call_id="policy-1",
            source_observation_id="obs-1",
            source_observation_captured_at_ns=100,
            command_space="franka.cartesian_delta_gripper.v0",
            values=np.array([[0.01, 0, 0, 0, 0, 0, 1.0]], dtype=np.float32),
        )
        scheduled = ScheduledAction(
            command_id="command-1",
            chunk_id=chunk.chunk_id,
            policy_call_id=chunk.policy_call_id,
            source_observation_id=chunk.source_observation_id,
            source_observation_captured_at_ns=chunk.source_observation_captured_at_ns,
            action_index=0,
            command_space=chunk.command_space,
            value=chunk.values[0],
        )
        decision = SafetyDecision(
            accepted=True,
            disposition="authorized",
            reason="within_envelope",
            checked_at_ns=200,
        )
        authorized = AuthorizedAction(scheduled=scheduled, decision=decision)
        receipt = ExecutionReceipt(
            command_id=scheduled.command_id,
            backend_id="sim",
            execution_mode=ExecutionMode.SIMULATION,
            command_applied=True,
            accepted=True,
            timestamp_ns=300,
            measured_state={"eef_position": [0.41, 0.0, 0.3]},
        )

        self.assertEqual(chunk.to_trace_fields()["shape"], [1, 7])
        self.assertEqual(scheduled.to_trace_fields()["action_index"], 0)
        self.assertEqual(authorized.to_trace_fields()["decision"]["reason"], "within_envelope")
        self.assertTrue(receipt.to_trace_fields()["command_applied"])

    def test_shadow_receipt_cannot_claim_a_command_was_applied(self):
        with self.assertRaisesRegex(ValueError, "hardware_shadow"):
            ExecutionReceipt(
                command_id="command-1",
                backend_id="franka_ros2_shadow",
                execution_mode=ExecutionMode.HARDWARE_SHADOW,
                command_applied=True,
                accepted=True,
                timestamp_ns=300,
            )

    def test_receipt_rejects_impossible_status_and_invalid_timestamp(self):
        with self.assertRaisesRegex(ValueError, "accepted=false"):
            ExecutionReceipt(
                command_id="command-1",
                backend_id="sim",
                execution_mode=ExecutionMode.SIMULATION,
                command_applied=True,
                accepted=False,
                timestamp_ns=300,
            )
        with self.assertRaisesRegex(ValueError, "timestamp_ns"):
            ExecutionReceipt(
                command_id="command-1",
                backend_id="sim",
                execution_mode=ExecutionMode.SIMULATION,
                command_applied=False,
                accepted=False,
                timestamp_ns=np.nan,
            )

    def test_remaining_runtime_values_preserve_phase_and_status(self):
        raw = RawObservation(
            observation_id="raw-1",
            sequence=0,
            captured_at_ns=10,
            payload={"eef": {"position": [0.4, 0.0, 0.3]}},
            source="fake",
        )
        context = InferenceContext(
            env_step=0,
            policy_call_id="policy-1",
            action_chunk_id="chunk-1",
            observation_id=raw.observation_id,
        )
        health = BackendHealth(
            backend_id="fake",
            connected=True,
            healthy=True,
            timestamp_ns=20,
        )
        result = EpisodeResult(
            run_id="run-1",
            success=True,
            steps=3,
            reason="scenario_success",
            trace_path="trace.jsonl",
        )
        error = RuntimePhaseError("policy_inference", "policy.fake", ValueError("bad tensor"))

        self.assertEqual(context.to_dict()["env_step"], 0)
        self.assertEqual(health.to_dict()["healthy"], True)
        self.assertEqual(result.to_dict()["steps"], 3)
        self.assertEqual(error.phase, "policy_inference")
        self.assertEqual(error.component_id, "policy.fake")
        self.assertEqual(error.original_type, "ValueError")


class RuntimeProtocolsTests(unittest.TestCase):
    def test_protocols_are_runtime_checkable(self):
        component = CompleteComponent()

        self.assertIsInstance(component, PolicyAdapter)
        self.assertIsInstance(component, EmbodimentAdapter)
        self.assertIsInstance(component, RuntimeBackend)
        self.assertIsInstance(component, ScenarioProvider)
        self.assertIsInstance(component, ActivationCapture)


class CompleteComponent:
    capabilities = {"component_id": "complete"}

    def load(self):
        return None

    def reset(self, scenario):
        return RawObservation(
            observation_id="raw",
            sequence=0,
            captured_at_ns=0,
            payload={},
            source="complete",
        )

    def predict(self, observation, context):
        return None

    def canonicalize_observation(self, raw):
        return None

    def decode_action_chunk(self, proposed):
        return proposed

    def connect(self):
        return None

    def observe(self):
        return self.reset(None)

    def execute(self, action):
        return None

    def health(self):
        return None

    def close(self):
        return None

    def build(self):
        return ScenarioSpec("complete", "complete", 0, 1)

    def session(self, model, context):
        return _NullContext()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _observation_kwargs():
    return {
        "observation_id": "obs",
        "sequence": 0,
        "captured_at_ns": 0,
        "images": {},
        "eef_position": np.zeros(3),
        "eef_quaternion": np.array([0.0, 0.0, 0.0, 1.0]),
        "gripper_qpos": np.zeros(2),
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "joint_efforts": np.zeros(7),
        "instruction": "test",
        "source": "fake",
        "frame_id": "base",
        "calibration_id": "test",
    }


if __name__ == "__main__":
    unittest.main()
