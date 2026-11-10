import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder
from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.orchestrator import EpisodeRunner
from robotprobe.runtime.safety import SafetyEnvelope, SafetyGateway
from robotprobe.runtime.scheduler import ActionScheduler
from robotprobe.runtime.types import (
    CanonicalObservation,
    ExecutionMode,
    ExecutionReceipt,
    ProposedActionChunk,
    RawObservation,
    RuntimePhaseError,
    ScenarioSpec,
)
from robotprobe.schema import validate_trace_file
from robotprobe.schema import validate_trace_events


class RuntimeOrchestratorTests(unittest.TestCase):
    def test_runs_complete_lifecycle_with_stable_lineage_and_chunk_refresh(self):
        log = []
        components = make_components(log=log, done_after=3, chunk_length=2)

        result, events, trace_path = run_episode(self, components)

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "scenario_success")
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.run_id, "run:test")
        self.assertEqual(result.trace_path, str(trace_path))
        self.assertEqual(components.policy.predict_calls, 2)
        self.assertEqual(components.backend.execute_calls, 3)
        self.assertEqual(
            [event["policy_call_id"] for event in events if event["event_type"] == "action_chunk"],
            ["policy:000000", "policy:000001"],
        )
        self.assertEqual(
            [event["chunk_id"] for event in events if event["event_type"] == "scheduled_action"],
            ["chunk:000000", "chunk:000000", "chunk:000001"],
        )
        self.assertEqual(
            [event["command_id"] for event in events if event["event_type"] == "execution_receipt"],
            [
                "chunk:000000:action:000",
                "chunk:000000:action:001",
                "chunk:000001:action:000",
            ],
        )
        self.assertEqual(
            [event["observation_id"] for event in events if event["event_type"] == "observation"],
            ["observation:000000", "observation:000001", "observation:000002"],
        )
        self.assertEqual(log[-2:], ["policy.close", "backend.close"])
        self.assert_subsequence(
            log,
            [
                "scenario.build",
                "backend.connect",
                "policy.load",
                "policy.reset",
                "backend.reset",
                "embodiment.canonicalize",
                "policy.predict",
                "embodiment.decode",
                "backend.execute",
                "backend.observe",
                "policy.close",
                "backend.close",
            ],
        )
        self.assertTrue(validate_trace_file(trace_path).valid)
        self.assertEqual(events[-1]["event_type"], "outcome")

    def test_emits_complete_semantic_sequence_and_max_step_outcome(self):
        components = make_components(log=[], done_after=None, max_steps=2)

        result, events, _ = run_episode(self, components)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "max_steps_exhausted")
        self.assertEqual(result.steps, 2)
        event_types = [event["event_type"] for event in events]
        self.assertEqual(
            event_types,
            [
                "instruction",
                "observation",
                "policy_inference",
                "policy_inference",
                "action_chunk",
                "scheduled_action",
                "safety_decision",
                "execution_receipt",
                "robot_state",
                "observation",
                "scheduled_action",
                "safety_decision",
                "execution_receipt",
                "robot_state",
                "outcome",
            ],
        )
        inference = [event for event in events if event["event_type"] == "policy_inference"]
        self.assertEqual([event["status"] for event in inference], ["started", "completed"])
        self.assertEqual(inference[0]["policy_call_id"], inference[1]["policy_call_id"])

    def test_safety_rejection_records_outcome_without_calling_backend_execute(self):
        components = make_components(log=[], first_action=0.2)
        components.gateway = make_gateway(max_delta=0.05)

        result, events, _ = run_episode(self, components)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "safety_rejected:action_limit_exceeded")
        self.assertEqual(result.steps, 0)
        self.assertEqual(components.backend.execute_calls, 0)
        decision = next(event for event in events if event["event_type"] == "safety_decision")
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "action_limit_exceeded")
        self.assertNotIn("execution_receipt", [event["event_type"] for event in events])

    def test_runtime_error_is_typed_redacted_and_cleanup_runs_in_reverse_order(self):
        log = []
        components = make_components(
            log=log,
            predict_error=ValueError("token=super-secret at https://10.0.0.1/private"),
        )

        with self.assertRaises(RuntimePhaseError) as raised:
            run_episode(self, components)

        self.assertEqual(raised.exception.phase, "policy_inference")
        self.assertEqual(raised.exception.component_id, "policy.fake")
        self.assertEqual(log[-2:], ["policy.close", "backend.close"])
        events = raised.exception.trace_events
        error = next(event for event in events if event["event_type"] == "runtime_error")
        rendered = json.dumps(error)
        self.assertEqual(error["phase"], "policy_inference")
        self.assertEqual(error["component_id"], "policy.fake")
        self.assertEqual(error["error_type"], "ValueError")
        self.assertEqual(error["message"], "component operation failed")
        self.assertNotIn("super-secret", rendered)
        self.assertNotIn("10.0.0.1", rendered)

    def test_cleanup_error_does_not_replace_primary_runtime_error(self):
        log = []
        components = make_components(
            log=log,
            predict_error=ValueError("primary"),
            policy_close_error=RuntimeError("cleanup-secret"),
        )

        with self.assertRaises(RuntimePhaseError) as raised:
            run_episode(self, components)

        self.assertEqual(raised.exception.phase, "policy_inference")
        events = raised.exception.trace_events
        errors = [event for event in events if event["event_type"] == "runtime_error"]
        self.assertEqual(len(errors), 2)
        self.assertNotIn("cleanup-secret", json.dumps(errors))
        self.assertTrue(errors[1]["secondary"])
        self.assertEqual(errors[1]["phase"], "cleanup")
        self.assertEqual(log[-2:], ["policy.close", "backend.close"])

    def test_shadow_backend_claiming_applied_command_terminates_run(self):
        log = []
        components = make_components(log=log, receipt_factory=AppliedShadowReceipt)
        components.backend.capabilities = capabilities(
            "backend",
            "backend.fake-shadow",
            is_hardware=True,
            supports_shadow=True,
        )
        components.gateway = make_gateway(
            mode=ExecutionMode.HARDWARE_SHADOW,
            backend_capabilities=components.backend.capabilities,
        )

        with self.assertRaises(RuntimePhaseError) as raised:
            run_episode(self, components)

        self.assertEqual(raised.exception.phase, "execution_receipt")
        events = raised.exception.trace_events
        self.assertEqual(components.backend.execute_calls, 1)
        self.assertNotIn("execution_receipt", [event["event_type"] for event in events])
        error = next(event for event in events if event["event_type"] == "runtime_error")
        self.assertEqual(error["phase"], "execution_receipt")
        self.assertEqual(log[-2:], ["policy.close", "backend.close"])

    def test_policy_cannot_forge_observation_timestamp_used_by_safety(self):
        components = make_components(log=[], source_timestamp_offset=1)

        with self.assertRaises(RuntimePhaseError) as raised:
            run_episode(self, components)

        self.assertEqual(raised.exception.phase, "policy_output")
        self.assertEqual(components.backend.execute_calls, 0)
        error = next(
            event
            for event in raised.exception.trace_events
            if event["event_type"] == "runtime_error"
        )
        self.assertEqual(error["component_id"], "policy.fake")

    def test_receipt_backend_identity_must_match_selected_backend(self):
        components = make_components(
            log=[],
            receipt_factory=MismatchedBackendReceipt,
        )

        with self.assertRaises(RuntimePhaseError) as raised:
            run_episode(self, components)

        self.assertEqual(raised.exception.phase, "execution_receipt")
        self.assertNotIn(
            "execution_receipt",
            [event["event_type"] for event in raised.exception.trace_events],
        )

    def test_backend_must_return_the_canonical_execution_receipt_type(self):
        components = make_components(
            log=[],
            receipt_factory=DuckTypedSimulationReceipt,
        )

        with self.assertRaises(RuntimePhaseError) as raised:
            run_episode(self, components)

        self.assertEqual(raised.exception.phase, "execution_receipt")
        self.assertNotIn(
            "execution_receipt",
            [event["event_type"] for event in raised.exception.trace_events],
        )

    def test_safe_backend_refusal_is_recorded_as_a_failed_outcome(self):
        components = make_components(
            log=[],
            receipt_factory=RejectedSimulationReceipt,
        )

        result, events, _ = run_episode(self, components)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "backend_rejected:workspace_limit")
        receipt = next(
            event
            for event in events
            if event["event_type"] == "execution_receipt"
        )
        self.assertFalse(receipt["accepted"])
        self.assertFalse(receipt["command_applied"])

    def test_incompatible_components_fail_before_backend_connection(self):
        log = []
        components = make_components(log=log)
        components.embodiment.capabilities = capabilities(
            "embodiment",
            "embodiment.incompatible",
            command_spaces=["other.command.v0"],
        )

        with self.assertRaises(RuntimePhaseError) as raised:
            run_episode(self, components)

        self.assertEqual(raised.exception.phase, "compatibility")
        self.assertNotIn("backend.connect", log)
        self.assertEqual(
            [event["event_type"] for event in raised.exception.trace_events],
            ["runtime_error"],
        )

    def test_runtime_event_schema_rejects_invalid_semantic_field_types(self):
        base = {
            "run_id": "run:test",
            "target": "fake",
            "scenario": "reach",
            "replay_level": "L2_SIM_REPLAY",
            "step": 0,
            "layer": "runtime",
        }
        events = [
            {
                **base,
                "event_type": "safety_decision",
                "command_id": "command:1",
                "accepted": "yes",
                "disposition": "authorized",
                "reason": "within_envelope",
            },
            {
                **base,
                "event_type": "execution_receipt",
                "command_id": "command:1",
                "backend_id": "backend.fake",
                "execution_mode": "invalid",
                "command_applied": 1,
                "accepted": True,
            },
        ]

        report = validate_trace_events(events)

        self.assertFalse(report.valid)
        self.assertEqual(
            {(issue.line, issue.field) for issue in report.issues},
            {(1, "accepted"), (2, "execution_mode"), (2, "command_applied")},
        )

    def assert_subsequence(self, values, expected):
        position = 0
        for value in values:
            if position < len(expected) and value == expected[position]:
                position += 1
        self.assertEqual(position, len(expected), f"{expected!r} is not a subsequence of {values!r}")


def capabilities(kind, component_id, **overrides):
    data = {
        "format": "robotprobe.runtime.component.v0",
        "kind": kind,
        "component_id": component_id,
        "command_spaces": (
            [] if kind == "scenario" else ["franka.cartesian_delta_gripper.v0"]
        ),
        "observation_streams": [],
        "runtime_family": "test",
        "is_hardware": False,
        "supports_reset": kind == "backend",
        "supports_ground_truth": kind == "backend",
        "supports_shadow": False,
        "can_apply_hardware_commands": False,
        "requires_reset": kind == "scenario",
        "optional_dependencies": [],
    }
    data.update(overrides)
    return RuntimeComponentCapabilities.from_dict(data)


class Components:
    def __init__(self, *, policy, embodiment, backend, scenario, scheduler, gateway):
        self.policy = policy
        self.embodiment = embodiment
        self.backend = backend
        self.scenario = scenario
        self.scheduler = scheduler
        self.gateway = gateway


class FakeScenario:
    capabilities = capabilities("scenario", "scenario.fake")

    def __init__(self, log, *, max_steps):
        self.log = log
        self.max_steps = max_steps

    def build(self):
        self.log.append("scenario.build")
        return ScenarioSpec(
            scenario_id="reach",
            instruction="Move to the target.",
            seed=7,
            max_steps=self.max_steps,
            options={"target": [0.45, 0.0, 0.3]},
        )


class FakePolicy:
    capabilities = capabilities("policy", "policy.fake")

    def __init__(
        self,
        log,
        *,
        chunk_length,
        first_action,
        source_timestamp_offset=0,
        predict_error=None,
        close_error=None,
    ):
        self.log = log
        self.chunk_length = chunk_length
        self.first_action = first_action
        self.source_timestamp_offset = source_timestamp_offset
        self.predict_error = predict_error
        self.close_error = close_error
        self.predict_calls = 0

    def load(self):
        self.log.append("policy.load")

    def reset(self, scenario):
        self.log.append("policy.reset")
        self.scenario = scenario

    def predict(self, observation, context):
        self.log.append("policy.predict")
        if self.predict_error is not None:
            raise self.predict_error
        self.predict_calls += 1
        values = np.zeros((self.chunk_length, 7), dtype=np.float32)
        values[:, 0] = self.first_action
        return ProposedActionChunk(
            chunk_id=context.action_chunk_id,
            policy_call_id=context.policy_call_id,
            source_observation_id=context.observation_id,
            source_observation_captured_at_ns=(
                observation.captured_at_ns + self.source_timestamp_offset
            ),
            command_space="franka.cartesian_delta_gripper.v0",
            values=values,
        )

    def close(self):
        self.log.append("policy.close")
        if self.close_error is not None:
            raise self.close_error


class FakeEmbodiment:
    capabilities = capabilities("embodiment", "embodiment.fake")

    def __init__(self, log):
        self.log = log

    def canonicalize_observation(self, raw):
        self.log.append("embodiment.canonicalize")
        payload = raw.payload
        return CanonicalObservation(
            observation_id=raw.observation_id,
            sequence=raw.sequence,
            captured_at_ns=raw.captured_at_ns,
            images={"agent": payload["image"]},
            eef_position=payload["eef_position"],
            eef_quaternion=[0.0, 0.0, 0.0, 1.0],
            gripper_qpos=[0.02, 0.02],
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            joint_efforts=np.zeros(7),
            instruction=payload["instruction"],
            source=raw.source,
            frame_id="franka_base",
            calibration_id="fake-v1",
        )

    def decode_action_chunk(self, proposed):
        self.log.append("embodiment.decode")
        return proposed


class FakeBackend:
    capabilities = capabilities("backend", "backend.fake")

    def __init__(self, log, *, done_after, receipt_factory):
        self.log = log
        self.done_after = done_after
        self.receipt_factory = receipt_factory
        self.execute_calls = 0
        self.sequence = 0

    def connect(self):
        self.log.append("backend.connect")

    def reset(self, scenario):
        self.log.append("backend.reset")
        self.scenario = scenario
        self.sequence = 0
        return self._observation()

    def observe(self):
        self.log.append("backend.observe")
        self.sequence += 1
        return self._observation()

    def execute(self, action):
        self.log.append("backend.execute")
        self.execute_calls += 1
        done = self.done_after is not None and self.execute_calls >= self.done_after
        return self.receipt_factory(
            command_id=action.command_id,
            backend_id=self.capabilities.component_id,
            done=done,
        )

    def health(self):
        raise AssertionError("health is not used by this orchestration contract")

    def close(self):
        self.log.append("backend.close")

    def _observation(self):
        return RawObservation(
            observation_id=f"observation:{self.sequence:06d}",
            sequence=self.sequence,
            captured_at_ns=1_000,
            payload={
                "image": np.full((2, 2, 3), self.sequence, dtype=np.uint8),
                "eef_position": [0.4 + self.sequence * 0.01, 0.0, 0.3],
                "instruction": self.scenario.instruction,
            },
            source="fake_backend",
        )


def SimulationReceipt(*, command_id, backend_id, done):
    return ExecutionReceipt(
        command_id=command_id,
        backend_id=backend_id,
        execution_mode=ExecutionMode.SIMULATION,
        command_applied=True,
        accepted=True,
        timestamp_ns=1_000,
        measured_state={"eef_position": [0.41, 0.0, 0.3]},
        done=done,
        success=done,
    )


def MismatchedBackendReceipt(*, command_id, backend_id, done):
    return SimulationReceipt(
        command_id=command_id,
        backend_id=f"{backend_id}.other",
        done=done,
    )


def RejectedSimulationReceipt(*, command_id, backend_id, done):
    return ExecutionReceipt(
        command_id=command_id,
        backend_id=backend_id,
        execution_mode=ExecutionMode.SIMULATION,
        command_applied=False,
        accepted=False,
        timestamp_ns=1_000,
        measured_state={"eef_position": [0.4, 0.0, 0.3]},
        done=True,
        success=False,
        metadata={"reason": "workspace_limit"},
    )


class AppliedShadowReceipt:
    def __init__(self, *, command_id, backend_id, done):
        self.command_id = command_id
        self.backend_id = backend_id
        self.execution_mode = ExecutionMode.HARDWARE_SHADOW
        self.command_applied = True
        self.accepted = True
        self.timestamp_ns = 1_000
        self.measured_state = {}
        self.done = done
        self.success = False


class DuckTypedSimulationReceipt:
    def __init__(self, *, command_id, backend_id, done):
        self.command_id = command_id
        self.backend_id = backend_id
        self.execution_mode = ExecutionMode.SIMULATION
        self.command_applied = True
        self.accepted = True
        self.timestamp_ns = 1_000
        self.measured_state = {"eef_position": [0.4, 0.0, 0.3]}
        self.done = done
        self.success = done
        self.metadata = {}


def make_gateway(
    *,
    max_delta=0.1,
    mode=ExecutionMode.SIMULATION,
    backend_capabilities=None,
):
    return SafetyGateway(
        envelope=SafetyEnvelope(
            command_space="franka.cartesian_delta_gripper.v0",
            max_abs=(max_delta, max_delta, max_delta, 0.2, 0.2, 0.2, 1.0),
            workspace_min=(0.2, -0.5, 0.1),
            workspace_max=(0.7, 0.5, 0.8),
            max_observation_age_ms=10.0,
        ),
        execution_mode=mode,
        backend_capabilities=backend_capabilities
        or capabilities("backend", "backend.fake"),
    )


def make_components(
    *,
    log,
    done_after=1,
    max_steps=5,
    chunk_length=2,
    first_action=0.01,
    predict_error=None,
    policy_close_error=None,
    source_timestamp_offset=0,
    receipt_factory=SimulationReceipt,
):
    policy = FakePolicy(
        log,
        chunk_length=chunk_length,
        first_action=first_action,
        source_timestamp_offset=source_timestamp_offset,
        predict_error=predict_error,
        close_error=policy_close_error,
    )
    embodiment = FakeEmbodiment(log)
    backend = FakeBackend(log, done_after=done_after, receipt_factory=receipt_factory)
    scenario = FakeScenario(log, max_steps=max_steps)
    return Components(
        policy=policy,
        embodiment=embodiment,
        backend=backend,
        scenario=scenario,
        scheduler=ActionScheduler(n_action_steps=2),
        gateway=make_gateway(backend_capabilities=backend.capabilities),
    )


def run_episode(testcase, components):
    directory = tempfile.TemporaryDirectory()
    testcase.addCleanup(directory.cleanup)
    trace_path = Path(directory.name) / "runtime.jsonl"
    caught = None
    result = None
    with TraceRecorder(
        trace_path,
        run_id="run:test",
        target="fake-policy",
        scenario="reach",
        replay_level=ReplayLevel.SIMULATOR_REPLAY,
    ) as recorder:
        runner = EpisodeRunner(
            policy=components.policy,
            embodiment=components.embodiment,
            backend=components.backend,
            scenario_provider=components.scenario,
            scheduler=components.scheduler,
            safety_gateway=components.gateway,
            recorder=recorder,
            clock_ns=lambda: 1_000,
        )
        try:
            result = runner.run()
        except Exception as error:
            caught = error
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if caught is not None:
        caught.trace_events = events
        caught.trace_path = trace_path
        # The caller needs both the raised error and trace evidence; attaching the
        # evidence lets unittest retain normal assertRaises behavior.
        raise caught
    assert result is not None
    return result, events, trace_path


if __name__ == "__main__":
    unittest.main()
