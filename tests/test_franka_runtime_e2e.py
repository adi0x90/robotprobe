import json
import tempfile
import unittest
from pathlib import Path

from robotprobe.integrations.franka.deterministic import (
    DeterministicFrankaBackend,
    DeterministicReachPolicy,
)
from robotprobe.integrations.franka.embodiment import (
    FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
    FrankaEmbodimentAdapter,
)
from robotprobe.integrations.franka.scenarios import DeterministicReachScenario
from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder
from robotprobe.runtime.orchestrator import EpisodeRunner
from robotprobe.runtime.safety import SafetyEnvelope, SafetyGateway
from robotprobe.runtime.scheduler import ActionScheduler
from robotprobe.runtime.types import ExecutionMode
from robotprobe.schema import validate_trace_file


class FrankaRuntimeEndToEndTests(unittest.TestCase):
    def test_deterministic_franka_episode_runs_end_to_end_with_valid_trace(self):
        backend = DeterministicFrankaBackend(clock_ns=lambda: 10_000)
        policy = DeterministicReachPolicy(step_size=0.02, action_horizon=2)
        embodiment = FrankaEmbodimentAdapter()
        scenario = DeterministicReachScenario(
            scenario_id="deterministic-reach",
            seed=5,
            max_steps=8,
            initial_position=(0.4, 0.0, 0.3),
            target_position=(0.45, 0.0, 0.3),
            tolerance=1e-8,
        )
        gateway = SafetyGateway(
            envelope=SafetyEnvelope(
                command_space=FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
                max_abs=(0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 1.0),
                workspace_min=(0.2, -0.5, 0.1),
                workspace_max=(0.7, 0.5, 0.8),
                max_observation_age_ms=10,
            ),
            execution_mode=ExecutionMode.SIMULATION,
            backend_capabilities=backend.capabilities,
        )

        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "franka-runtime.jsonl"
            with TraceRecorder(
                trace_path,
                run_id="run:franka-e2e",
                target="deterministic-franka-policy",
                scenario="deterministic-reach",
                replay_level=ReplayLevel.SIMULATOR_REPLAY,
                seed=5,
            ) as recorder:
                result = EpisodeRunner(
                    policy=policy,
                    embodiment=embodiment,
                    backend=backend,
                    scenario_provider=scenario,
                    scheduler=ActionScheduler(n_action_steps=2),
                    safety_gateway=gateway,
                    recorder=recorder,
                    clock_ns=lambda: 10_000,
                ).run()

            report = validate_trace_file(trace_path)
            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "scenario_success")
        self.assertEqual(result.steps, 3)
        self.assertTrue(report.valid, report.issues)
        event_types = [event["event_type"] for event in events]
        self.assertEqual(event_types.count("action_chunk"), 2)
        self.assertEqual(event_types.count("scheduled_action"), 3)
        self.assertEqual(event_types.count("safety_decision"), 3)
        self.assertEqual(event_types.count("execution_receipt"), 3)
        self.assertEqual(event_types[-1], "outcome")
        self.assertTrue(all(event["accepted"] for event in events if event["event_type"] == "safety_decision"))
        self.assertTrue(
            all(
                event["command_applied"]
                for event in events
                if event["event_type"] == "execution_receipt"
            )
        )
        receipts = [event for event in events if event["event_type"] == "execution_receipt"]
        self.assertAlmostEqual(receipts[-1]["measured_state"]["eef_position"][0], 0.45)
        self.assertEqual(
            [event["action_index"] for event in events if event["event_type"] == "scheduled_action"],
            [0, 1, 0],
        )
        self.assertFalse(backend.health().connected)


if __name__ == "__main__":
    unittest.main()
