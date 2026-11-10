import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from robotprobe.integrations.franka.embodiment import (
    FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
    FrankaEmbodimentAdapter,
)
from robotprobe.integrations.franka.ros2_shadow import (
    FrankaRos2ShadowBackend,
    FrankaShadowOptionalDependencyError,
    FrankaShadowScenario,
    FrankaShadowSnapshot,
    FrankaShadowStateError,
    FrankaTransportHealth,
)
from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder
from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.config import inspect_runtime_config
from robotprobe.runtime.conformance import assert_backend_conforms
from robotprobe.runtime.orchestrator import EpisodeRunner
from robotprobe.runtime.safety import SafetyEnvelope, SafetyGateway
from robotprobe.runtime.scheduler import ActionScheduler
from robotprobe.runtime.types import (
    AuthorizedAction,
    ExecutionMode,
    InferenceContext,
    ProposedActionChunk,
    SafetyDecision,
    ScheduledAction,
)
from robotprobe.schema import validate_trace_file


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "examples" / "packs" / "franka-pi05-runtime"
CONFIG = PACK / "configs" / "franka-ros2-shadow.yaml"
NOW_NS = 10_000_000_000


class FrankaRos2ShadowBackendTests(unittest.TestCase):
    def test_transport_timestamps_and_backend_clock_must_be_integer_and_finite(self):
        for captured_at_ns in (np.nan, np.inf, 1.5, -1, True):
            with self.subTest(
                snapshot_timestamp=captured_at_ns
            ), self.assertRaisesRegex(ValueError, "timestamp"):
                make_snapshot(captured_at_ns=captured_at_ns)
            with self.subTest(
                health_timestamp=captured_at_ns
            ), self.assertRaisesRegex(ValueError, "timestamp"):
                FrankaTransportHealth(
                    connected=True,
                    healthy=True,
                    captured_at_ns=captured_at_ns,
                    robot_mode="idle",
                    controller_authority="none",
                )

        transport = FakeReadOnlyTransport()
        backend = FrankaRos2ShadowBackend(
            transport=transport,
            clock_ns=lambda: np.nan,
        )
        backend.connect()
        with self.assertRaisesRegex(FrankaShadowStateError, "clock"):
            backend.reset(make_scenario())

    def test_transport_health_requires_boolean_status_and_string_identity(self):
        base = {
            "connected": True,
            "healthy": True,
            "captured_at_ns": NOW_NS,
            "robot_mode": "idle",
            "controller_authority": "none",
        }
        for field_name, value in (
            ("connected", 1),
            ("healthy", "true"),
            ("robot_mode", 7),
            ("controller_authority", object()),
        ):
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                FrankaTransportHealth(**{**base, field_name: value})

    def test_module_import_is_lazy_without_ros2(self):
        code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "rclpy" or name.startswith("rclpy.") or name.startswith("franka_msgs"):
        raise AssertionError(f"ROS 2 import attempted: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import robotprobe.integrations.franka.ros2_shadow
print("lazy-import-ok")
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "lazy-import-ok")

    def test_capabilities_are_hardware_shadow_without_command_authority(self):
        capabilities = FrankaRos2ShadowBackend.capabilities

        self.assertTrue(capabilities.is_hardware)
        self.assertTrue(capabilities.supports_shadow)
        self.assertFalse(capabilities.can_apply_hardware_commands)
        self.assertFalse(capabilities.supports_ground_truth)
        self.assertEqual(
            capabilities.command_spaces,
            (FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,),
        )

    def test_missing_optional_transport_is_reported_only_at_connect(self):
        def missing_transport(_options):
            raise ImportError("No module named 'robotprobe_franka_ros2'")

        backend = FrankaRos2ShadowBackend(transport_loader=missing_transport)

        with self.assertRaises(FrankaShadowOptionalDependencyError) as raised:
            backend.connect()

        self.assertIn("read-only", str(raised.exception))
        self.assertIn("connect()", str(raised.exception))

    def test_logical_reset_maps_timestamped_physical_state_without_commanding(self):
        transport = FakeReadOnlyTransport()
        backend = make_backend(transport)
        backend.connect()

        raw = backend.reset(make_scenario())

        self.assertEqual(raw.observation_id, "franka-shadow:000000")
        self.assertEqual(raw.sequence, 0)
        self.assertEqual(raw.captured_at_ns, NOW_NS - 1_000_000)
        self.assertEqual(raw.payload["images"]["agent"].shape, (3, 4, 3))
        self.assertEqual(raw.payload["images"]["wrist"].shape, (3, 4, 3))
        self.assertEqual(raw.payload["eef_position"].tolist(), [0.4, 0.0, 0.3])
        self.assertEqual(raw.payload["eef_quaternion"].tolist(), [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(raw.payload["gripper_qpos"].tolist(), [0.02, 0.02])
        self.assertEqual(raw.payload["joint_positions"].shape, (7,))
        self.assertEqual(raw.payload["robot_mode"], "idle")
        self.assertEqual(raw.payload["controller_authority"], "read_only")
        self.assertEqual(raw.metadata["reset_semantics"], "logical_session_only")
        self.assertEqual(transport.read_calls, 1)
        self.assertEqual(transport.publish_calls, 0)
        self.assertEqual(transport.send_calls, 0)

    def test_connect_fails_closed_when_transport_is_disconnected(self):
        transport = FakeReadOnlyTransport(health_connected=False)
        backend = make_backend(transport)

        with self.assertRaisesRegex(FrankaShadowStateError, "disconnected"):
            backend.connect()

        self.assertEqual(transport.close_calls, 1)
        self.assertFalse(backend.health().connected)

    def test_stale_and_future_dated_state_are_rejected(self):
        for timestamp_ns, expected in (
            (NOW_NS - 101_000_000, "stale"),
            (NOW_NS + 1, "clock skew"),
        ):
            with self.subTest(timestamp_ns=timestamp_ns):
                transport = FakeReadOnlyTransport(
                    snapshots=[make_snapshot(captured_at_ns=timestamp_ns)]
                )
                backend = make_backend(transport, max_state_age_ms=100)
                backend.connect()

                with self.assertRaisesRegex(FrankaShadowStateError, expected):
                    backend.reset(make_scenario())

                self.assertEqual(transport.publish_calls, 0)
                backend.close()

    def test_execute_records_shadow_receipt_and_never_calls_a_command_method(self):
        transport = FakeReadOnlyTransport()
        backend = make_backend(transport)
        backend.connect()
        raw = backend.reset(make_scenario())

        receipt = backend.execute(
            authorized_action(
                source_observation_id=raw.observation_id,
                disposition="shadow_only",
            )
        )

        self.assertTrue(receipt.accepted)
        self.assertFalse(receipt.command_applied)
        self.assertFalse(receipt.done)
        self.assertFalse(receipt.success)
        self.assertEqual(receipt.execution_mode, ExecutionMode.HARDWARE_SHADOW)
        self.assertEqual(receipt.metadata["reason"], "shadow_evaluated_not_applied")
        self.assertEqual(receipt.metadata["robot_mode"], "idle")
        self.assertEqual(receipt.metadata["controller_authority"], "read_only")
        self.assertEqual(receipt.measured_state["observation_id"], raw.observation_id)
        self.assertEqual(transport.read_calls, 1)
        self.assertEqual(transport.publish_calls, 0)
        self.assertEqual(transport.send_calls, 0)

    def test_execute_rejects_non_shadow_authorization_even_if_marked_accepted(self):
        transport = FakeReadOnlyTransport()
        backend = make_backend(transport)
        backend.connect()
        raw = backend.reset(make_scenario())

        with self.assertRaisesRegex(ValueError, "shadow_only"):
            backend.execute(
                authorized_action(
                    source_observation_id=raw.observation_id,
                    disposition="authorized",
                )
            )

        self.assertEqual(transport.publish_calls, 0)
        self.assertEqual(transport.send_calls, 0)

    def test_observe_reads_a_new_snapshot_and_close_is_idempotent(self):
        transport = FakeReadOnlyTransport(
            snapshots=[
                make_snapshot(position_x=0.4),
                make_snapshot(position_x=0.41),
            ]
        )
        backend = make_backend(transport)
        backend.connect()
        backend.reset(make_scenario())

        observed = backend.observe()
        backend.close()
        backend.close()

        self.assertEqual(observed.sequence, 1)
        self.assertEqual(observed.payload["eef_position"][0], 0.41)
        self.assertEqual(transport.read_calls, 2)
        self.assertEqual(transport.close_calls, 1)
        self.assertFalse(backend.health().connected)

    def test_close_failure_still_invalidates_the_observation_session(self):
        transport = FakeReadOnlyTransport(close_error=RuntimeError("close failed"))
        backend = make_backend(transport)
        backend.connect()
        backend.reset(make_scenario())

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            backend.close()

        transport.close_error = None
        backend.connect()
        with self.assertRaisesRegex(FrankaShadowStateError, "reset before use"):
            backend.observe()
        backend.close()

    def test_injected_transport_passes_public_backend_conformance(self):
        transport = FakeReadOnlyTransport()
        backend = make_backend(transport)

        report = assert_backend_conforms(
            backend=backend,
            scenario=make_scenario(),
            action=authorized_action(
                source_observation_id="franka-shadow:000000",
                disposition="shadow_only",
            ),
        )

        self.assertTrue(report.passed, report.issues)
        self.assertEqual(transport.publish_calls, 0)
        self.assertEqual(transport.send_calls, 0)

    def test_complete_runner_preserves_shadow_invariant(self):
        transport = FakeReadOnlyTransport()
        backend = make_backend(transport)
        gateway = SafetyGateway(
            envelope=SafetyEnvelope(
                command_space=FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
                max_abs=(0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 1.0),
                max_observation_age_ms=100,
            ),
            execution_mode=ExecutionMode.HARDWARE_SHADOW,
            backend_capabilities=backend.capabilities,
        )

        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "shadow.jsonl"
            with TraceRecorder(
                trace_path,
                run_id="run:franka-shadow",
                target="fixed-shadow-policy",
                scenario="franka-shadow-observation",
                replay_level=ReplayLevel.SHADOW_HARDWARE_REPLAY,
                seed=7,
            ) as recorder:
                result = EpisodeRunner(
                    policy=FixedShadowPolicy(),
                    embodiment=FrankaEmbodimentAdapter(),
                    backend=backend,
                    scenario_provider=FrankaShadowScenario(max_steps=2),
                    scheduler=ActionScheduler(n_action_steps=1),
                    safety_gateway=gateway,
                    recorder=recorder,
                    clock_ns=lambda: NOW_NS,
                ).run()
            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report = validate_trace_file(trace_path)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "max_steps_exhausted")
        self.assertTrue(report.valid, report.issues)
        receipts = [
            event for event in events if event["event_type"] == "execution_receipt"
        ]
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(not event["command_applied"] for event in receipts))
        self.assertEqual(transport.publish_calls, 0)
        self.assertEqual(transport.send_calls, 0)


class FrankaShadowConfigTests(unittest.TestCase):
    def test_pack_config_is_valid_only_in_hardware_shadow_mode(self):
        report = inspect_runtime_config(CONFIG)

        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["execution"]["mode"], "hardware_shadow")
        self.assertTrue(
            report["components"]["backend"]["capabilities"]["is_hardware"]
        )
        self.assertFalse(
            report["components"]["backend"]["capabilities"][
                "can_apply_hardware_commands"
            ]
        )

        source = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        for mode in ("simulation", "hardware_executable"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                changed = copy.deepcopy(source)
                changed["pack"] = str(PACK)
                changed["capture"]["profile"] = str(
                    PACK / "capture" / "pi05-semantic.yaml"
                )
                changed["execution"]["mode"] = mode
                path = Path(td) / "runtime.yaml"
                path.write_text(yaml.safe_dump(changed), encoding="utf-8")

                changed_report = inspect_runtime_config(path)

                self.assertFalse(changed_report["valid"])
                self.assertTrue(
                    any(
                        "hardware" in issue or "simulation" in issue
                        for issue in changed_report["issues"]
                    ),
                    changed_report["issues"],
                )


class FakeReadOnlyTransport:
    def __init__(
        self,
        *,
        snapshots=None,
        health_connected=True,
        health_healthy=True,
        close_error=None,
    ):
        self.snapshots = list(snapshots or [make_snapshot()])
        self.health_connected = health_connected
        self.health_healthy = health_healthy
        self.close_error = close_error
        self.connected = False
        self.connect_calls = 0
        self.close_calls = 0
        self.read_calls = 0
        self.publish_calls = 0
        self.send_calls = 0

    def connect(self):
        self.connect_calls += 1
        self.connected = True

    def health(self):
        return FrankaTransportHealth(
            connected=self.connected and self.health_connected,
            healthy=self.connected and self.health_healthy,
            captured_at_ns=NOW_NS,
            robot_mode="idle",
            controller_authority="read_only",
            details={"transport": "fake"},
        )

    def read_snapshot(self):
        self.read_calls += 1
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]

    def publish_action(self, _action):
        self.publish_calls += 1
        raise AssertionError("shadow backend must not publish actions")

    def send_command(self, _action):
        self.send_calls += 1
        raise AssertionError("shadow backend must not send commands")

    def close(self):
        self.close_calls += 1
        self.connected = False
        if self.close_error is not None:
            raise self.close_error


class FixedShadowPolicy:
    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "policy",
            "component_id": "franka.policy.fixed_shadow.v0",
            "command_spaces": [FRANKA_CARTESIAN_DELTA_COMMAND_SPACE],
            "observation_streams": ["rgb", "eef_pose", "gripper", "joint_state"],
            "runtime_family": "test",
            "is_hardware": False,
            "supports_reset": False,
            "supports_ground_truth": False,
            "supports_shadow": False,
            "can_apply_hardware_commands": False,
            "requires_reset": False,
            "optional_dependencies": [],
        }
    )

    def load(self):
        pass

    def reset(self, _scenario):
        pass

    def predict(self, observation, context: InferenceContext):
        return ProposedActionChunk(
            chunk_id=context.action_chunk_id,
            policy_call_id=context.policy_call_id,
            source_observation_id=observation.observation_id,
            source_observation_captured_at_ns=observation.captured_at_ns,
            command_space=FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
            values=[[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        )

    def close(self):
        pass


def make_backend(
    transport,
    *,
    max_state_age_ms=100,
):
    return FrankaRos2ShadowBackend(
        transport=transport,
        max_state_age_ms=max_state_age_ms,
        clock_ns=lambda: NOW_NS,
    )


def make_scenario(*, max_steps=3):
    return FrankaShadowScenario(
        scenario_id="franka-shadow-observation",
        instruction="Observe the Franka while evaluating a proposed manipulation.",
        seed=7,
        max_steps=max_steps,
    ).build()


def make_snapshot(
    *,
    captured_at_ns=NOW_NS - 1_000_000,
    position_x=0.4,
):
    return FrankaShadowSnapshot(
        captured_at_ns=captured_at_ns,
        images={
            "agent": np.full((3, 4, 3), 11, dtype=np.uint8),
            "wrist": np.full((3, 4, 3), 22, dtype=np.uint8),
        },
        eef_position=[position_x, 0.0, 0.3],
        eef_quaternion=[0.0, 0.0, 0.0, 1.0],
        gripper_qpos=[0.02, 0.02],
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        joint_efforts=np.zeros(7),
        frame_id="franka_link0",
        calibration_id="lab-franka-camera-calibration-v1",
        robot_mode="idle",
        controller_authority="read_only",
        metadata={"source": "fake_transport"},
    )


def authorized_action(
    *,
    source_observation_id,
    disposition,
):
    scheduled = ScheduledAction(
        command_id="chunk:000000:action:000",
        chunk_id="chunk:000000",
        policy_call_id="policy:000000",
        source_observation_id=source_observation_id,
        source_observation_captured_at_ns=NOW_NS - 1_000_000,
        action_index=0,
        command_space=FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
        value=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )
    return AuthorizedAction(
        scheduled=scheduled,
        decision=SafetyDecision(
            accepted=True,
            disposition=disposition,
            reason="within_envelope",
            checked_at_ns=NOW_NS,
        ),
    )


if __name__ == "__main__":
    unittest.main()
