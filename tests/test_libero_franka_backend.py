import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from robotprobe.integrations.franka.embodiment import FRANKA_CARTESIAN_DELTA_COMMAND_SPACE
from robotprobe.integrations.franka.libero_backend import (
    LiberoEnvironmentSettings,
    LiberoFrankaBackend,
    LiberoOptionalDependencyError,
    LiberoScenarioProvider,
)
from robotprobe.runtime.conformance import assert_backend_conforms
from robotprobe.runtime.types import (
    AuthorizedAction,
    ExecutionMode,
    SafetyDecision,
    ScheduledAction,
)


ROOT = Path(__file__).resolve().parents[1]


class LiberoFrankaBackendTests(unittest.TestCase):
    def test_module_import_is_lazy_without_lerobot_or_libero(self):
        code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "lerobot" or name.startswith("lerobot.") or name == "libero" or name.startswith("libero."):
        raise AssertionError(f"heavy import attempted: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import robotprobe.integrations.franka.libero_backend
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

    def test_optional_dependency_failure_is_precise_at_connect(self):
        def missing_factory(_settings):
            raise ImportError("No module named 'libero'")

        backend = LiberoFrankaBackend(environment_factory=missing_factory)

        with self.assertRaises(LiberoOptionalDependencyError) as raised:
            backend.connect()

        self.assertIn("LIBERO", str(raised.exception))
        self.assertIn("connect()", str(raised.exception))

    def test_connect_passes_suite_task_relative_control_and_image_configuration(self):
        factory = FakeEnvironmentFactory()
        backend = LiberoFrankaBackend(
            suite="libero_object",
            task_id=4,
            episode_length=123,
            observation_height=240,
            observation_width=320,
            control_mode="relative",
            environment_factory=factory,
            clock_ns=lambda: 1_000,
        )

        backend.connect()

        self.assertEqual(len(factory.settings), 1)
        settings = factory.settings[0]
        self.assertIsInstance(settings, LiberoEnvironmentSettings)
        self.assertEqual(settings.suite, "libero_object")
        self.assertEqual(settings.task_id, 4)
        self.assertEqual(settings.episode_length, 123)
        self.assertEqual(settings.observation_height, 240)
        self.assertEqual(settings.observation_width, 320)
        self.assertEqual(settings.control_mode, "relative")
        self.assertEqual(settings.obs_type, "pixels_agent_pos")
        self.assertEqual(
            settings.camera_name,
            "agentview_image,robot0_eye_in_hand_image",
        )
        with self.assertRaisesRegex(ValueError, "relative"):
            LiberoFrankaBackend(control_mode="absolute")

    def test_reset_uses_scenario_seed_and_extracts_canonical_franka_payload(self):
        environment = FakeVectorEnvironment()
        backend = make_backend(environment)
        scenario = make_scenario(seed=19)
        backend.connect()

        raw = backend.reset(scenario)

        self.assertEqual(environment.reset_seeds, [19])
        self.assertEqual(raw.observation_id, "libero:libero_spatial:0:000000")
        self.assertEqual(raw.sequence, 0)
        self.assertEqual(raw.payload["instruction"], environment.task_description)
        self.assertEqual(raw.payload["quaternion_convention"], "xyzw")
        self.assertEqual(raw.payload["eef_position"].tolist(), [0.4, 0.0, 0.3])
        self.assertEqual(raw.payload["eef_quaternion"].tolist(), [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(raw.payload["gripper_qpos"].tolist(), [0.02, 0.02])
        self.assertEqual(raw.payload["joint_positions"].shape, (7,))
        self.assertEqual(raw.payload["joint_velocities"].shape, (7,))
        self.assertEqual(raw.payload["joint_efforts"].tolist(), [0.0] * 7)
        self.assertEqual(raw.payload["images"]["agent"].shape, (3, 4, 3))
        self.assertEqual(raw.payload["images"]["wrist"].shape, (3, 4, 3))
        self.assertFalse(raw.metadata["joint_efforts_available"])

    def test_reset_rejects_environment_and_scenario_instruction_mismatch(self):
        environment = FakeVectorEnvironment()
        backend = make_backend(environment)
        scenario = LiberoScenarioProvider(
            suite="libero_spatial",
            task_id=0,
            seed=7,
            max_steps=5,
            instruction="Put the red mug on the plate.",
        ).build()
        backend.connect()

        with self.assertRaisesRegex(ValueError, "instruction"):
            backend.reset(scenario)

    def test_execute_batches_action_and_extracts_success_across_info_variants(self):
        variants = (
            {"is_success": np.array([True])},
            {"final_info": {"is_success": np.array([True])}},
            [{"success": True}],
            {"task_success": [True]},
        )
        for info in variants:
            with self.subTest(info=repr(info)):
                environment = FakeVectorEnvironment(
                    step_results=[
                        (
                            make_vector_observation(position_x=0.41),
                            np.array([1.0]),
                            np.array([False]),
                            np.array([False]),
                            info,
                        )
                    ]
                )
                backend = make_backend(environment)
                backend.connect()
                backend.reset(make_scenario())

                receipt = backend.execute(
                    authorized_action([0.1, 0, 0, 0, 0, 0, 1.0])
                )

                self.assertEqual(environment.actions[0].shape, (1, 7))
                self.assertEqual(environment.actions[0][0, 0], 0.1)
                self.assertTrue(receipt.accepted)
                self.assertTrue(receipt.command_applied)
                self.assertTrue(receipt.done)
                self.assertTrue(receipt.success)
                self.assertEqual(receipt.execution_mode, ExecutionMode.SIMULATION)
                self.assertEqual(receipt.measured_state["eef_position"][0], 0.41)
                self.assertEqual(backend.observe().sequence, 1)

    def test_four_tuple_step_and_max_step_termination_are_normalized(self):
        environment = FakeVectorEnvironment(
            step_results=[
                (
                    make_vector_observation(position_x=0.41),
                    np.array([0.0]),
                    np.array([False]),
                    {"is_success": np.array([False])},
                )
            ]
        )
        backend = make_backend(environment)
        backend.connect()
        backend.reset(make_scenario(max_steps=1))

        receipt = backend.execute(authorized_action([0.1, 0, 0, 0, 0, 0, 1.0]))

        self.assertTrue(receipt.done)
        self.assertFalse(receipt.success)
        self.assertEqual(receipt.metadata["reason"], "max_steps_exhausted")
        self.assertFalse(receipt.metadata["environment_terminated"])
        self.assertFalse(receipt.metadata["environment_truncated"])

    def test_environment_cache_is_instance_owned_and_close_is_deterministic(self):
        factory = FakeEnvironmentFactory()
        backend = LiberoFrankaBackend(
            environment_factory=factory,
            clock_ns=lambda: 1_000,
        )
        backend.connect()
        backend.connect()
        backend.reset(make_scenario())
        backend.reset(make_scenario(seed=8))

        backend.close()
        backend.close()
        backend.connect()

        self.assertEqual(len(factory.environments), 2)
        self.assertEqual(factory.environments[0].close_calls, 1)
        self.assertEqual(factory.environments[1].close_calls, 0)
        self.assertTrue(backend.health().connected)

    def test_injected_fake_environment_passes_public_backend_conformance(self):
        environment = FakeVectorEnvironment()
        backend = make_backend(environment)

        report = assert_backend_conforms(
            backend=backend,
            scenario=make_scenario(max_steps=3),
            action=authorized_action([0.01, 0, 0, 0, 0, 0, 1.0]),
        )

        self.assertTrue(report.passed, report.issues)
        self.assertEqual(environment.close_calls, 1)


class LiberoScenarioProviderTests(unittest.TestCase):
    def test_uses_injected_task_metadata_and_preserves_suite_task_seed(self):
        calls = []

        def metadata(suite, task_id):
            calls.append((suite, task_id))
            return "Put the bowl on the plate."

        provider = LiberoScenarioProvider(
            suite="libero_goal",
            task_id=2,
            seed=11,
            max_steps=77,
            metadata_loader=metadata,
        )

        scenario = provider.build()

        self.assertEqual(calls, [("libero_goal", 2)])
        self.assertEqual(scenario.instruction, "Put the bowl on the plate.")
        self.assertEqual(scenario.seed, 11)
        self.assertEqual(scenario.max_steps, 77)
        self.assertEqual(scenario.options["suite"], "libero_goal")
        self.assertEqual(scenario.options["task_id"], 2)


class FakeEnvironmentFactory:
    def __init__(self):
        self.settings = []
        self.environments = []

    def __call__(self, settings):
        self.settings.append(settings)
        environment = FakeVectorEnvironment()
        self.environments.append(environment)
        return environment


class FakeVectorEnvironment:
    num_envs = 1
    task_description = "Pick up the black bowl."

    def __init__(self, *, step_results=None):
        self.step_results = list(step_results or [])
        self.reset_seeds = []
        self.actions = []
        self.close_calls = 0

    def reset(self, *, seed):
        self.reset_seeds.append(seed)
        return make_vector_observation(), {"is_success": np.array([False])}

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        if self.step_results:
            return self.step_results.pop(0)
        return (
            make_vector_observation(position_x=0.41),
            np.array([0.0]),
            np.array([False]),
            np.array([False]),
            {"is_success": np.array([False])},
        )

    def call(self, name):
        if name != "task_description":
            raise KeyError(name)
        return [self.task_description]

    def close(self):
        self.close_calls += 1


def make_backend(environment):
    return LiberoFrankaBackend(
        suite="libero_spatial",
        task_id=0,
        environment_factory=lambda _settings: environment,
        clock_ns=lambda: 1_000,
    )


def make_scenario(*, seed=7, max_steps=5):
    return LiberoScenarioProvider(
        suite="libero_spatial",
        task_id=0,
        seed=seed,
        max_steps=max_steps,
        instruction="Pick up the black bowl.",
    ).build()


def make_vector_observation(*, position_x=0.4):
    return {
        "pixels": {
            "image": np.full((1, 3, 4, 3), 11, dtype=np.uint8),
            "image2": np.full((1, 3, 4, 3), 22, dtype=np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": np.array([[position_x, 0.0, 0.3]], dtype=np.float64),
                "quat": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            },
            "gripper": {
                "qpos": np.array([[0.02, 0.02]], dtype=np.float64),
                "qvel": np.zeros((1, 2), dtype=np.float64),
            },
            "joints": {
                "pos": np.zeros((1, 7), dtype=np.float64),
                "vel": np.zeros((1, 7), dtype=np.float64),
            },
        },
    }


def authorized_action(value):
    scheduled = ScheduledAction(
        command_id="chunk:000000:action:000",
        chunk_id="chunk:000000",
        policy_call_id="policy:000000",
        source_observation_id="libero:libero_spatial:0:000000",
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
