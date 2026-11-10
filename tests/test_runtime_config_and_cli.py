import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from robotprobe.integrations.franka.deterministic import (
    DeterministicFrankaBackend,
    DeterministicReachPolicy,
)
from robotprobe.integrations.franka.embodiment import FrankaEmbodimentAdapter
from robotprobe.integrations.franka.scenarios import DeterministicReachScenario
from robotprobe.runtime.config import (
    RUNTIME_CONFIG_FORMAT,
    RuntimeConfigError,
    inspect_runtime_config,
    load_runtime_config,
    run_runtime_config,
)
from robotprobe.runtime.registry import RuntimePluginRegistry
from robotprobe.runtime.types import ExecutionMode


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "examples" / "packs" / "franka-pi05-runtime"
EXAMPLE_CONFIG = PACK_DIR / "configs" / "deterministic.yaml"
PI05_CONFIG = PACK_DIR / "configs" / "pi05-libero.yaml"
PI05_CAPTURE_PROFILE = PACK_DIR / "capture" / "pi05-semantic.yaml"


class RuntimeConfigurationTests(unittest.TestCase):
    def test_loads_yaml_and_json_and_resolves_paths_from_config_location(self):
        yaml_config = load_runtime_config(EXAMPLE_CONFIG)

        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "runtime.json"
            data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
            data["pack"] = str(PACK_DIR)
            data["trace"]["out"] = "artifacts/from-json.jsonl"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            json_config = load_runtime_config(config_path)

            self.assertEqual(
                json_config.trace.out,
                (Path(td) / "artifacts/from-json.jsonl").resolve(),
            )

        self.assertEqual(yaml_config.format, RUNTIME_CONFIG_FORMAT)
        self.assertEqual(yaml_config.pack_dir, PACK_DIR.resolve())
        self.assertEqual(
            yaml_config.trace.out,
            (EXAMPLE_CONFIG.parent / "../artifacts/deterministic-trace.jsonl").resolve(),
        )
        self.assertEqual(json_config.components.policy.plugin, yaml_config.components.policy.plugin)

    def test_safe_default_mode_is_simulation(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = copy_example_config(Path(td))
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            del data["execution"]["mode"]
            config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

            config = load_runtime_config(config_path)

        self.assertEqual(config.execution.mode, ExecutionMode.SIMULATION)

    def test_reports_required_and_unknown_fields_with_precise_paths(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = copy_example_config(Path(td))
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            data["format"] = "wrong"
            del data["components"]["scenario"]
            data["components"]["policy"]["typo"] = True
            data["unexpected"] = {}
            config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

            with self.assertRaises(RuntimeConfigError) as raised:
                load_runtime_config(config_path)

        paths = {issue.path for issue in raised.exception.issues}
        self.assertIn("format", paths)
        self.assertIn("components.scenario", paths)
        self.assertIn("components.policy.typo", paths)
        self.assertIn("unexpected", paths)

    def test_static_inspection_does_not_import_selected_entrypoints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            write_static_pack(pack, entrypoint="module_that_must_not_import:Component")
            config_path = write_runtime_config(root / "runtime.yaml", pack=pack)

            inspected = inspect_runtime_config(config_path)

        self.assertTrue(inspected["valid"], inspected)
        self.assertEqual(inspected["components"]["backend"]["plugin"], "backend.test")
        self.assertEqual(
            inspected["compatibility"]["common_command_spaces"],
            ["franka.cartesian_delta_gripper.v0"],
        )

    def test_selected_plugin_kind_is_checked_before_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            write_static_pack(pack, entrypoint="missing_heavy_runtime:Component")
            config_path = write_runtime_config(root / "runtime.yaml", pack=pack)
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            data["components"]["policy"]["plugin"] = "backend.test"
            config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected policy"):
                inspect_runtime_config(config_path)

    def test_static_inspection_rejects_safety_command_space_outside_component_contract(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = copy_example_config(Path(td))
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            data["execution"]["safety"]["command_space"] = "franka.undeclared.v0"
            config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

            inspected = inspect_runtime_config(config_path)

        self.assertFalse(inspected["valid"])
        self.assertTrue(
            any("safety command space" in issue for issue in inspected["issues"]),
            inspected,
        )

    def test_capture_configuration_resolves_profile_and_is_visible_to_inspection(self):
        config = load_runtime_config(PI05_CONFIG)
        inspected = inspect_runtime_config(PI05_CONFIG)

        self.assertIsNotNone(config.capture)
        self.assertEqual(config.capture.profile, PI05_CAPTURE_PROFILE.resolve())
        self.assertEqual(config.capture.storage, "summary")
        self.assertEqual(
            inspected["capture"]["profile"],
            str(PI05_CAPTURE_PROFILE.resolve()),
        )
        self.assertEqual(inspected["capture"]["storage"], "summary")

    def test_runtime_binds_capture_to_selected_policy_before_policy_load(self):
        calls = []
        policy = CaptureAwareDeterministicPolicy(calls)
        instances = {
            "robotprobe.policies.franka_deterministic": policy,
            "robotprobe.embodiments.franka": FrankaEmbodimentAdapter(),
            "robotprobe.backends.franka_deterministic": DeterministicFrankaBackend(),
            "robotprobe.scenarios.franka_deterministic_reach": DeterministicReachScenario(
                max_steps=8,
                target_position=(0.45, 0.0, 0.3),
                tolerance=1e-8,
            ),
        }
        with tempfile.TemporaryDirectory() as td:
            config_path = copy_example_config(Path(td))
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            data["capture"] = {
                "profile": str(PI05_CAPTURE_PROFILE),
                "storage": "summary",
            }
            config_path.write_text(
                yaml.safe_dump(data, sort_keys=False),
                encoding="utf-8",
            )

            with patch.object(
                RuntimePluginRegistry,
                "instantiate",
                side_effect=lambda plugin_id, _kind, **_kwargs: instances[plugin_id],
            ):
                result = run_runtime_config(config_path)
            events = [
                json.loads(line)
                for line in Path(result["trace_path"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["profile_path"], PI05_CAPTURE_PROFILE.resolve())
        self.assertEqual(calls[0]["storage"], "summary")
        self.assertFalse(calls[0]["policy_was_loaded"])
        activations = [
            event for event in events if event["event_type"] == "model_activation"
        ]
        self.assertGreater(len(activations), 0)
        action_chunks = {
            event["chunk_id"]: event
            for event in events
            if event["event_type"] == "action_chunk"
        }
        self.assertTrue(
            all(event["action_chunk_id"] in action_chunks for event in activations)
        )
        self.assertTrue(
            all(
                event["policy_call_id"]
                == action_chunks[event["action_chunk_id"]]["policy_call_id"]
                for event in activations
            )
        )

    def test_runtime_rejects_capture_when_selected_policy_cannot_instrument(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = copy_example_config(Path(td))
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            data["capture"] = {
                "profile": str(PI05_CAPTURE_PROFILE),
                "storage": "summary",
            }
            config_path.write_text(
                yaml.safe_dump(data, sort_keys=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeConfigError,
                "configurable capture",
            ):
                run_runtime_config(config_path)


class RuntimeCliTests(unittest.TestCase):
    def test_example_pack_validates_and_runtime_inspect_is_static_json(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = copy_example_config(Path(td))
            inspect_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "runtime",
                    "inspect",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            pack_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "pack",
                    "validate",
                    str(PACK_DIR),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(inspect_proc.returncode, 0, inspect_proc.stderr)
        inspected = json.loads(inspect_proc.stdout)
        self.assertTrue(inspected["valid"], inspected)
        self.assertEqual(inspected["execution"]["mode"], "simulation")
        self.assertEqual(pack_proc.returncode, 0, pack_proc.stderr)
        self.assertTrue(json.loads(pack_proc.stdout)["valid"])

    def test_runtime_run_returns_status_trace_count_and_outcome(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            config_path = copy_example_config(directory)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "runtime",
                    "run",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            output = json.loads(proc.stdout)
            trace_path = Path(output["trace_path"])

            self.assertTrue(trace_path.is_file())
            self.assertGreater(output["event_count"], 0)

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(output["status"], "completed")
        self.assertTrue(output["outcome"]["success"])
        self.assertEqual(output["outcome"]["reason"], "scenario_success")
        self.assertTrue(output["trace_valid"])


def copy_example_config(directory):
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    data["pack"] = str(PACK_DIR)
    data["trace"]["out"] = "runtime-output.jsonl"
    config_path = directory / "runtime.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


class CaptureAwareDeterministicPolicy(DeterministicReachPolicy):
    def __init__(self, calls):
        super().__init__(step_size=0.02, action_horizon=2)
        self.calls = calls

    def configure_capture(
        self,
        *,
        profile_path,
        recorder,
        storage,
        output_dir,
        max_bytes_per_run,
    ):
        self.capture_recorder = recorder
        self.calls.append(
            {
                "profile_path": profile_path,
                "recorder": recorder,
                "storage": storage,
                "output_dir": output_dir,
                "max_bytes_per_run": max_bytes_per_run,
                "policy_was_loaded": self._loaded,
            }
        )

    def predict(self, observation, context):
        chunk = super().predict(observation, context)
        self.capture_recorder.event(
            context.env_step,
            "model_activation",
            "model_activation",
            probe_id="test_configured_capture",
            module="test.capture",
            module_kind="test_boundary",
            activation_ref="summary:sha256:test",
            activation_shape=[1, 1],
            activation_dtype="float32",
            model_ref={"model_name": "capture-aware-deterministic"},
            capture={
                "capture_mode": "test",
                "storage_mode": "summary",
                "stored_bytes": 0,
            },
            policy_call_id=context.policy_call_id,
            action_chunk_id=context.action_chunk_id,
            observation_id=context.observation_id,
            module_call_index=0,
            phase="test",
        )
        return chunk


def write_runtime_config(path, *, pack):
    path.write_text(
        yaml.safe_dump(
            {
                "format": RUNTIME_CONFIG_FORMAT,
                "pack": str(pack),
                "components": {
                    role: {"plugin": f"{role}.test"}
                    for role in ("policy", "embodiment", "backend", "scenario")
                },
                "execution": {
                    "mode": "simulation",
                    "n_action_steps": 1,
                    "safety": {
                        "command_space": "franka.cartesian_delta_gripper.v0",
                        "max_abs": [0.1] * 7,
                        "max_observation_age_ms": 100,
                    },
                },
                "trace": {"out": "trace.jsonl"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def write_static_pack(pack, *, entrypoint):
    blocks = []
    for role in ("policy", "embodiment", "backend", "scenario"):
        command_spaces = (
            ""
            if role == "scenario"
            else "        - franka.cartesian_delta_gripper.v0\n"
        )
        blocks.append(
            f"""
  - id: {role}.test
    type: adapter
    version: 0.1.0
    entrypoint: {entrypoint}
    capabilities:
      format: robotprobe.runtime.component.v0
      kind: {role}
      component_id: {role}.test
      command_spaces:
{command_spaces.rstrip()}
      observation_streams:
      runtime_family: static-test
      is_hardware: false
      supports_reset: {"true" if role == "backend" else "false"}
      supports_ground_truth: false
      supports_shadow: false
      can_apply_hardware_commands: false
      requires_reset: {"true" if role == "scenario" else "false"}
      optional_dependencies:
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - supports static runtime configuration testing
      does_not_prove:
        - imports or executes the selected runtime
""".rstrip()
        )
    (pack / "manifest.yaml").write_text(
        "id: robotprobe-pack-static-runtime\n"
        "version: 0.1.0\n"
        "plugins:\n"
        + "\n".join(blocks)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
