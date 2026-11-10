import json
import tempfile
import unittest
from pathlib import Path

import yaml

from robotprobe.runtime.config import RuntimeConfigError, run_runtime_config


class RuntimeBackendSwapTests(unittest.TestCase):
    def test_changing_only_backend_plugin_preserves_runner_and_semantic_sequence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            write_swap_module(pack)
            write_swap_manifest(pack)
            config_path = root / "runtime.yaml"
            base = runtime_data(pack, trace_out=root / "trace.jsonl")
            config_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

            first = run_runtime_config(config_path)
            first_events = read_events(Path(first["trace_path"]))

            swapped = json.loads(json.dumps(base))
            swapped["components"]["backend"]["plugin"] = "backend.alternative"
            config_path.write_text(yaml.safe_dump(swapped, sort_keys=False), encoding="utf-8")
            second = run_runtime_config(config_path)
            second_events = read_events(Path(second["trace_path"]))

        self.assertTrue(first["outcome"]["success"])
        self.assertTrue(second["outcome"]["success"])
        self.assertEqual(
            [event["event_type"] for event in first_events],
            [event["event_type"] for event in second_events],
        )
        self.assertEqual(
            [event["policy_call_id"] for event in first_events if "policy_call_id" in event],
            [event["policy_call_id"] for event in second_events if "policy_call_id" in event],
        )
        self.assertEqual(
            [event["component_id"] for event in first["components"].values()],
            [
                "franka.policy.deterministic_reach.v0",
                "franka.canonical.v0",
                "backend.primary",
                "franka.scenario.deterministic_reach.v0",
            ],
        )
        self.assertEqual(second["components"]["backend"]["component_id"], "backend.alternative")

    def test_runtime_rejects_capability_drift_from_static_pack_declaration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            write_swap_module(pack)
            write_swap_manifest(pack)
            manifest_path = pack / "manifest.yaml"
            manifest = manifest_path.read_text(encoding="utf-8")
            primary_start = manifest.index("  - id: backend.primary")
            primary_end = manifest.index("  - id: backend.alternative")
            primary = manifest[primary_start:primary_end].replace(
                "runtime_family: swap-test",
                "runtime_family: false-static-claim",
            )
            manifest_path.write_text(
                manifest[:primary_start] + primary + manifest[primary_end:],
                encoding="utf-8",
            )
            config_path = root / "runtime.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    runtime_data(pack, trace_out=root / "trace.jsonl"),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaises(RuntimeConfigError) as raised:
                run_runtime_config(config_path)

        self.assertIn(
            "components.backend.capabilities",
            {issue.path for issue in raised.exception.issues},
        )


def read_events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def runtime_data(pack, *, trace_out):
    return {
        "format": "robotprobe.runtime.run.v0",
        "pack": str(pack),
        "components": {
            "policy": {
                "plugin": "policy.deterministic",
                "options": {"step_size": 0.02, "action_horizon": 2},
            },
            "embodiment": {"plugin": "embodiment.franka"},
            "backend": {"plugin": "backend.primary"},
            "scenario": {
                "plugin": "scenario.reach",
                "options": {
                    "initial_position": [0.4, 0, 0.3],
                    "target_position": [0.43, 0, 0.3],
                    "tolerance": 1e-8,
                    "max_steps": 5,
                },
            },
        },
        "execution": {
            "mode": "simulation",
            "n_action_steps": 2,
            "safety": {
                "command_space": "franka.cartesian_delta_gripper.v0",
                "max_abs": [0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 1.0],
                "workspace_min": [0.2, -0.5, 0.1],
                "workspace_max": [0.7, 0.5, 0.8],
                "max_observation_age_ms": 1000,
            },
        },
        "trace": {
            "out": str(trace_out),
            "run_id": "run:backend-swap",
            "target": "deterministic-policy",
            "scenario": "deterministic-reach",
        },
    }


def write_swap_module(pack):
    (pack / "swap_components.py").write_text(
        """
from robotprobe.integrations.franka.deterministic import DeterministicFrankaBackend
from robotprobe.runtime.capabilities import RuntimeComponentCapabilities


class PrimaryBackend(DeterministicFrankaBackend):
    capabilities = RuntimeComponentCapabilities.from_dict({
        "format": "robotprobe.runtime.component.v0",
        "kind": "backend",
        "component_id": "backend.primary",
        "command_spaces": ["franka.cartesian_delta_gripper.v0"],
        "observation_streams": ["synthetic_rgb", "eef_pose", "gripper", "joint_state"],
        "runtime_family": "swap-test",
        "is_hardware": False,
        "supports_reset": True,
        "supports_ground_truth": True,
        "supports_shadow": False,
        "can_apply_hardware_commands": False,
        "requires_reset": False,
        "optional_dependencies": [],
    })


class AlternativeBackend(PrimaryBackend):
    capabilities = RuntimeComponentCapabilities.from_dict({
        **PrimaryBackend.capabilities.to_dict(),
        "component_id": "backend.alternative",
    })
""".strip()
        + "\n",
        encoding="utf-8",
    )


def write_swap_manifest(pack):
    plugins = [
        (
            "policy.deterministic",
            "policy",
            "robotprobe.integrations.franka.deterministic:DeterministicReachPolicy",
            "franka.policy.deterministic_reach.v0",
            False,
        ),
        (
            "embodiment.franka",
            "embodiment",
            "robotprobe.integrations.franka.embodiment:FrankaEmbodimentAdapter",
            "franka.canonical.v0",
            False,
        ),
        (
            "backend.primary",
            "backend",
            "swap_components:PrimaryBackend",
            "backend.primary",
            True,
        ),
        (
            "backend.alternative",
            "backend",
            "swap_components:AlternativeBackend",
            "backend.alternative",
            True,
        ),
        (
            "scenario.reach",
            "scenario",
            "robotprobe.integrations.franka.scenarios:DeterministicReachScenario",
            "franka.scenario.deterministic_reach.v0",
            False,
        ),
    ]
    blocks = []
    for plugin_id, kind, entrypoint, component_id, supports_reset in plugins:
        command_space = (
            ""
            if kind == "scenario"
            else "        - franka.cartesian_delta_gripper.v0\n"
        )
        observation_streams = {
            "policy": ("rgb", "eef_pose", "gripper", "joint_state"),
            "embodiment": ("rgb", "eef_pose", "gripper", "joint_state"),
            "backend": ("synthetic_rgb", "eef_pose", "gripper", "joint_state"),
            "scenario": (),
        }[kind]
        observation_block = "".join(
            f"        - {stream}\n" for stream in observation_streams
        )
        runtime_family = {
            "policy": "franka_deterministic",
            "embodiment": "franka",
            "backend": "swap-test",
            "scenario": "franka_deterministic",
        }[kind]
        blocks.append(
            f"""
  - id: {plugin_id}
    type: adapter
    version: 0.1.0
    entrypoint: {entrypoint}
    capabilities:
      format: robotprobe.runtime.component.v0
      kind: {kind}
      component_id: {component_id}
      command_spaces:
{command_space.rstrip()}
      observation_streams:
{observation_block.rstrip()}
      runtime_family: {runtime_family}
      is_hardware: false
      supports_reset: {"true" if supports_reset else "false"}
      supports_ground_truth: {"true" if kind == "backend" else "false"}
      supports_shadow: false
      can_apply_hardware_commands: false
      requires_reset: {"true" if kind == "scenario" else "false"}
      optional_dependencies:
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - supplies one swappable runtime component
      does_not_prove:
        - physical Franka execution
""".rstrip()
        )
    (pack / "manifest.yaml").write_text(
        "id: robotprobe-pack-backend-swap\n"
        "version: 0.1.0\n"
        "plugins:\n"
        + "\n".join(blocks)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
