import tempfile
import unittest
from pathlib import Path

from robotprobe.runtime.capabilities import (
    RUNTIME_CAPABILITY_FORMAT,
    RuntimeComponentCapabilities,
    RuntimeComponentKind,
    check_runtime_compatibility,
    is_runtime_capability_map,
    validate_runtime_component_capabilities,
)
from robotprobe.runtime.registry import RuntimePluginRegistry
from robotprobe.runtime.types import ExecutionMode


class RuntimeCapabilityTests(unittest.TestCase):
    def test_parses_runtime_component_capabilities(self):
        data = _capability(
            RuntimeComponentKind.BACKEND,
            "backend.sim",
            command_spaces=["franka.cartesian_delta_gripper.v0"],
            runtime_family="deterministic",
            supports_reset=True,
            supports_ground_truth=True,
        )

        capabilities = RuntimeComponentCapabilities.from_dict(data)

        self.assertTrue(is_runtime_capability_map(data))
        self.assertEqual(capabilities.format, RUNTIME_CAPABILITY_FORMAT)
        self.assertEqual(capabilities.kind, RuntimeComponentKind.BACKEND)
        self.assertEqual(capabilities.component_id, "backend.sim")
        self.assertEqual(capabilities.command_spaces, ("franka.cartesian_delta_gripper.v0",))
        self.assertTrue(capabilities.supports_reset)

    def test_validation_requires_format_kind_id_and_command_spaces(self):
        data = {
            "format": "wrong",
            "kind": "unknown",
            "component_id": "",
            "command_spaces": [],
        }

        issues = validate_runtime_component_capabilities(data, plugin_id="broken")
        fields = {issue.field for issue in issues}

        self.assertIn("capabilities.format", fields)
        self.assertIn("capabilities.kind", fields)
        self.assertIn("capabilities.component_id", fields)
        self.assertIn("capabilities.command_spaces", fields)

    def test_hardware_command_capability_requires_hardware_executable_classification(self):
        data = _capability(
            RuntimeComponentKind.BACKEND,
            "backend.hardware",
            command_spaces=["franka.cartesian_delta_gripper.v0"],
            is_hardware=True,
            supports_shadow=True,
            can_apply_hardware_commands=True,
        )

        issues = validate_runtime_component_capabilities(
            data,
            plugin_id="backend.hardware",
            release_classification="hardware_shadow",
            hardware_allowed=False,
        )

        self.assertTrue(any("hardware_executable" in issue.message for issue in issues))
        self.assertTrue(any("hardware_allowed" in issue.message for issue in issues))

    def test_compatibility_requires_common_command_space(self):
        policy = RuntimeComponentCapabilities.from_dict(
            _capability(
                RuntimeComponentKind.POLICY,
                "policy",
                command_spaces=["franka.cartesian_delta_gripper.v0"],
            )
        )
        embodiment = RuntimeComponentCapabilities.from_dict(
            _capability(
                RuntimeComponentKind.EMBODIMENT,
                "embodiment",
                command_spaces=["franka.cartesian_delta_gripper.v0"],
            )
        )
        backend = RuntimeComponentCapabilities.from_dict(
            _capability(
                RuntimeComponentKind.BACKEND,
                "backend",
                command_spaces=["franka.joint_position.v0"],
                supports_reset=True,
            )
        )
        scenario = RuntimeComponentCapabilities.from_dict(
            _capability(
                RuntimeComponentKind.SCENARIO,
                "scenario",
                command_spaces=[],
                requires_reset=True,
            )
        )

        report = check_runtime_compatibility(
            policy=policy,
            embodiment=embodiment,
            backend=backend,
            scenario=scenario,
            execution_mode=ExecutionMode.SIMULATION,
        )

        self.assertFalse(report.valid)
        self.assertEqual(report.common_command_spaces, ())
        self.assertTrue(any("common command space" in issue for issue in report.issues))

    def test_shadow_mode_requires_hardware_shadow_backend(self):
        components = _compatible_capabilities()
        backend = RuntimeComponentCapabilities.from_dict(
            _capability(
                RuntimeComponentKind.BACKEND,
                "backend.sim",
                command_spaces=["franka.cartesian_delta_gripper.v0"],
                supports_reset=True,
            )
        )
        components["backend"] = backend

        report = check_runtime_compatibility(
            **components,
            execution_mode=ExecutionMode.HARDWARE_SHADOW,
        )

        self.assertFalse(report.valid)
        self.assertTrue(any("hardware shadow" in issue for issue in report.issues))

    def test_compatible_components_report_shared_space(self):
        report = check_runtime_compatibility(
            **_compatible_capabilities(),
            execution_mode=ExecutionMode.SIMULATION,
        )

        self.assertTrue(report.valid, report.issues)
        self.assertEqual(
            report.common_command_spaces,
            ("franka.cartesian_delta_gripper.v0",),
        )


class RuntimePluginRegistryTests(unittest.TestCase):
    def test_resolves_and_instantiates_pack_local_entrypoint_without_switches(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td)
            (pack / "sample_components.py").write_text(
                """
class SampleBackend:
    def __init__(self, label="default"):
        self.label = label
""".strip()
                + "\n",
                encoding="utf-8",
            )
            _write_manifest(
                pack,
                [
                    _manifest_plugin(
                        "backend.sample",
                        "backend",
                        "sample_components:SampleBackend",
                    )
                ],
            )

            registry = RuntimePluginRegistry.from_pack(pack)
            loaded = registry.resolve("backend.sample", RuntimeComponentKind.BACKEND)
            instance = registry.instantiate(
                "backend.sample",
                RuntimeComponentKind.BACKEND,
                options={"label": "swapped"},
            )

        self.assertEqual(loaded.__name__, "SampleBackend")
        self.assertEqual(instance.label, "swapped")

    def test_missing_and_wrong_kind_are_reported_before_import(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td)
            _write_manifest(
                pack,
                [_manifest_plugin("policy.sample", "policy", "missing_module:Sample")],
            )
            registry = RuntimePluginRegistry.from_pack(pack)

            with self.assertRaisesRegex(KeyError, "backend.missing"):
                registry.resolve("backend.missing", RuntimeComponentKind.BACKEND)
            with self.assertRaisesRegex(ValueError, "expected backend"):
                registry.resolve("policy.sample", RuntimeComponentKind.BACKEND)

    def test_duplicate_plugin_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td)
            plugin = _manifest_plugin("backend.sample", "backend", "missing:Sample")
            _write_manifest(pack, [plugin, plugin])

            with self.assertRaisesRegex(ValueError, "duplicate runtime plugin id"):
                RuntimePluginRegistry.from_pack(pack)


def _compatible_capabilities():
    return {
        "policy": RuntimeComponentCapabilities.from_dict(
            _capability(
                RuntimeComponentKind.POLICY,
                "policy",
                command_spaces=["franka.cartesian_delta_gripper.v0"],
            )
        ),
        "embodiment": RuntimeComponentCapabilities.from_dict(
            _capability(
                RuntimeComponentKind.EMBODIMENT,
                "embodiment",
                command_spaces=["franka.cartesian_delta_gripper.v0"],
            )
        ),
        "backend": RuntimeComponentCapabilities.from_dict(
            _capability(
                RuntimeComponentKind.BACKEND,
                "backend",
                command_spaces=["franka.cartesian_delta_gripper.v0"],
                supports_reset=True,
            )
        ),
        "scenario": RuntimeComponentCapabilities.from_dict(
            _capability(
                RuntimeComponentKind.SCENARIO,
                "scenario",
                command_spaces=[],
                requires_reset=True,
            )
        ),
    }


def _capability(kind, component_id, *, command_spaces, **fields):
    return {
        "format": RUNTIME_CAPABILITY_FORMAT,
        "kind": kind.value if isinstance(kind, RuntimeComponentKind) else kind,
        "component_id": component_id,
        "command_spaces": command_spaces,
        "observation_streams": fields.pop("observation_streams", []),
        "runtime_family": fields.pop("runtime_family", "test"),
        "is_hardware": fields.pop("is_hardware", False),
        "supports_reset": fields.pop("supports_reset", False),
        "supports_ground_truth": fields.pop("supports_ground_truth", False),
        "supports_shadow": fields.pop("supports_shadow", False),
        "can_apply_hardware_commands": fields.pop("can_apply_hardware_commands", False),
        "requires_reset": fields.pop("requires_reset", False),
        "optional_dependencies": fields.pop("optional_dependencies", []),
        **fields,
    }


def _manifest_plugin(plugin_id, kind, entrypoint):
    return {
        "id": plugin_id,
        "kind": kind,
        "entrypoint": entrypoint,
    }


def _write_manifest(pack, plugins):
    blocks = []
    for plugin in plugins:
        blocks.append(
            f"""
  - id: {plugin["id"]}
    type: adapter
    version: 0.1.0
    entrypoint: {plugin["entrypoint"]}
    capabilities:
      format: {RUNTIME_CAPABILITY_FORMAT}
      kind: {plugin["kind"]}
      component_id: {plugin["id"]}
      command_spaces:
        - franka.cartesian_delta_gripper.v0
      observation_streams:
        - state
      runtime_family: test
      is_hardware: false
      supports_reset: true
      supports_ground_truth: false
      supports_shadow: false
      can_apply_hardware_commands: false
      requires_reset: false
      optional_dependencies:
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - exercises a test runtime component
      does_not_prove:
        - hardware safety
""".rstrip()
        )
    (pack / "manifest.yaml").write_text(
        "id: robotprobe-pack-runtime-test\n"
        "version: 0.1.0\n"
        "plugins:\n"
        + "\n".join(blocks)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
