import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.integrations.lerobot.pi05_probes import (
    PI05SemanticActivationCapture,
    Pi05CaptureConfigurationError,
    discover_pi05_semantic_probes,
    load_pi05_capture_settings,
)
from robotprobe.runtime.types import InferenceContext


class Pi05SemanticProbeDiscoveryTests(unittest.TestCase):
    def test_discovers_early_middle_late_model_families_and_projections(self):
        model = FakeNamedModel(complete_pi05_module_names())

        report = discover_pi05_semantic_probes(model)
        probes = {probe.role: probe for probe in report.probes}

        self.assertEqual(report.missing_roles, ())
        self.assertEqual(probes["vision.early"].layer_index, 0)
        self.assertEqual(probes["vision.middle"].layer_index, 13)
        self.assertEqual(probes["vision.late"].layer_index, 26)
        self.assertEqual(probes["language.early"].layer_index, 0)
        self.assertEqual(probes["language.middle"].layer_index, 8)
        self.assertEqual(probes["language.late"].layer_index, 17)
        self.assertEqual(probes["action_expert.early"].layer_index, 0)
        self.assertEqual(probes["action_expert.middle"].layer_index, 8)
        self.assertEqual(probes["action_expert.late"].layer_index, 17)
        self.assertTrue(probes["action_expert.early"].module_name.endswith(".layers.0.mlp"))
        self.assertEqual(probes["action_expert.early"].module_kind, "action_expert_mlp")
        self.assertTrue(probes["action_input_projection"].module_name.endswith("action_in_proj"))
        self.assertTrue(probes["action_output_projection"].module_name.endswith("action_out_proj"))
        self.assertTrue(probes["time_mlp_input"].module_name.endswith("time_mlp_in"))
        self.assertTrue(probes["time_mlp_output"].module_name.endswith("time_mlp_out"))

    def test_reports_missing_semantic_roles_instead_of_guessing(self):
        names = [
            name
            for name in complete_pi05_module_names()
            if "gemma_expert.model.layers" not in name
            and not name.endswith("action_out_proj")
        ]

        report = discover_pi05_semantic_probes(FakeNamedModel(names))

        self.assertIn("action_expert.early", report.missing_roles)
        self.assertIn("action_expert.middle", report.missing_roles)
        self.assertIn("action_expert.late", report.missing_roles)
        self.assertIn("action_output_projection", report.missing_roles)
        self.assertNotIn("vision.early", report.missing_roles)

    def test_user_exact_and_glob_selectors_are_additive_and_missing_is_explicit(self):
        names = complete_pi05_module_names() + [
            "model.custom.guard",
            "model.custom.audit.0",
            "model.custom.audit.1",
        ]

        report = discover_pi05_semantic_probes(
            FakeNamedModel(names),
            user_selectors=(
                "model.custom.guard",
                "model.custom.audit.*",
                "model.custom.missing",
            ),
        )
        user_modules = {
            probe.module_name for probe in report.probes if probe.role.startswith("user.")
        }

        self.assertEqual(
            user_modules,
            {
                "model.custom.guard",
                "model.custom.audit.0",
                "model.custom.audit.1",
            },
        )
        self.assertIn("user:model.custom.missing", report.missing_roles)

    def test_only_declared_denoising_roles_derive_denoise_step_from_call_index(self):
        report = discover_pi05_semantic_probes(
            FakeNamedModel(complete_pi05_module_names())
        )
        probes = {probe.role: probe for probe in report.probes}

        for role in (
            "action_expert.early",
            "action_expert.middle",
            "action_expert.late",
            "action_input_projection",
            "action_output_projection",
            "time_mlp_input",
            "time_mlp_output",
        ):
            with self.subTest(role=role):
                self.assertEqual(probes[role].phase, "denoise")
                self.assertTrue(probes[role].denoise_step_from_module_call)
        for role in (
            "vision.early",
            "vision.middle",
            "vision.late",
            "language.early",
            "language.middle",
            "language.late",
        ):
            with self.subTest(role=role):
                self.assertEqual(probes[role].phase, "prefix")
                self.assertFalse(probes[role].denoise_step_from_module_call)

        profile = report.to_capture_profile()
        by_id = {probe.probe_id: probe for probe in profile.probes}
        self.assertTrue(
            by_id["pi05_action_output_projection"].denoise_step_from_module_call
        )
        self.assertFalse(by_id["pi05_vision_early"].denoise_step_from_module_call)


class Pi05ConfiguredCaptureTests(unittest.TestCase):
    def test_loads_strict_profile_and_applies_runtime_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile_path = write_profile(root)
            output_dir = root / "activations"

            settings = load_pi05_capture_settings(
                profile_path,
                storage_override="npy",
                output_dir=output_dir,
                max_bytes_per_run_override=4096,
            )

        self.assertEqual(settings.profile_id, "test-pi05-capture")
        self.assertEqual(settings.semantic_roles, ("action_output_projection",))
        self.assertEqual(settings.user_selectors, ())
        self.assertEqual(settings.storage.value, "npy")
        self.assertEqual(settings.output_dir, output_dir.resolve())
        self.assertEqual(settings.max_bytes_per_run, 4096)
        self.assertEqual(settings.every_n_policy_calls, 1)

    def test_profile_rejects_unknown_fields_and_full_storage_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            unknown = write_profile(root, extra="unexpected: true\n")
            with self.assertRaisesRegex(Pi05CaptureConfigurationError, "unexpected"):
                load_pi05_capture_settings(unknown)

            valid = write_profile(root, filename="valid.yaml")
            with self.assertRaisesRegex(Pi05CaptureConfigurationError, "output_dir"):
                load_pi05_capture_settings(valid, storage_override="safetensors")

    def test_semantic_capture_resolves_selected_role_and_emits_correlated_events(self):
        with tempfile.TemporaryDirectory() as td:
            settings = load_pi05_capture_settings(write_profile(Path(td)))
            recorder = FakeCaptureRecorder()
            capture = PI05SemanticActivationCapture(
                settings,
                recorder=recorder,
                model_ref={"model_name": "fake-pi05", "revision": "test"},
            )
            model = FakeHookModel("model.action_out_proj")
            context = InferenceContext(
                env_step=4,
                policy_call_id="policy:000003",
                action_chunk_id="chunk:000003",
                observation_id="observation:000004",
            )

            with capture.session(model, context):
                model.modules["model.action_out_proj"].run(
                    np.arange(8, dtype=np.float32).reshape(1, 2, 4)
                )
                model.modules["model.action_out_proj"].run(
                    np.ones((1, 2, 4), dtype=np.float32)
                )

        events = [
            event
            for event in recorder.events
            if event["event_type"] == "model_activation"
        ]
        self.assertEqual(len(events), 2)
        self.assertEqual([event["denoise_step"] for event in events], [0, 1])
        self.assertTrue(
            all(event["policy_call_id"] == "policy:000003" for event in events)
        )
        self.assertTrue(
            all(event["action_chunk_id"] == "chunk:000003" for event in events)
        )
        self.assertTrue(
            all(event["observation_id"] == "observation:000004" for event in events)
        )
        self.assertTrue(all(event["step"] == 4 for event in events))
        self.assertTrue(all(event["capture"]["storage_mode"] == "summary" for event in events))
        self.assertTrue(
            all(event["probe_id"] == "pi05_action_output_projection" for event in events)
        )
        self.assertTrue(
            all(
                event["model_ref"]["capture_profile_sha256"].startswith("sha256:")
                for event in events
            )
        )
        self.assertTrue(
            all(
                event["model_ref"]["semantic_roles"]
                == ["action_output_projection"]
                for event in events
            )
        )
        self.assertEqual(model.modules["model.action_out_proj"].hooks, [])

    def test_missing_requested_semantic_role_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as td:
            settings = load_pi05_capture_settings(write_profile(Path(td)))
            capture = PI05SemanticActivationCapture(
                settings,
                recorder=FakeCaptureRecorder(),
                model_ref={"model_name": "fake-pi05"},
            )

            with self.assertRaisesRegex(
                Pi05CaptureConfigurationError,
                "action_output_projection",
            ):
                capture.session(
                    FakeHookModel("model.action_in_proj"),
                    InferenceContext(
                        env_step=0,
                        policy_call_id="policy:000000",
                        action_chunk_id="chunk:000000",
                        observation_id="observation:000000",
                    ),
                )


class FakeNamedModel:
    def __init__(self, names):
        self.names = names

    def named_modules(self):
        return [(name, object()) for name in self.names]


class FakeHookHandle:
    def __init__(self, module, hook):
        self.module = module
        self.hook = hook

    def remove(self):
        if self.hook in self.module.hooks:
            self.module.hooks.remove(self.hook)


class FakeHookModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHookHandle(self, hook)

    def run(self, output):
        for hook in list(self.hooks):
            hook(self, (), output)
        return output


class FakeHookModel:
    def __init__(self, *module_names):
        self.modules = {
            module_name: FakeHookModule() for module_name in module_names
        }

    def named_modules(self):
        return list(self.modules.items())


class FakeCaptureRecorder:
    def __init__(self):
        self.events = []

    def event(self, step, event_type, layer, **fields):
        self.events.append(
            {
                "step": step,
                "event_type": event_type,
                "layer": layer,
                **fields,
            }
        )


def write_profile(root, *, filename="profile.yaml", extra=""):
    path = root / filename
    path.write_text(
        (
            "format: robotprobe.runtime.capture_profile.v0\n"
            "id: test-pi05-capture\n"
            "storage: summary\n"
            "max_bytes_per_run: 2048\n"
            "every_n_policy_calls: 1\n"
            "semantic_roles:\n"
            "  - action_output_projection\n"
            "user_selectors: []\n"
            "claim_boundary:\n"
            "  summaries_are_observational: true\n"
            "  no_causal_claims: true\n"
            f"{extra}"
        ),
        encoding="utf-8",
    )
    return path


def complete_pi05_module_names():
    names = [""]
    names.extend(
        f"model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{index}"
        for index in range(27)
    )
    names.extend(
        f"model.paligemma_with_expert.paligemma.model.language_model.layers.{index}"
        for index in range(18)
    )
    names.extend(
        f"model.paligemma_with_expert.gemma_expert.model.layers.{index}"
        for index in range(18)
    )
    names.extend(
        f"model.paligemma_with_expert.gemma_expert.model.layers.{index}.mlp"
        for index in range(18)
    )
    names.extend(
        (
            "model.action_in_proj",
            "model.time_mlp_in",
            "model.time_mlp_out",
            "model.action_out_proj",
        )
    )
    return names


if __name__ == "__main__":
    unittest.main()
