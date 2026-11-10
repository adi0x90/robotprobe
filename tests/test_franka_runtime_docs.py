import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from robotprobe.packs import validate_pack
from robotprobe.plugins import load_plugin_manifest
from robotprobe.runtime.config import inspect_runtime_config


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "examples" / "packs" / "franka-pi05-runtime"
SCRIPT = ROOT / "scripts" / "run_pi05_libero_franka_smoke.sh"
REQUIRED_DOCS = (
    ROOT / "docs" / "franka.md",
    ROOT / "docs" / "runtime.md",
)
CONFIGS = (
    PACK / "configs" / "deterministic.yaml",
    PACK / "configs" / "franka-recorded-replay.yaml",
    PACK / "configs" / "pi05-libero.yaml",
    PACK / "configs" / "franka-ros2-shadow.yaml",
)
PI05_CONFIGS = (
    PACK / "configs" / "pi05-libero.yaml",
    PACK / "configs" / "franka-ros2-shadow.yaml",
)
PI05_REVISION = "dbf8a3f794a9c4297b44f40b752712f50073d945"
PI05_TOKENIZER_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"


class FrankaRuntimeDocumentationTests(unittest.TestCase):
    def test_required_guides_script_and_readme_links_exist(self):
        for path in (*REQUIRED_DOCS, SCRIPT):
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for path in REQUIRED_DOCS:
            expected = f"https://github.com/adi0x90/robotprobe/blob/main/docs/{path.name}"
            with self.subTest(link=expected):
                self.assertIn(f"]({expected})", readme)

    def test_new_markdown_local_links_resolve(self):
        for document in REQUIRED_DOCS:
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                resolved = (document.parent / target.split("#", 1)[0]).resolve()
                with self.subTest(document=document.name, target=target):
                    self.assertTrue(resolved.exists(), resolved)

    def test_documented_commands_reference_real_repository_paths(self):
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "README.md", *REQUIRED_DOCS)
        )
        required_paths = (
            "examples/demos/franka/run_demos.py",
            "examples/packs/franka-pi05-runtime/configs/deterministic.yaml",
            "examples/packs/franka-pi05-runtime/configs/franka-recorded-replay.yaml",
            "examples/packs/franka-pi05-runtime/configs/pi05-libero.yaml",
            "examples/packs/franka-pi05-runtime/configs/franka-ros2-shadow.yaml",
            "examples/packs/franka-pi05-runtime/capture/pi05-semantic.yaml",
            "scripts/run_pi05_libero_franka_smoke.sh",
        )
        for relative in required_paths:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).exists())
                self.assertIn(relative, combined)

    def test_demo_docs_define_three_audiences_and_five_evidence_boundaries(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        demo_guide = (ROOT / "docs" / "franka.md").read_text(
            encoding="utf-8"
        )
        combined = f"{readme}\n{demo_guide}".lower()

        self.assertIn("choose your franka demo", readme.lower())
        for demo_name in ("safety gate", "backend swap", "action journey"):
            with self.subTest(demo_name=demo_name):
                self.assertIn(demo_name, combined)
        for phrase in (
            "deterministic kinematic contract",
            "synthetic recorded-format fixture",
            "libero simulation",
            "physical observation only",
            "physical actuation is not implemented",
            "robotprobe runtime evidence",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, combined)

    def test_pack_and_all_example_configs_validate_statically(self):
        pack_report = validate_pack(PACK)

        self.assertTrue(pack_report.valid, pack_report.issues)
        for config in CONFIGS:
            with self.subTest(config=config.name):
                inspected = inspect_runtime_config(config)
                self.assertTrue(inspected["valid"], inspected["issues"])

    def test_pi05_configs_pin_the_same_full_checkpoint_revision(self):
        for config in PI05_CONFIGS:
            with self.subTest(config=config.name):
                data = yaml.safe_load(config.read_text(encoding="utf-8"))
                revision = data["components"]["policy"]["options"].get("revision")
                self.assertEqual(revision, PI05_REVISION)
                self.assertRegex(revision, r"^[0-9a-f]{40}$")
                tokenizer_revision = data["components"]["policy"]["options"].get(
                    "tokenizer_revision"
                )
                self.assertEqual(tokenizer_revision, PI05_TOKENIZER_REVISION)
                self.assertRegex(tokenizer_revision, r"^[0-9a-f]{40}$")

        validation = (ROOT / "docs" / "franka.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(PI05_REVISION, validation)
        self.assertIn(PI05_TOKENIZER_REVISION, validation)

    def test_franka_pack_contains_no_hardware_executable_component(self):
        manifest = load_plugin_manifest(PACK / "manifest.yaml")

        for plugin in manifest.plugins:
            with self.subTest(plugin=plugin.id):
                self.assertNotEqual(
                    plugin.release_classification,
                    "hardware_executable",
                )
                self.assertFalse(plugin.safety["hardware_allowed"])
                capabilities = plugin.capabilities
                if capabilities.get("is_hardware") is True:
                    self.assertTrue(capabilities["supports_shadow"])
                    self.assertFalse(
                        capabilities["can_apply_hardware_commands"]
                    )

    def test_recorded_replay_plugin_is_simulation_only_and_synthetic(self):
        manifest = load_plugin_manifest(PACK / "manifest.yaml")

        plugin = next(
            item
            for item in manifest.plugins
            if item.id == "robotprobe.backends.franka_recorded_replay"
        )

        self.assertEqual(plugin.release_classification, "sim_only")
        self.assertFalse(plugin.capabilities["is_hardware"])
        self.assertFalse(plugin.capabilities["can_apply_hardware_commands"])
        boundaries = " ".join(plugin.claims["does_not_prove"]).lower()
        self.assertIn("synthetic", boundaries)
        self.assertIn("physical", boundaries)

    def test_guides_preserve_safety_and_evidence_claim_boundaries(self):
        runtime = (ROOT / "docs" / "runtime.md").read_text(
            encoding="utf-8"
        ).lower()
        migration = (
            ROOT / "docs" / "franka.md"
        ).read_text(encoding="utf-8").lower()
        validation = (ROOT / "docs" / "franka.md").read_text(
            encoding="utf-8"
        ).lower()
        safety = (ROOT / "docs" / "safety.md").read_text(
            encoding="utf-8"
        ).lower()

        self.assertIn("deterministic kinematic contract", runtime)
        self.assertIn("not a physics simulator", runtime)
        self.assertIn("does not prove checkpoint task success", runtime)
        self.assertIn("command_applied: false", migration)
        self.assertIn("action normalization", migration)
        self.assertIn("camera calibration", migration)
        self.assertIn("new hardware-executable plugin", migration)
        self.assertIn("completed", validation)
        self.assertIn("failed", validation)
        self.assertIn("skipped", validation)
        self.assertIn("does not prove physical franka safety", validation)
        self.assertIn("hardware-shadow", safety)
        self.assertIn("cannot publish", safety)

    def test_smoke_script_is_strict_bounded_and_records_evidence_state(self):
        text = SCRIPT.read_text(encoding="utf-8")
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        self.assertTrue(text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n"))
        self.assertIn('MUJOCO_GL="${MUJOCO_GL:-egl}"', text)
        self.assertRegex(
            text,
            r'MAX_STEPS="\$\{ROBOTPROBE_LIBERO_MAX_STEPS:-([0-9]+)\}"',
        )
        default_steps = int(
            re.search(
                r'MAX_STEPS="\$\{ROBOTPROBE_LIBERO_MAX_STEPS:-([0-9]+)\}"',
                text,
            ).group(1)
        )
        self.assertLessEqual(default_steps, 20)
        for token in (
            "Linux",
            "torch.cuda.is_available",
            "lerobot.envs.configs",
            "libero",
            "mujoco",
            "df -Pk",
            "validate-trace",
            "pack validate",
            "model_activation",
            "checkpoint_revision",
            "tokenizer_revision",
            '"completed"',
            '"failed"',
            '"skipped"',
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

    def test_smoke_script_rejects_unsafe_suite_with_valid_json_receipt(self):
        with tempfile.TemporaryDirectory() as output_dir:
            environment = {
                **os.environ,
                "ROBOTPROBE_SMOKE_OUT": output_dir,
                "ROBOTPROBE_LIBERO_SUITE": 'bad"suite\nvalue',
            }
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            state = json.loads(
                (Path(output_dir) / "evidence-state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["reason"], "invalid_suite")


if __name__ == "__main__":
    unittest.main()
