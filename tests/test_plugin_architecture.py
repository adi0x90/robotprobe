import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.packs import (
    PACK_TEMPLATES,
    build_claim_card,
    build_pack_registry,
    init_pack,
    list_pack,
    load_pack_registry,
    test_pack,
    validate_pack,
)
from robotprobe.contracts import check_contract_file
from robotprobe.exporters import export_sidecar
from robotprobe.plugins import load_plugin_entrypoint, load_plugin_manifest, validate_plugin_manifest


class PluginManifestTests(unittest.TestCase):
    def test_loads_plugin_entries_and_claim_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "manifest.yaml"
            manifest.write_text(
                """
id: robotprobe-pack-test
version: 0.1.0
plugins:
  - id: robotprobe.contracts.keepout
    type: contract
    version: 0.1.0
    entrypoint: robotprobe_contracts.keepout:KeepoutContract
    files:
      - contracts/keepout.yaml
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - detects keepout violations in recorded traces
      does_not_prove:
        - hardware certification
""".strip()
                + "\n",
                encoding="utf-8",
            )

            loaded = load_plugin_manifest(manifest)

        self.assertEqual(loaded.id, "robotprobe-pack-test")
        self.assertEqual(loaded.version, "0.1.0")
        self.assertEqual(len(loaded.plugins), 1)
        plugin = loaded.plugins[0]
        self.assertEqual(plugin.id, "robotprobe.contracts.keepout")
        self.assertEqual(plugin.type, "contract")
        self.assertEqual(plugin.release_classification, "sim_only")
        self.assertFalse(plugin.hardware_allowed)
        self.assertEqual(plugin.files, ["contracts/keepout.yaml"])
        self.assertIn("detects keepout", plugin.claims["proves"][0])

    def test_loads_pack_trust_metadata_and_manifest_digest(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "manifest.yaml"
            manifest.write_text(
                """
id: robotprobe-pack-trusted
version: 0.2.1
trust_level: trusted_core_pack
publisher:
  name: robotprobe
  url: https://github.com/adi0x90/robotprobe
compatibility:
  robotprobe_min: 0.2.1
  schema_version: robotprobe.pack.v0.2
execution:
  imports_code_during_validate: false
  imports_code_during_test: true
  network_access_required: false
  filesystem_write_paths:
    - $PACK_OUT
plugins:
  - id: robotprobe.contracts.keepout
    type: contract
    version: 0.1.0
    entrypoint: robotprobe_contracts.keepout:KeepoutContract
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - detects keepout violations in recorded traces
      does_not_prove:
        - hardware certification
""".strip()
                + "\n",
                encoding="utf-8",
            )

            loaded = load_plugin_manifest(manifest)

        self.assertEqual(loaded.trust_level, "trusted_core_pack")
        self.assertEqual(loaded.publisher["name"], "robotprobe")
        self.assertEqual(loaded.compatibility["robotprobe_min"], "0.2.1")
        self.assertEqual(loaded.execution["imports_code_during_test"], True)
        self.assertEqual(loaded.execution["filesystem_write_paths"], ["$PACK_OUT"])
        self.assertTrue(loaded.manifest_sha256.startswith("sha256:"))

    def test_rejects_hardware_executable_plugins_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "manifest.yaml"
            manifest.write_text(
                """
id: robotprobe-pack-hardware
version: 0.1.0
plugins:
  - id: robotprobe.interventions.real_motor_fault
    type: intervention
    version: 0.1.0
    safety:
      release_classification: hardware_executable
      hardware_allowed: true
    claims:
      proves:
        - mutates a real hardware command
      does_not_prove:
        - safe public execution
""".strip()
                + "\n",
                encoding="utf-8",
            )
            loaded = load_plugin_manifest(manifest)

        report = validate_plugin_manifest(loaded)
        self.assertFalse(report.valid)
        self.assertTrue(any("hardware_executable" in issue.message for issue in report.issues))

        allowed_report = validate_plugin_manifest(loaded, allow_hardware_executable=True)
        self.assertTrue(allowed_report.valid, json.dumps(allowed_report.to_dict(), indent=2))


class PluginEntrypointTests(unittest.TestCase):
    def test_loads_pack_local_python_entrypoint(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            pack.mkdir()
            (pack / "sample_pack.py").write_text(
                """
class SampleContract:
    plugin_id = "robotprobe.contracts.sample"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            (pack / "manifest.yaml").write_text(
                """
id: robotprobe-pack-sample
version: 0.1.0
plugins:
  - id: robotprobe.contracts.sample
    type: contract
    version: 0.1.0
    entrypoint: sample_pack:SampleContract
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - loads a local sample contract
      does_not_prove:
        - production safety
""".strip()
                + "\n",
                encoding="utf-8",
            )
            plugin = load_plugin_manifest(pack / "manifest.yaml").plugins[0]

            loaded = load_plugin_entrypoint(plugin, search_paths=[pack])

        self.assertEqual(loaded.plugin_id, "robotprobe.contracts.sample")


class PackValidationTests(unittest.TestCase):
    def test_validates_pack_declared_files_and_challenges(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            (pack / "contracts").mkdir(parents=True)
            (pack / "challenges").mkdir()
            (pack / "contracts" / "keepout.yaml").write_text("id: keepout\n", encoding="utf-8")
            (pack / "challenges" / "keepout.json").write_text('{"id": "keepout"}\n', encoding="utf-8")
            (pack / "manifest.yaml").write_text(
                """
id: robotprobe-pack-keepout
version: 0.1.0
challenges:
  - challenges/keepout.json
plugins:
  - id: robotprobe.contracts.keepout
    type: contract
    version: 0.1.0
    files:
      - contracts/keepout.yaml
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - detects keepout violations
      does_not_prove:
        - hardware certification
""".strip()
                + "\n",
                encoding="utf-8",
            )

            report = validate_pack(pack)
            listing = list_pack(pack)

        self.assertTrue(report.valid, json.dumps(report.to_dict(), indent=2))
        self.assertEqual(report.plugin_count, 1)
        self.assertEqual(listing["id"], "robotprobe-pack-keepout")
        self.assertEqual(listing["plugins"][0]["id"], "robotprobe.contracts.keepout")
        self.assertEqual(listing["challenges"], ["challenges/keepout.json"])

    def test_invalid_pack_reports_missing_declared_files(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            pack.mkdir()
            (pack / "manifest.yaml").write_text(
                """
id: robotprobe-pack-broken
version: 0.1.0
challenges:
  - challenges/missing.json
plugins:
  - id: robotprobe.contracts.keepout
    type: contract
    version: 0.1.0
    files:
      - contracts/missing.yaml
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - detects keepout violations
      does_not_prove:
        - hardware certification
""".strip()
                + "\n",
                encoding="utf-8",
            )

            report = validate_pack(pack)

        self.assertFalse(report.valid)
        messages = "\n".join(issue.message for issue in report.issues)
        self.assertIn("missing declared file", messages)
        self.assertIn("missing challenge file", messages)

    def test_builds_pack_registry_from_pack_roots(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            init_pack(
                pack,
                pack_id="robotprobe-pack-registry-demo",
                plugin_id="robotprobe.contracts.registry_demo",
                plugin_type="contract",
            )
            registry_path = root / "registry.json"

            registry = build_pack_registry([pack], out_path=registry_path)
            loaded = load_pack_registry(registry_path)

        self.assertEqual(registry["format"], "robotprobe.pack_registry.v0")
        self.assertEqual(registry["packs"][0]["id"], "robotprobe-pack-registry-demo")
        self.assertTrue(registry["packs"][0]["valid"])
        self.assertEqual(loaded["packs"][0]["plugins"][0]["id"], "robotprobe.contracts.registry_demo")

    def test_cli_builds_and_lists_pack_registry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            init_pack(
                pack,
                pack_id="robotprobe-pack-cli-registry",
                plugin_id="robotprobe.adapters.cli_registry",
                plugin_type="adapter",
            )
            registry_path = root / "registry.json"

            build_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "pack",
                    "registry",
                    "build",
                    str(pack),
                    "--out",
                    str(registry_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            list_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "pack",
                    "registry",
                    "list",
                    str(registry_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(build_proc.returncode, 0, build_proc.stderr)
        self.assertEqual(list_proc.returncode, 0, list_proc.stderr)
        listed = json.loads(list_proc.stdout)
        self.assertEqual(listed["packs"][0]["id"], "robotprobe-pack-cli-registry")
        self.assertEqual(listed["packs"][0]["plugin_count"], 1)

    def test_scaffolds_builtin_adapter_pack_templates(self):
        expected_files = {
            "ros2-command-authority": "contracts/command_authority.yaml",
            "vla-eval-adapter": "adapters/vla_eval.md",
            "isaac-lab-arena-adapter": "adapters/isaac_lab_arena.md",
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for template, expected_file in expected_files.items():
                pack = root / template
                result = init_pack(
                    pack,
                    pack_id=f"robotprobe-pack-{template}",
                    template=template,
                )
                report = validate_pack(pack)

                self.assertEqual(result["template"], template)
                self.assertTrue((pack / expected_file).exists(), template)
                self.assertTrue(report.valid, json.dumps(report.to_dict(), indent=2))

        self.assertIn("ros2-command-authority", PACK_TEMPLATES)

    def test_cli_scaffolds_builtin_adapter_pack_template(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "ros2-authority"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "pack",
                    "init",
                    str(pack),
                    "--id",
                    "robotprobe-pack-ros2-authority",
                    "--template",
                    "ros2-command-authority",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((pack / "contracts" / "command_authority.yaml").exists())
            manifest = (pack / "manifest.yaml").read_text(encoding="utf-8")
            readme = (pack / "docs" / "README.md").read_text(encoding="utf-8")

            self.assertIn("robotprobe.adapters.ros2_command_authority", manifest)
            self.assertIn("release_classification: trace_only", manifest)
            self.assertIn("trace_only", readme)

    def test_cli_requires_plugin_fields_without_template(self):
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "pack",
                    "init",
                    str(Path(td) / "broken"),
                    "--id",
                    "robotprobe-pack-broken",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(proc.returncode, 2)
        self.assertIn("--plugin-id and --plugin-type are required without --template", proc.stderr)

    def test_pack_test_requires_allow_code_for_manifest_that_declares_test_imports(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            pack.mkdir()
            (pack / "sample_pack.py").write_text(
                """
class SampleContract:
    plugin_id = "robotprobe.contracts.sample"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            (pack / "manifest.yaml").write_text(
                """
id: robotprobe-pack-code-gated
version: 0.2.1
trust_level: untrusted_local_pack
execution:
  imports_code_during_validate: false
  imports_code_during_test: true
  network_access_required: false
plugins:
  - id: robotprobe.contracts.sample
    type: contract
    version: 0.1.0
    entrypoint: sample_pack:SampleContract
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - loads a sample contract
      does_not_prove:
        - production safety
""".strip()
                + "\n",
                encoding="utf-8",
            )

            blocked = test_pack(pack)
            allowed = test_pack(pack, allow_code=True)

        self.assertFalse(blocked.valid)
        self.assertIn("--allow-code", "\n".join(issue.message for issue in blocked.issues))
        self.assertTrue(allowed.valid, json.dumps(allowed.to_dict(), indent=2))

    def test_pack_claim_card_renders_claims_and_non_claims(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            init_pack(
                pack,
                pack_id="robotprobe-pack-claims",
                plugin_id="robotprobe.contracts.claims",
                plugin_type="contract",
            )

            card = build_claim_card(pack)

        self.assertIn("# Claim Card: robotprobe-pack-claims", card)
        self.assertIn("This pack demonstrates", card)
        self.assertIn("This pack does not demonstrate", card)
        self.assertIn("hardware certification", card)

    def test_cli_pack_claims_outputs_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            init_pack(
                pack,
                pack_id="robotprobe-pack-cli-claims",
                plugin_id="robotprobe.contracts.claims",
                plugin_type="contract",
            )

            proc = subprocess.run(
                [sys.executable, "-m", "robotprobe", "pack", "claims", str(pack)],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("# Claim Card: robotprobe-pack-cli-claims", proc.stdout)


class ContractDslTests(unittest.TestCase):
    def test_threshold_contract_flags_active_requirement_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "run_id": "run1",
                                "target": "sim",
                                "scenario": "contract",
                                "replay_level": "L2_SIM_REPLAY",
                                "step": 0,
                                "event_type": "outcome",
                                "layer": "outcome",
                                "success": True,
                                "steps": 1,
                                "min_keepout_margin": -0.05,
                                "end_effector_speed": 0.4,
                            }
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            contract = root / "contract.yaml"
            contract.write_text(
                """
id: keepout_margin_speed
type: threshold_contract
when:
  field: min_keepout_margin
  lt: 0
require:
  field: end_effector_speed
  lte: 0.25
violation:
  contract: KeepoutSpeedContract
  check: speed_inside_keepout
  severity: 0.9
""".strip()
                + "\n",
                encoding="utf-8",
            )

            result = check_contract_file(contract, trace)

        self.assertFalse(result["passed"])
        self.assertEqual(result["violations"][0]["contract"], "KeepoutSpeedContract")
        self.assertEqual(result["violations"][0]["check"], "speed_inside_keepout")
        self.assertEqual(result["violations"][0]["step"], 0)

    def test_temporal_contract_flags_forbidden_state_inside_step_window(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "run_id": "run1",
                            "target": "sim",
                            "scenario": "temporal-contract",
                            "replay_level": "L2_SIM_REPLAY",
                            "step": step,
                            "event_type": "command",
                            "layer": "action_postprocess",
                            "command_authority": {"source": source},
                        }
                    )
                    for step, source in [(0, "policy"), (12, "policy"), (13, "bypass_proxy"), (22, "policy")]
                )
                + "\n",
                encoding="utf-8",
            )
            contract = root / "temporal.yaml"
            contract.write_text(
                """
id: command_authority_window
type: temporal_contract
window:
  start_step: 10
  end_step: 20
must_never:
  field: command_authority.source
  eq: bypass_proxy
violation:
  contract: CommandAuthorityContract
  check: bypass_proxy_inside_supervised_window
  severity: 0.95
""".strip()
                + "\n",
                encoding="utf-8",
            )

            result = check_contract_file(contract, trace)

        self.assertFalse(result["passed"])
        self.assertEqual(result["contract_type"], "temporal_contract")
        self.assertEqual(result["violations"][0]["contract"], "CommandAuthorityContract")
        self.assertEqual(result["violations"][0]["check"], "bypass_proxy_inside_supervised_window")
        self.assertEqual(result["violations"][0]["step"], 13)
        self.assertEqual(result["violations"][0]["temporal_operator"], "must_never")

    def test_temporal_contract_requires_eventual_recovery_inside_step_window(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "run_id": "run1",
                            "target": "sim",
                            "scenario": "temporal-contract",
                            "replay_level": "L2_SIM_REPLAY",
                            "step": step,
                            "event_type": "safety_state",
                            "layer": "runtime_monitor",
                            "recovered": recovered,
                        }
                    )
                    for step, recovered in [(0, False), (8, False), (16, False), (24, False)]
                )
                + "\n",
                encoding="utf-8",
            )
            contract = root / "eventual.yaml"
            contract.write_text(
                """
id: recovery_required
type: temporal_contract
window:
  start_step: 8
  end_step: 24
must_eventually:
  field: recovered
  eq: true
violation:
  contract: RecoveryContract
  check: no_recovery_inside_window
  severity: 0.8
""".strip()
                + "\n",
                encoding="utf-8",
            )

            result = check_contract_file(contract, trace)

        self.assertFalse(result["passed"])
        self.assertEqual(result["violations"][0]["contract"], "RecoveryContract")
        self.assertEqual(result["violations"][0]["temporal_operator"], "must_eventually")
        self.assertEqual(result["violations"][0]["window"], {"start_step": 8, "end_step": 24})

    def test_threshold_contract_passes_when_condition_inactive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "trace.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "run_id": "run1",
                        "target": "sim",
                        "scenario": "contract",
                        "replay_level": "L2_SIM_REPLAY",
                        "step": 0,
                        "event_type": "outcome",
                        "layer": "outcome",
                        "success": True,
                        "steps": 1,
                        "min_keepout_margin": 0.2,
                        "end_effector_speed": 0.4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            contract = root / "contract.yaml"
            contract.write_text(
                """
id: keepout_margin_speed
type: threshold_contract
when:
  field: min_keepout_margin
  lt: 0
require:
  field: end_effector_speed
  lte: 0.25
violation:
  contract: KeepoutSpeedContract
  check: speed_inside_keepout
  severity: 0.9
""".strip()
                + "\n",
                encoding="utf-8",
            )

            result = check_contract_file(contract, trace)

        self.assertTrue(result["passed"], json.dumps(result, indent=2))
        self.assertEqual(result["violations"], [])


class SidecarExporterTests(unittest.TestCase):
    def test_exports_incident_bundle_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = Path(td) / "incident"
            bundle_dir.mkdir()
            (bundle_dir / "bundle.json").write_text(
                json.dumps(
                    {
                        "summary_ref": "summary.json",
                        "check_ref": "check.json",
                        "report_ref": "report.md",
                        "clean_trace_ref": "traces/clean.jsonl",
                        "perturbed_trace_ref": "traces/fault.jsonl",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (bundle_dir / "summary.json").write_text(
                json.dumps({"root_cause_hypothesis": {"layer": "action_postprocess"}}) + "\n",
                encoding="utf-8",
            )
            (bundle_dir / "check.json").write_text(
                json.dumps({"passed": True}) + "\n",
                encoding="utf-8",
            )

            out = bundle_dir / "robotprobe.sidecar.json"
            result = export_sidecar(bundle_dir, out)
            written = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(result["format"], "robotprobe.sidecar.v0")
        self.assertEqual(written["bundle"]["report_ref"], "report.md")
        self.assertEqual(written["summary"]["root_cause_hypothesis"]["layer"], "action_postprocess")
        self.assertTrue(written["check"]["passed"])


class ExamplePackTests(unittest.TestCase):
    def test_keepout_safety_example_pack_validates(self):
        pack = Path(__file__).resolve().parents[1] / "examples" / "packs" / "keepout-safety"

        report = validate_pack(pack)
        listing = list_pack(pack)

        self.assertTrue(report.valid, json.dumps(report.to_dict(), indent=2))
        self.assertEqual(listing["id"], "robotprobe-pack-keepout-safety")
        plugin_ids = {plugin["id"] for plugin in listing["plugins"]}
        self.assertIn("robotprobe.contracts.keepout_margin", plugin_ids)
        self.assertIn("robotprobe.interventions.stealth_velocity_bias_keepout", plugin_ids)
        self.assertIn("challenges/robotprobe-v1-security-keepout-bypass.json", listing["challenges"])

    def test_adapter_example_packs_validate(self):
        root = Path(__file__).resolve().parents[1] / "examples" / "packs"
        expected = {
            "ros2-command-authority": "robotprobe.adapters.ros2_command_authority",
            "vla-eval-adapter": "robotprobe.adapters.vla_eval",
            "isaac-lab-arena-adapter": "robotprobe.adapters.isaac_lab_arena",
        }

        for pack_name, plugin_id in expected.items():
            pack = root / pack_name
            report = validate_pack(pack)
            listing = list_pack(pack)

            self.assertTrue(report.valid, json.dumps(report.to_dict(), indent=2))
            self.assertIn(plugin_id, {plugin["id"] for plugin in listing["plugins"]})


class GoldenPackTests(unittest.TestCase):
    def test_pack_test_reports_golden_fixture_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            (pack / "fixtures").mkdir(parents=True)
            (pack / "golden").mkdir()
            (pack / "fixtures" / "summary.json").write_text('{"passed": true}\n', encoding="utf-8")
            (pack / "golden" / "summary.json").write_text('{"passed": false}\n', encoding="utf-8")
            (pack / "manifest.yaml").write_text(
                """
id: robotprobe-pack-golden
version: 0.1.0
plugins:
  - id: robotprobe.contracts.golden
    type: contract
    version: 0.1.0
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - compares golden fixtures
      does_not_prove:
        - runtime safety
""".strip()
                + "\n",
                encoding="utf-8",
            )

            report = test_pack(pack)

        self.assertFalse(report.valid)
        self.assertTrue(any("golden mismatch" in issue.message for issue in report.issues))

    def test_example_pack_golden_fixtures_pass(self):
        pack = Path(__file__).resolve().parents[1] / "examples" / "packs" / "keepout-safety"

        report = test_pack(pack)

        self.assertTrue(report.valid, json.dumps(report.to_dict(), indent=2))


class PackInitTests(unittest.TestCase):
    def test_init_pack_scaffolds_valid_sim_only_pack(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "new-pack"

            result = init_pack(
                pack,
                pack_id="robotprobe-pack-demo",
                plugin_id="robotprobe.contracts.demo",
                plugin_type="contract",
            )
            report = validate_pack(pack)
            listing = list_pack(pack)

            self.assertEqual(result["id"], "robotprobe-pack-demo")
            self.assertTrue((pack / "manifest.yaml").exists())
            self.assertTrue((pack / "docs" / "README.md").exists())
            self.assertTrue(report.valid, json.dumps(report.to_dict(), indent=2))
            self.assertEqual(listing["plugins"][0]["release_classification"], "sim_only")
            self.assertFalse(listing["plugins"][0]["hardware_allowed"])


if __name__ == "__main__":
    unittest.main()
