import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from robotprobe.mech.capabilities import (
    MechAdapterCapabilities,
    validate_mech_adapter_capabilities,
)
from robotprobe.packs import build_claim_card, list_pack, validate_pack
from robotprobe.plugins import load_plugin_manifest


ROOT = Path(__file__).resolve().parents[1]
OPENVLA_PACK = ROOT / "examples" / "packs" / "mechprobe-openvla-libero"


class MechCapabilitiesTests(unittest.TestCase):
    def test_loads_mech_capabilities_from_pack_manifest(self):
        manifest = load_plugin_manifest(OPENVLA_PACK / "manifest.yaml")
        plugin = manifest.plugins[0]

        capabilities = MechAdapterCapabilities.from_plugin(plugin)
        validation = validate_pack(OPENVLA_PACK)
        listing = list_pack(OPENVLA_PACK)
        claim_card = build_claim_card(OPENVLA_PACK)

        self.assertEqual(capabilities.format, "robotprobe.mech.adapter_capabilities.v0")
        self.assertEqual(capabilities.model_family, "openvla")
        self.assertFalse(capabilities.supports_activation_capture)
        self.assertFalse(capabilities.supports_temporal_patch)
        self.assertFalse(capabilities.imports_heavy_runtime)
        self.assertEqual(capabilities.safe_default_mode, "offline_only")
        self.assertTrue(validation.valid, validation.to_dict())
        self.assertFalse(listing["plugins"][0]["capabilities"]["supports_activation_capture"])
        self.assertIn("supports_activation_capture=false", claim_card)

    def test_stub_adapter_does_not_advertise_runtime_capabilities(self):
        spec = importlib.util.spec_from_file_location(
            "openvla_capability_test", OPENVLA_PACK / "adapters" / "openvla_activation_adapter.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.OpenVLAActivationAdapter().capture_pair(None, None, None, None)
        capabilities = list_pack(OPENVLA_PACK)["plugins"][0]["capabilities"]

        self.assertEqual(result["status"], "not_implemented")
        for flag, value in capabilities.items():
            if flag.startswith("supports_"):
                self.assertFalse(value, flag)

    def test_capability_validator_rejects_non_boolean_support_flags(self):
        issues = validate_mech_adapter_capabilities(
            {
                "format": "robotprobe.mech.adapter_capabilities.v0",
                "adapter_id": "bad",
                "model_family": "openvla",
                "runtime": "torch",
                "supports_activation_capture": "yes",
                "supports_temporal_patch": False,
                "supports_action_head": True,
                "supports_uncertainty": False,
                "supports_policy_replay": True,
                "supports_sim_replay": True,
                "supports_hardware_shadow": False,
                "requires_white_box_model": True,
                "imports_heavy_runtime": True,
                "safe_default_mode": "capture_only",
            }
        )

        self.assertTrue(any(issue.field == "capabilities.supports_activation_capture" for issue in issues))

    def test_pack_validation_reports_invalid_mech_capability_maps(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            pack.mkdir()
            (pack / "manifest.yaml").write_text(
                """
id: robotprobe-pack-bad-mech
version: 0.1.0
plugins:
  - id: robotprobe.mech.adapters.bad
    type: adapter
    version: 0.1.0
    capabilities:
      format: robotprobe.mech.adapter_capabilities.v0
      adapter_id: bad
      model_family: openvla
      runtime: torch
      supports_activation_capture: true
      supports_temporal_patch: maybe
      safe_default_mode: capture_only
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - declares a bad mech adapter
      does_not_prove:
        - validity
""".strip()
                + "\n",
                encoding="utf-8",
            )

            report = validate_pack(pack)

        self.assertFalse(report.valid)
        self.assertTrue(
            any(issue.field == "capabilities.supports_temporal_patch" for issue in report.issues),
            json.dumps(report.to_dict(), indent=2),
        )


if __name__ == "__main__":
    unittest.main()
