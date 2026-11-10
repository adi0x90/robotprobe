import unittest
from pathlib import Path

from robotprobe.mech.adapters.base import WhiteBoxActivationAdapter
from robotprobe.packs import test_pack, validate_pack
from robotprobe.plugins import load_plugin_entrypoint, load_plugin_manifest


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "examples" / "packs" / "mechprobe-openvla-libero"


class MechOpenVLAPackAdapterTests(unittest.TestCase):
    def test_pack_validates_without_importing_runtime_code_by_default(self):
        report = validate_pack(PACK)

        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(report.id, "robotprobe-pack-mechprobe-openvla-libero")

    def test_pack_entrypoint_is_loadable_when_code_is_explicitly_allowed(self):
        manifest = load_plugin_manifest(PACK / "manifest.yaml")
        adapter_plugin = manifest.plugins[0]

        adapter_cls = load_plugin_entrypoint(adapter_plugin, search_paths=[PACK])
        adapter = adapter_cls()

        self.assertTrue(issubclass(adapter_cls, WhiteBoxActivationAdapter))
        self.assertEqual(adapter.target, "openvla-libero")

    def test_pack_test_imports_entrypoint_only_with_allow_code(self):
        blocked = test_pack(PACK, allow_code=False)
        allowed = test_pack(PACK, allow_code=True)

        self.assertFalse(blocked.valid)
        self.assertTrue(any(issue.field == "execution.imports_code_during_test" for issue in blocked.issues))
        self.assertTrue(allowed.valid, allowed.to_dict())


if __name__ == "__main__":
    unittest.main()
