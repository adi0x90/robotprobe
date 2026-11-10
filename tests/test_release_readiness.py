import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseReadinessTests(unittest.TestCase):
    def _doc(self, relative: str) -> str:
        path = ROOT / "docs" / relative
        self.assertTrue(path.exists(), f"required doc missing: docs/{relative}")
        return path.read_text(encoding="utf-8")

    def test_license_claims_and_non_claims_are_present(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        claims = self._doc("status-and-claims.md")

        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0", license_text)
        self.assertIn("## Claims", claims)
        self.assertIn("## Non-Claims", claims)
        self.assertIn("not a robotics simulator", claims)
        self.assertIn("not evidence of broad model robustness", claims)
        self.assertIn("## Evidence", claims)

    def test_validation_notes_state_what_was_not_established(self):
        notes = self._doc("status-and-claims.md")
        self.assertIn("What these runs do not establish", notes)
        self.assertIn("did not run", notes)

    def test_safety_policy_and_intervention_manifests_are_sim_only(self):
        safety = self._doc("safety.md")

        self.assertIn("Simulation-first", safety)
        self.assertIn("hardware_allowed: false", safety)
        self.assertIn("No hardware-destructive intervention is shipped", safety)

        for manifest in (ROOT / "interventions").glob("*.yaml"):
            text = manifest.read_text(encoding="utf-8")
            self.assertIn("release_classification: sim_only", text, manifest.name)
            self.assertIn("sim_only: true", text, manifest.name)
            self.assertIn("hardware_allowed: false", text, manifest.name)

    def test_demo_command_writes_a_complete_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "demo-out"
            proc = subprocess.run(
                [sys.executable, "-m", "robotprobe", "demo", "--out", str(out_dir)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            for name in ("bundle.json", "summary.json", "causal_diff.json", "check.json", "report.md"):
                self.assertTrue((out_dir / name).exists(), name)
            check = json.loads((out_dir / "check.json").read_text(encoding="utf-8"))
            self.assertTrue(check["passed"], check)

    def test_mechprobe_docs_state_white_box_and_evidence_boundaries(self):
        mechprobe = self._doc("mechprobe.md")
        claims = self._doc("status-and-claims.md")

        self.assertIn("optional white-box mechanistic-forensics layer", mechprobe)
        self.assertIn("MechProbe is disabled by default", mechprobe)
        self.assertIn("does not work on closed model-server-only systems", claims)
        self.assertIn("findings below evidence Level 3", claims)


if __name__ == "__main__":
    unittest.main()
