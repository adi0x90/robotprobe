import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pi05_libero_franka_smoke.sh"


class Pi05LiberoFrankaSmokeScriptTests(unittest.TestCase):
    def test_runtime_evidence_runs_after_runtime_trace_and_pack_validation(self):
        text = SCRIPT.read_text(encoding="utf-8")

        runtime_run = text.index('CURRENT_STAGE="runtime_run"')
        trace_validation = text.index('CURRENT_STAGE="trace_validation"')
        pack_validation = text.index(
            'uv run robotprobe pack validate "${PACK_DIR}"'
        )
        evidence = text.index("uv run robotprobe runtime evidence")
        evidence_check = text.index('CURRENT_STAGE="trace_evidence_check"')

        self.assertLess(runtime_run, trace_validation)
        self.assertLess(trace_validation, pack_validation)
        self.assertLess(pack_validation, evidence)
        self.assertLess(evidence, evidence_check)
        self.assertIn(
            'RUNTIME_EVIDENCE_DIR="${OUT_DIR}/runtime-evidence"',
            text,
        )
        self.assertIn('--out "${RUNTIME_EVIDENCE_DIR}"', text)

    def test_smoke_summary_references_the_runtime_evidence_manifest(self):
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"runtime_evidence_manifest"', text)
        self.assertIn('"runtime-evidence/manifest.json"', text)

    def test_darwin_preflight_is_skipped_without_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_uname = fake_bin / "uname"
            fake_uname.write_text(
                "#!/usr/bin/env bash\nprintf 'Darwin\\n'\n",
                encoding="utf-8",
            )
            fake_uname.chmod(0o755)
            output_dir = root / "out"
            environment = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "ROBOTPROBE_SMOKE_OUT": str(output_dir),
            }

            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            state = json.loads(
                (output_dir / "evidence-state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["status"], "skipped")
            self.assertEqual(state["reason"], "missing_prerequisite:Linux")
            self.assertFalse((output_dir / "runtime-evidence").exists())


if __name__ == "__main__":
    unittest.main()
