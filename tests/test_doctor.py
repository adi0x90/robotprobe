import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.doctor import run_doctor
from robotprobe.packs import init_pack


class DoctorTests(unittest.TestCase):
    def test_doctor_reports_core_environment_checks(self):
        result = run_doctor()
        check_names = {check["name"] for check in result["checks"]}

        self.assertEqual(result["format"], "robotprobe.doctor.v0")
        self.assertIn("python_version", check_names)
        self.assertIn("robotprobe_import", check_names)
        self.assertIn("robotsec_range_import", check_names)
        self.assertIn("mcap_import", check_names)
        self.assertIn("hardware_executable_blocked_by_default", check_names)

    def test_doctor_can_validate_a_pack(self):
        with tempfile.TemporaryDirectory() as td:
            pack = Path(td) / "pack"
            init_pack(
                pack,
                pack_id="robotprobe-pack-doctor",
                plugin_id="robotprobe.contracts.doctor",
                plugin_type="contract",
            )

            result = run_doctor(pack_path=pack)
            pack_checks = [check for check in result["checks"] if check["name"] == "pack_valid"]

        self.assertEqual(pack_checks[0]["status"], "pass")
        self.assertEqual(pack_checks[0]["details"]["id"], "robotprobe-pack-doctor")

    def test_cli_doctor_outputs_json(self):
        proc = subprocess.run(
            [sys.executable, "-m", "robotprobe", "doctor"],
            text=True,
            capture_output=True,
            check=False,
        )
        result = json.loads(proc.stdout)

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["format"], "robotprobe.doctor.v0")
        self.assertTrue(any(check["name"] == "python_version" for check in result["checks"]))
