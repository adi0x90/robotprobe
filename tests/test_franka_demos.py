import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from examples.demos.franka import run_demos


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "demos" / "franka" / "run_demos.py"


class FrankaDemoSuiteTests(unittest.TestCase):
    def test_demo_run_guard_rejects_invalid_trace_or_evidence(self):
        passing_evidence = {
            "runtime_integrity": {"status": "passed"},
        }
        failed_evidence = {
            "runtime_integrity": {"status": "failed"},
        }

        with self.assertRaisesRegex(RuntimeError, "trace validation"):
            run_demos._require_run_integrity(
                "invalid-trace",
                {"trace_valid": False},
                passing_evidence,
            )
        with self.assertRaisesRegex(RuntimeError, "runtime integrity"):
            run_demos._require_run_integrity(
                "invalid-evidence",
                {"trace_valid": True},
                failed_evidence,
            )

    def test_safety_gate_demo_proves_pre_backend_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "franka-demos"

            completed = run_demo("safety-gate", out)

            suite = load_json(out / "suite.json")
            candidate = load_json(
                out
                / "runs"
                / "rejected"
                / "evidence"
                / "evidence.json"
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            suite["demos"]["safety-gate"]["status"],
            "completed",
        )
        self.assertEqual(
            candidate["motion"]["disposition"],
            "rejected_pre_backend",
        )
        self.assertEqual(candidate["motion"]["execution_receipts"], 0)
        self.assertEqual(candidate["motion"]["applied_commands"], 0)
        self.assertEqual(
            candidate["task_outcome"]["reason"],
            "safety_rejected:action_limit_exceeded",
        )

    def test_backend_swap_demo_executes_local_backends_and_bounds_optional_ones(
        self,
    ):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "franka-demos"

            completed = run_demo("backend-swap", out)

            matrix = load_json(out / "demos" / "backend-matrix.json")
            deterministic = yaml.safe_load(
                (
                    out / "runs" / "safe" / "runtime.yaml"
                ).read_text(encoding="utf-8")
            )
            replay = yaml.safe_load(
                (
                    out / "runs" / "replay" / "runtime.yaml"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(matrix["deterministic"]["status"], "completed")
        self.assertEqual(matrix["recorded_replay"]["status"], "completed")
        self.assertEqual(matrix["pi05_libero"]["status"], "inspect_only")
        self.assertEqual(
            matrix["franka_ros2_shadow"]["status"],
            "inspect_only",
        )
        self.assertEqual(matrix["recorded_replay"]["applied_commands"], 0)
        self.assertTrue(matrix["all_static_configs_valid"])
        for role in ("policy", "embodiment", "scenario"):
            with self.subTest(role=role):
                self.assertEqual(
                    deterministic["components"][role]["plugin"],
                    replay["components"][role]["plugin"],
                )
        self.assertNotEqual(
            deterministic["components"]["backend"]["plugin"],
            replay["components"]["backend"]["plugin"],
        )

    def test_action_journey_renders_the_complete_runtime_story(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "franka-demos"

            completed = run_demo("action-journey", out)

            report = (
                out / "demos" / "action-journey.md"
            ).read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for token in (
            "instruction",
            "observation",
            "policy_inference",
            "action_chunk",
            "policy:000000",
            "chunk:000000",
            "chunk:000000:action:000",
            "safety_decision",
            "execution_receipt",
            "measured state",
            "outcome",
            "evidence boundary",
        ):
            with self.subTest(token=token):
                self.assertIn(token, report.lower())

    def test_all_reuses_three_runs_for_three_demo_views(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "franka-demos"

            completed = run_demo("all", out)

            suite = load_json(out / "suite.json")
            run_names = sorted(
                path.name
                for path in (out / "runs").iterdir()
                if path.is_dir()
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            sorted(suite["demos"]),
            ["action-journey", "backend-swap", "safety-gate"],
        )
        self.assertEqual(run_names, ["rejected", "replay", "safe"])

    def test_demo_refuses_non_empty_output_without_removing_user_files(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "franka-demos"
            out.mkdir()
            unrelated = out / "unrelated.txt"
            unrelated.write_text("preserve me", encoding="utf-8")

            completed = run_demo("safety-gate", out)

            preserved = unrelated.read_text(encoding="utf-8")

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(preserved, "preserve me")
        self.assertIn("must be empty", completed.stderr)


def run_demo(name: str, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--demo",
            name,
            "--out",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
