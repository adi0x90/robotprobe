import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from robotprobe.runtime.config import run_runtime_config


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "examples" / "packs" / "franka-pi05-runtime"
BASE_CONFIG = PACK / "configs" / "deterministic.yaml"


class RuntimeEvidenceCliTests(unittest.TestCase):
    def test_cli_writes_paired_evidence_for_expected_safety_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            safe_config, safe_trace = write_runtime_config(
                root,
                name="safe",
                step_size=0.02,
            )
            rejected_config, rejected_trace = write_runtime_config(
                root,
                name="rejected",
                step_size=0.2,
            )
            self.assertTrue(run_runtime_config(safe_config)["outcome"]["success"])
            self.assertFalse(
                run_runtime_config(rejected_config)["outcome"]["success"]
            )
            out = root / "evidence"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "runtime",
                    "evidence",
                    "--trace",
                    str(rejected_trace),
                    "--baseline",
                    str(safe_trace),
                    "--config",
                    str(rejected_config),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            manifest = json.loads(completed.stdout)
            evidence = json.loads(
                (out / "evidence.json").read_text(encoding="utf-8")
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            manifest["format"],
            "robotprobe.runtime.evidence_bundle.v0",
        )
        self.assertEqual(manifest["runtime_integrity"], "passed")
        self.assertEqual(
            evidence["motion"]["disposition"],
            "rejected_pre_backend",
        )
        self.assertEqual(evidence["motion"]["execution_receipts"], 0)

    def test_cli_writes_diagnostic_bundle_and_exits_one_for_broken_lineage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, trace = write_runtime_config(
                root,
                name="broken",
                step_size=0.02,
            )
            run_runtime_config(config)
            events = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            receipt = next(
                event
                for event in events
                if event["event_type"] == "execution_receipt"
            )
            receipt["command_id"] = "command:unknown"
            trace.write_text(
                "".join(
                    json.dumps(event, sort_keys=True) + "\n"
                    for event in events
                ),
                encoding="utf-8",
            )
            out = root / "evidence"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "runtime",
                    "evidence",
                    "--trace",
                    str(trace),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            manifest = json.loads(completed.stdout)
            evidence = json.loads(
                (out / "evidence.json").read_text(encoding="utf-8")
            )

        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(manifest["runtime_integrity"], "failed")
        self.assertEqual(evidence["runtime_integrity"]["status"], "failed")
        self.assertIn(
            "unknown_authorized_command",
            {
                issue["code"]
                for issue in evidence["runtime_integrity"]["issues"]
            },
        )

    def test_cli_writes_diagnostic_bundle_for_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = root / "malformed.jsonl"
            trace.write_text('{"run_id": "unfinished"\n', encoding="utf-8")
            out = root / "evidence"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "runtime",
                    "evidence",
                    "--trace",
                    str(trace),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            manifest = json.loads(completed.stdout)
            evidence = json.loads(
                (out / "evidence.json").read_text(encoding="utf-8")
            )

        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertEqual(manifest["runtime_integrity"], "failed")
        self.assertIn(
            "trace_schema_error",
            {
                issue["code"]
                for issue in evidence["runtime_integrity"]["issues"]
            },
        )


def write_runtime_config(
    root: Path,
    *,
    name: str,
    step_size: float,
) -> tuple[Path, Path]:
    data = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    trace = root / f"{name}.jsonl"
    data["pack"] = str(PACK)
    data["components"]["policy"]["options"]["step_size"] = step_size
    data["components"]["scenario"]["options"]["target_position"] = [
        0.5,
        0.0,
        0.3,
    ]
    data["trace"]["out"] = str(trace)
    data["trace"]["run_id"] = f"runtime:cli:{name}"
    config = root / f"{name}.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config, trace


if __name__ == "__main__":
    unittest.main()
