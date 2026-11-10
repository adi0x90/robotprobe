import json
import tempfile
import unittest
from pathlib import Path

import yaml

from robotprobe.runtime.config import run_runtime_config
from robotprobe.runtime.evidence import analyze_runtime_trace


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "examples" / "packs" / "franka-pi05-runtime"
REPLAY_CONFIG = PACK / "configs" / "franka-recorded-replay.yaml"


class FrankaRecordedReplayRuntimeTests(unittest.TestCase):
    def test_configured_replay_runs_end_to_end_without_applying_commands(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = copy_replay_config(root)

            result = run_runtime_config(config_path)
            trace_path = Path(result["trace_path"])
            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            receipts = [
                event
                for event in events
                if event["event_type"] == "execution_receipt"
            ]
            evidence = analyze_runtime_trace(
                trace_path,
                config_path=config_path,
            )

        self.assertEqual(
            result["components"]["backend"]["component_id"],
            "franka.recorded_snapshot_replay.v0",
        )
        self.assertTrue(result["trace_valid"])
        self.assertFalse(result["outcome"]["success"])
        self.assertEqual(result["outcome"]["reason"], "max_steps_exhausted")
        self.assertTrue(receipts)
        self.assertTrue(
            all(not event["command_applied"] for event in receipts)
        )
        self.assertFalse(
            any(event["event_type"] == "runtime_error" for event in events)
        )
        self.assertEqual(evidence["runtime_integrity"]["status"], "passed")
        self.assertEqual(
            evidence["motion"]["disposition"],
            "evaluated_not_applied",
        )


def copy_replay_config(root: Path) -> Path:
    data = yaml.safe_load(REPLAY_CONFIG.read_text(encoding="utf-8"))
    data["pack"] = str(PACK)
    data["trace"]["out"] = str(root / "replay.jsonl")
    config_path = root / "replay.yaml"
    config_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


if __name__ == "__main__":
    unittest.main()
