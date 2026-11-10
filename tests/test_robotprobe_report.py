import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.report import build_incident_report


ROOT = Path(__file__).resolve().parents[1]


class RobotProbeReportTests(unittest.TestCase):
    def test_build_incident_report_renders_root_cause_and_evidence(self):
        summary = {
            "root_cause_hypothesis": {
                "layer": "action_postprocess",
                "contract": "ActionSpaceContract",
                "fault": "gripper_freeze_mid",
                "rule_match_score": 0.87,
            },
            "evidence": {
                "first_divergence_step": 35,
                "raw_policy_action_changed": False,
                "postprocessed_action_changed": True,
                "instruction_changed": False,
                "observation_replayed": False,
                "outcome_delta": {
                    "success_changed": False,
                    "step_inflation": 1.89,
                },
            },
            "replay_level": "L0_TRACE_PLAYBACK",
        }
        check = {"passed": True, "failures": [], "rule_match_score": 0.87}

        report = build_incident_report(summary, check=check, title="Gripper freeze")

        self.assertIn("# Gripper freeze", report)
        self.assertIn("Layer: `action_postprocess`", report)
        self.assertIn("Contract: `ActionSpaceContract`", report)
        self.assertIn("Fault: `gripper_freeze_mid`", report)
        self.assertIn("First divergence step: `35`", report)
        self.assertIn("Step inflation: `1.89x`", report)
        self.assertIn("RobotSec check: `PASS`", report)

    def test_build_incident_report_renders_metric_deltas_when_present(self):
        summary = {
            "root_cause_hypothesis": {
                "layer": "action_postprocess",
                "contract": "ActionSpaceContract",
                "fault": "gripper_freeze_mid",
                "rule_match_score": 0.87,
            },
            "evidence": {
                "first_divergence_step": 1,
                "outcome_delta": {"success_changed": False, "step_inflation": 1.5},
            },
            "replay_level": "L0_TRACE_PLAYBACK",
            "metrics": {
                "delta": {
                    "near_miss_score": 0.42,
                    "trajectory_deviation": 0.12,
                }
            },
        }

        report = build_incident_report(summary, title="Metric incident")

        self.assertIn("## Metrics", report)
        self.assertIn("delta.near_miss_score: `0.4200`", report)
        self.assertIn("delta.trajectory_deviation: `0.1200`", report)

    def test_robotprobe_report_cli_writes_markdown(self):
        summary = {
            "root_cause_hypothesis": {
                "layer": "instruction",
                "contract": "InstructionIntentContract",
                "fault": "instruction_swap",
                "rule_match_score": 0.87,
            },
            "evidence": {
                "first_divergence_step": 0,
                "instruction_changed": True,
                "observation_replayed": False,
                "raw_policy_action_changed": True,
                "postprocessed_action_changed": True,
                "outcome_delta": {"success_changed": False, "step_inflation": 0.96},
            },
            "replay_level": "L0_TRACE_PLAYBACK",
        }
        check = {"passed": True, "failures": [], "rule_match_score": 0.87}
        with tempfile.TemporaryDirectory() as td:
            summary_path = Path(td) / "summary.json"
            check_path = Path(td) / "check.json"
            out_path = Path(td) / "report.md"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            check_path.write_text(json.dumps(check), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "report",
                    "--summary",
                    str(summary_path),
                    "--check",
                    str(check_path),
                    "--title",
                    "Instruction incident",
                    "--out",
                    str(out_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            output = out_path.read_text(encoding="utf-8")

        self.assertEqual(proc.stdout, "")
        self.assertIn("Instruction incident", output)


if __name__ == "__main__":
    unittest.main()
