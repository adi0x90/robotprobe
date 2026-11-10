import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotsec_range.challenge_spec import ChallengeSpec, load_builtin_challenge
from robotsec_range.mech_checker import check_mech_report


class MechRobotSecChallengeTests(unittest.TestCase):
    def test_mech_checker_accepts_observed_internal_divergence_without_feature_promotion(self):
        challenge = ChallengeSpec(
            id="mech-instruction-leadtime",
            diagnosis={},
            required_evidence={
                "min_evidence_level": 1,
                "internal_divergence_precedes_action_divergence": True,
                "min_internal_lead_time": 4,
                "model_root_cause_claim_allowed": False,
                "feature_promotions_allowed": False,
            },
        )
        result = check_mech_report(challenge, _observed_report())

        self.assertTrue(result.passed, result.failures)
        self.assertEqual(result.rule_match_score, 0.0)

    def test_mech_checker_rejects_low_evidence_report(self):
        challenge = ChallengeSpec(
            id="mech-instruction-leadtime",
            diagnosis={},
            required_evidence={"min_evidence_level": 2},
        )
        report = _observed_report()
        report["evidence_level"] = 1

        result = check_mech_report(challenge, report)

        self.assertFalse(result.passed)
        self.assertIn("mech.evidence_level", result.failures[0])

    def test_cli_check_mech_uses_builtin_challenge(self):
        challenge = load_builtin_challenge("robotprobe-mech-v0-instruction-internal-leadtime")
        self.assertEqual(challenge.required_evidence["min_evidence_level"], 1)
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "mech_report.json"
            report_path.write_text(json.dumps(_observed_report()), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotsec_range",
                    "check-mech",
                    "--challenge",
                    "robotprobe-mech-v0-instruction-internal-leadtime",
                    "--report",
                    str(report_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout)

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(result["passed"], result)


def _observed_report() -> dict:
    return {
        "format": "robotprobe.mech.report.v0",
        "synthetic": False,
        "evidence_eligible": True,
        "comparison_complete": True,
        "evidence_level": 1,
        "model_root_cause_claim_allowed": False,
        "activation_layers": [
            {
                "module": "policy.backbone.layers.18",
                "activation_diff_significant": True,
                "first_significant_divergence_step": 10,
                "external_first_divergence_step": 14,
                "lead_time_steps": 4,
            }
        ],
        "feature_diff": {
            "features": [
                {
                    "feature_id": "target_object_direction",
                    "label": "candidate: target-object selection",
                    "evidence_level": 1,
                    "promotion_allowed": False,
                }
            ]
        },
    }


if __name__ == "__main__":
    unittest.main()
