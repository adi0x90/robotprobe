import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotsec_range.challenge_spec import load_builtin_challenge, list_builtin_challenges
from robotsec_range.checker import check_summary
from robotprobe.summarize import summarize_trace_pair


ROOT = Path(__file__).resolve().parents[1]


class RobotSecChallengeTests(unittest.TestCase):
    def test_builtin_challenges_include_v0_external_fault_layers(self):
        challenge_ids = [challenge.id for challenge in list_builtin_challenges()]

        self.assertIn("robotprobe-v0-instruction-swap", challenge_ids)
        self.assertIn("robotprobe-v0-stale-observation", challenge_ids)
        self.assertIn("robotprobe-v0-gripper-freeze", challenge_ids)
        self.assertIn("robotprobe-v0-step-inflation", challenge_ids)
        self.assertIn("robotprobe-v1-action-scale", challenge_ids)
        self.assertIn("robotprobe-v1-action-chunk-boundary", challenge_ids)
        self.assertIn("robotprobe-v1-libero-gripper-realtrace", challenge_ids)
        self.assertIn("robotprobe-v1-libero-stale-observation-realtrace", challenge_ids)
        self.assertIn("robotprobe-v1-libero-instruction-realtrace", challenge_ids)
        self.assertIn("robotprobe-v1-libero-near-miss-metrics", challenge_ids)

    def test_load_builtin_challenge_returns_required_diagnosis(self):
        challenge = load_builtin_challenge("robotprobe-v0-stale-observation")

        self.assertEqual(challenge.diagnosis["layer"], "observation")
        self.assertEqual(challenge.diagnosis["contract"], "SensorFreshnessContract")
        self.assertEqual(challenge.required_evidence["observation_replayed"], True)

    def test_robotsec_range_list_challenges_cli_outputs_json_ids(self):
        proc = subprocess.run(
            [sys.executable, "-m", "robotsec_range", "list-challenges"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        data = json.loads(proc.stdout)
        self.assertIn("robotprobe-v0-gripper-freeze", [item["id"] for item in data])

    def test_robotsec_range_check_accepts_trace_verified_builtin_challenge(self):
        challenge = load_builtin_challenge("robotprobe-v1-action-scale")
        refs = challenge.metadata["fixture_refs"]
        manifest = (
            ROOT
            / "examples"
            / "packs"
            / "libero-action-contracts"
            / "interventions"
            / "z_scale_mismatch.yaml"
        )
        summary = summarize_trace_pair(
            ROOT / refs["clean_trace"],
            ROOT / refs["perturbed_trace"],
            intervention_manifest_ref=manifest,
        )
        with tempfile.TemporaryDirectory() as td:
            summary_path = Path(td) / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotsec_range",
                    "check",
                    "--challenge",
                    challenge.id,
                    "--summary",
                    str(summary_path),
                    "--clean-trace",
                    str(ROOT / refs["clean_trace"]),
                    "--perturbed-trace",
                    str(ROOT / refs["perturbed_trace"]),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

        result = json.loads(proc.stdout)
        self.assertTrue(result["passed"], result)

    def test_robotsec_range_check_rejects_forged_summary_without_traces(self):
        summary = {
            "root_cause_hypothesis": {
                "layer": "action_postprocess",
                "contract": "KeepoutZoneContract",
                "fault": "stealth_velocity_bias_keepout",
            },
            "evidence": {
                "postprocessed_action_changed": True,
                "raw_policy_action_changed": False,
                "contract_violated": True,
                "perturbed_keepout_violated": True,
                "hidden_success_safety_violation": True,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            summary_path = Path(td) / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotsec_range",
                    "check",
                    "--challenge",
                    "robotprobe-v1-security-keepout-bypass",
                    "--summary",
                    str(summary_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        result = json.loads(proc.stdout)
        self.assertEqual(proc.returncode, 1)
        self.assertFalse(result["passed"])
        self.assertIn("evidence.clean_trace: required for independent recomputation", result["failures"])

    def test_check_summary_rejects_summary_that_disagrees_with_traces(self):
        challenge = load_builtin_challenge("robotprobe-v1-action-scale")
        refs = challenge.metadata["fixture_refs"]
        manifest = (
            ROOT
            / "examples"
            / "packs"
            / "libero-action-contracts"
            / "interventions"
            / "z_scale_mismatch.yaml"
        )
        summary = summarize_trace_pair(
            ROOT / refs["clean_trace"],
            ROOT / refs["perturbed_trace"],
            intervention_manifest_ref=manifest,
        )
        summary["evidence"]["postprocessed_action_changed"] = False

        result = check_summary(
            challenge,
            summary,
            clean_trace=ROOT / refs["clean_trace"],
            perturbed_trace=ROOT / refs["perturbed_trace"],
            intervention_manifest_ref=manifest,
        )

        self.assertFalse(result.passed)
        self.assertTrue(
            any("submitted False, but traces compute True" in failure for failure in result.failures),
            result.failures,
        )

    def test_v1_challenge_pack_includes_fixtures_and_golden_diagnosis(self):
        challenge = load_builtin_challenge("robotprobe-v1-action-scale")

        self.assertEqual(challenge.diagnosis["layer"], "action_postprocess")
        self.assertEqual(challenge.required_evidence["postprocessed_action_changed"], True)
        self.assertIn("fixture_refs", challenge.metadata)
        self.assertIn("golden_summary", challenge.metadata["fixture_refs"])
        golden = ROOT / challenge.metadata["fixture_refs"]["golden_summary"]
        self.assertTrue(golden.exists())
        self.assertEqual(
            json.loads(golden.read_text(encoding="utf-8"))["root_cause_hypothesis"]["fault"],
            "action_scale_mismatch",
        )

    def test_real_libero_trace_challenges_include_trace_fixtures(self):
        challenge = load_builtin_challenge("robotprobe-v1-libero-gripper-realtrace")

        refs = challenge.metadata["fixture_refs"]
        for key in ("clean_trace", "perturbed_trace", "golden_summary"):
            self.assertTrue((ROOT / refs[key]).exists(), key)
        self.assertEqual(challenge.metadata["source"]["benchmark"], "LIBERO Spatial")
        self.assertEqual(challenge.metadata["source"]["policy"], "Pi0.5")

    def test_real_libero_trace_metrics_pass_near_miss_challenge(self):
        challenge = load_builtin_challenge("robotprobe-v1-libero-near-miss-metrics")
        refs = challenge.metadata["fixture_refs"]

        summary = summarize_trace_pair(
            ROOT / refs["clean_trace"],
            ROOT / refs["perturbed_trace"],
            intervention_manifest_ref=ROOT / "interventions" / "gripper_freeze_mid.yaml",
            metrics=["action_energy", "jerk", "trajectory_deviation", "recovery_latency", "near_miss"],
        )

        self.assertEqual(summary["root_cause_hypothesis"]["layer"], "action_postprocess")
        self.assertGreater(summary["metrics"]["delta"]["near_miss_score"], 0.5)
        self.assertGreater(summary["metrics"]["delta"]["trajectory_deviation"], 0.0)
        self.assertTrue(check_summary(challenge, summary).passed)


if __name__ == "__main__":
    unittest.main()
