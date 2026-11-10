import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.conformance import SPEC_ID, check_conformance_file
from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "interventions" / "gripper_freeze_mid.yaml"


def _check(report, requirement):
    matches = [c for c in report.checks if c.requirement == requirement]
    assert matches, f"no check named {requirement}"
    return matches[0].passed


class FailureSemanticsConformanceTests(unittest.TestCase):
    def test_clean_baseline_with_seed_and_outcome_reaches_l1_not_l2(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "clean.jsonl"
            with TraceRecorder(
                trace,
                run_id="rp_run_clean_001",
                target="pi05-libero",
                scenario="clean",
                seed=7,
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(0, "postprocessed_action", "action_postprocess", value=[0.1, 0.2, 0.3])
                recorder.event(20, "outcome", "outcome", success=True, steps=20)

            report = check_conformance_file(trace)

        self.assertEqual(report.spec, SPEC_ID)
        self.assertEqual(report.run_id, "rp_run_clean_001")
        self.assertTrue(_check(report, "run_seed_present"))
        self.assertTrue(_check(report, "provenance_pinned"))  # clean baseline
        self.assertTrue(_check(report, "terminal_outcome_present"))
        self.assertFalse(_check(report, "contract_bound"))
        self.assertEqual(report.achieved_level, "L1_REPRODUCIBLE_INCIDENT")
        self.assertTrue(report.conformant)  # no target -> at least L0

    def test_manifest_pinned_perturbed_with_contract_check_reaches_l2(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "perturbed.jsonl"
            with TraceRecorder(
                trace,
                run_id="rp_run_gripper_001",
                target="pi05-libero",
                scenario="gripper_freeze_mid",
                intervention_manifest_ref=MANIFEST,
                seed=11,
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(35, "postprocessed_action", "action_postprocess", value=[0.1, 0.2, 0.0])
                recorder.contract_check(
                    step=35,
                    contract="ActionSpaceContract",
                    check="gripper_freshness",
                    status="violated",
                    severity=0.7,
                    evidence_refs=["actions/post/000035.npy"],
                )
                recorder.event(153, "outcome", "outcome", success=True, steps=153)

            report = check_conformance_file(trace, target_level="L2_CONTRACT_CHECKABLE")

        self.assertTrue(_check(report, "provenance_pinned"))  # manifest hash
        self.assertTrue(_check(report, "contract_bound"))
        self.assertEqual(report.achieved_level, "L2_CONTRACT_CHECKABLE")
        self.assertTrue(report.conformant)

    def test_missing_seed_caps_at_l0_and_fails_l1_target(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "noseed.jsonl"
            with TraceRecorder(
                trace,
                run_id="rp_run_clean_002",
                target="pi05-libero",
                scenario="clean",
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(20, "outcome", "outcome", success=True, steps=20)

            report = check_conformance_file(trace, target_level="L1_REPRODUCIBLE_INCIDENT")

        self.assertFalse(_check(report, "run_seed_present"))
        self.assertEqual(report.achieved_level, "L0_WELL_FORMED")
        self.assertFalse(report.conformant)

    def test_non_monotonic_steps_fail_l1(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "outoforder.jsonl"
            with TraceRecorder(
                trace,
                run_id="rp_run_clean_003",
                target="pi05-libero",
                scenario="clean",
                seed=3,
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(5, "postprocessed_action", "action_postprocess", value=[0.1])
                recorder.event(2, "postprocessed_action", "action_postprocess", value=[0.2])
                recorder.event(20, "outcome", "outcome", success=True, steps=20)

            report = check_conformance_file(trace)

        self.assertFalse(_check(report, "monotonic_steps"))
        self.assertEqual(report.achieved_level, "L0_WELL_FORMED")

    def test_cli_conformance_exit_codes_and_json(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "clean.jsonl"
            with TraceRecorder(
                trace,
                run_id="rp_run_clean_004",
                target="pi05-libero",
                scenario="clean",
                seed=9,
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(20, "outcome", "outcome", success=True, steps=20)

            ok = subprocess.run(
                [sys.executable, "-m", "robotprobe", "conformance", str(trace), "--level", "L1_REPRODUCIBLE_INCIDENT"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(ok.returncode, 0, ok.stderr)
            payload = json.loads(ok.stdout)
            self.assertEqual(payload["spec"], SPEC_ID)
            self.assertEqual(payload["achieved_level"], "L1_REPRODUCIBLE_INCIDENT")
            self.assertTrue(payload["conformant"])

            fail = subprocess.run(
                [sys.executable, "-m", "robotprobe", "conformance", str(trace), "--level", "L2_CONTRACT_CHECKABLE"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(fail.returncode, 1)
            self.assertFalse(json.loads(fail.stdout)["conformant"])


if __name__ == "__main__":
    unittest.main()
