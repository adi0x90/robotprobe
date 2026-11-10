import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder
from robotprobe.schema import validate_trace_file


ROOT = Path(__file__).resolve().parents[1]


class RobotProbeV1FeatureTests(unittest.TestCase):
    def test_validate_trace_file_and_cli_report_schema_errors(self):
        with tempfile.TemporaryDirectory() as td:
            valid_trace = Path(td) / "valid.jsonl"
            invalid_trace = Path(td) / "invalid.jsonl"
            with TraceRecorder(
                valid_trace,
                run_id="rp_run_valid",
                target="pi05-libero",
                scenario="clean",
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(0, "instruction", "instruction", value="pick red")
                recorder.event(1, "raw_action", "policy_forward", value=[0.1, 0.2, 0.3])
                recorder.event(2, "outcome", "outcome", success=True, steps=2)
            invalid_trace.write_text(json.dumps({"run_id": "missing required fields"}) + "\n", encoding="utf-8")

            valid_report = validate_trace_file(valid_trace)
            proc = subprocess.run(
                [sys.executable, "-m", "robotprobe", "validate-trace", str(invalid_trace)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        self.assertTrue(valid_report.valid)
        self.assertEqual(valid_report.event_count, 3)
        self.assertNotEqual(proc.returncode, 0)
        invalid_report = json.loads(proc.stdout)
        self.assertFalse(invalid_report["valid"])
        self.assertIn("missing required field", invalid_report["issues"][0]["message"])

    def test_replay_cli_supports_l0_l1_l2_and_l3_shadow_envelope(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "trace.jsonl"
            envelope = Path(td) / "envelope.json"
            envelope.write_text(json.dumps({"max_abs_action": 1.0}), encoding="utf-8")
            with TraceRecorder(
                trace,
                run_id="rp_replay",
                target="pi05-libero",
                scenario="sim_replay",
                replay_level=ReplayLevel.SIMULATOR_REPLAY,
            ) as recorder:
                recorder.event(0, "instruction", "instruction", value="pick red", sim_seed=123, sim_config="libero_spatial")
                recorder.event(0, "observation", "observation", image_sha256="sha256:a")
                recorder.event(0, "policy_input", "policy_forward", observation_ref="obs/000000.json")
                recorder.event(0, "postprocessed_action", "action_postprocess", value=[0.0, 1.2, 0.0])

            l0 = subprocess.run(
                [sys.executable, "-m", "robotprobe", "replay", "--level", "L0", str(trace)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            l1 = subprocess.run(
                [sys.executable, "-m", "robotprobe", "replay", "--level", "L1", str(trace)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            l2 = subprocess.run(
                [sys.executable, "-m", "robotprobe", "replay", "--level", "L2", str(trace)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            l3 = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "replay",
                    "--level",
                    "L3",
                    "--action-envelope",
                    str(envelope),
                    str(trace),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertEqual(json.loads(l0.stdout)["timeline"][0]["event_type"], "instruction")
        self.assertEqual(json.loads(l1.stdout)["policy_input_count"], 1)
        self.assertEqual(json.loads(l2.stdout)["sim_seed"], 123)
        shadow = json.loads(l3.stdout)
        self.assertFalse(shadow["shadow_passed"])
        self.assertEqual(shadow["violations"][0]["step"], 0)

    def test_summarize_cli_computes_metric_plugins(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = ROOT / "interventions" / "gripper_freeze_mid.yaml"
            clean, perturbed = _write_metric_trace_pair(Path(td), manifest)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "summarize",
                    "--clean",
                    str(clean),
                    "--perturbed",
                    str(perturbed),
                    "--manifest",
                    str(manifest),
                    "--metrics",
                    "action_energy,jerk,trajectory_deviation,recovery_latency,near_miss",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

        summary = json.loads(proc.stdout)
        self.assertIn("metrics", summary)
        self.assertGreater(summary["metrics"]["perturbed"]["action_energy"], summary["metrics"]["clean"]["action_energy"])
        self.assertGreater(summary["metrics"]["delta"]["trajectory_deviation"], 0.0)
        self.assertEqual(summary["metrics"]["delta"]["recovery_latency"], 2)
        self.assertGreater(summary["metrics"]["delta"]["near_miss_score"], 0.0)

    def test_bundle_cli_writes_complete_incident_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = ROOT / "interventions" / "gripper_freeze_mid.yaml"
            clean, perturbed = _write_metric_trace_pair(Path(td), manifest)
            out_dir = Path(td) / "incident"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "bundle",
                    "--clean",
                    str(clean),
                    "--perturbed",
                    str(perturbed),
                    "--manifest",
                    str(manifest),
                    "--challenge",
                    "robotprobe-v0-gripper-freeze",
                    "--metrics",
                    "action_energy,near_miss",
                    "--out",
                    str(out_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            bundle = json.loads(proc.stdout)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "causal_diff.json").exists())
            self.assertTrue((out_dir / "check.json").exists())
            self.assertTrue((out_dir / "report.md").exists())
            self.assertTrue((out_dir / "manifests" / "gripper_freeze_mid.yaml").exists())
            self.assertTrue((out_dir / "traces" / "clean.jsonl").exists())
            self.assertEqual(bundle["replay_level"], "L0_TRACE_PLAYBACK")
            self.assertEqual(bundle["summary_ref"], "summary.json")


def _write_metric_trace_pair(directory: Path, manifest_ref: Path):
    clean_path = directory / "clean.jsonl"
    perturbed_path = directory / "fault.jsonl"
    clean_actions = [[0.0, 0.0, 0.2], [0.1, 0.0, 0.3], [0.2, 0.0, 0.4], [0.3, 0.0, 0.5]]
    perturbed_actions = [[0.0, 0.0, 0.2], [0.1, 0.0, 1.2], [0.2, 0.0, 1.2], [0.3, 0.0, 0.5]]

    with TraceRecorder(
        clean_path,
        run_id="rp_clean_metrics",
        target="pi05-libero",
        scenario="clean",
        replay_level=ReplayLevel.TRACE_PLAYBACK,
    ) as recorder:
        for step, action in enumerate(clean_actions):
            recorder.event(step, "raw_action", "policy_forward", value=action)
            recorder.event(step, "postprocessed_action", "action_postprocess", value=action)
        recorder.event(4, "outcome", "outcome", success=True, steps=4)

    with TraceRecorder(
        perturbed_path,
        run_id="rp_fault_metrics",
        target="pi05-libero",
        scenario="gripper_freeze_mid",
        baseline_run_id="rp_clean_metrics",
        perturbation_group_id="rp_pair_metrics",
        intervention_manifest_ref=manifest_ref,
        replay_level=ReplayLevel.TRACE_PLAYBACK,
    ) as recorder:
        for step, (raw, post) in enumerate(zip(clean_actions, perturbed_actions)):
            recorder.event(step, "raw_action", "policy_forward", value=raw)
            recorder.event(step, "postprocessed_action", "action_postprocess", value=post)
        recorder.event(7, "outcome", "outcome", success=True, steps=7)

    return clean_path, perturbed_path


if __name__ == "__main__":
    unittest.main()
