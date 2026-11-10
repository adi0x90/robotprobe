import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe import ReplayLevel, TraceRecorder
from robotprobe.worldmodel import check_world_model_faithfulness, render_markdown


ROOT = Path(__file__).resolve().parents[1]


def _trace(path, *, predicted_state, realized_state, predicted_success, success):
    with TraceRecorder(
        path, run_id="rp_wm", target="wm-bot", scenario="imagined_evidence",
        seed=0, replay_level=ReplayLevel.TRACE_PLAYBACK,
    ) as r:
        r.event(
            5, "world_model_output", "imagined_evidence",
            generated_image_ref="gen/000005.png", request_ref="req/000005.json",
            predicted_state=predicted_state, predicted_success=predicted_success,
        )
        r.event(6, "observation", "observation", image_sha256="sha256:real", state=realized_state)
        r.event(7, "postprocessed_action", "action_postprocess", value=[0.1, 0.2])
        r.event(20, "outcome", "outcome", success=success, steps=20)


class WorldModelFaithfulnessTests(unittest.TestCase):
    def test_faithful_imagined_evidence_scores_one(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td) / "faithful.jsonl"
            _trace(t, predicted_state=[1.0, 2.0], realized_state=[1.0, 2.0], predicted_success=True, success=True)
            report = check_world_model_faithfulness(t)
        self.assertEqual(report["n_imagined_events"], 1)
        self.assertEqual(report["n_divergent"], 0)
        self.assertEqual(report["faithfulness_rate"], 1.0)
        self.assertTrue(report["records"][0]["faithful"])
        self.assertEqual(report["records"][0]["lead_time_steps"], 2)  # action at 7, imagined at 5

    def test_state_divergence_and_success_mismatch_are_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td) / "divergent.jsonl"
            _trace(t, predicted_state=[1.0, 2.0], realized_state=[9.0, 9.0], predicted_success=True, success=False)
            report = check_world_model_faithfulness(t)
        rec = report["records"][0]
        self.assertFalse(rec["faithful"])
        self.assertIn("state_divergence", rec["divergence_signals"])
        self.assertIn("success_mismatch", rec["divergence_signals"])
        self.assertGreater(rec["state_divergence"], 0.5)
        self.assertEqual(report["n_divergent"], 1)
        self.assertEqual(report["min_divergent_lead_time"], 2)
        self.assertIn("World-Model Faithfulness", render_markdown(report))

    def test_no_imagined_events_reports_no_faithfulness_rate(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td) / "none.jsonl"
            with TraceRecorder(t, run_id="rp_n", target="b", scenario="clean", seed=0,
                               replay_level=ReplayLevel.TRACE_PLAYBACK) as r:
                r.event(0, "outcome", "outcome", success=True, steps=1)
            report = check_world_model_faithfulness(t)
        self.assertEqual(report["n_imagined_events"], 0)
        # No imagined events means no evidence either way, not perfect faithfulness.
        self.assertIsNone(report["faithfulness_rate"])

    def test_cli_fail_on_divergence(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td) / "divergent.jsonl"
            _trace(t, predicted_state=[0.0], realized_state=[5.0], predicted_success=False, success=False)
            proc = subprocess.run(
                [sys.executable, "-m", "robotprobe", "worldmodel", "faithfulness",
                 "--trace", str(t), "--fail-on-divergence"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["n_divergent"], 1)


if __name__ == "__main__":
    unittest.main()
