import json
import tempfile
import unittest
from pathlib import Path

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder
from robotprobe.schema import validate_trace_file
from robotsec_range.challenge_spec import load_builtin_challenge


class ImaginedEvidenceTests(unittest.TestCase):
    def test_world_model_output_trace_event_validates(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "imagined.jsonl"
            with TraceRecorder(
                trace,
                run_id="rp_imagined_001",
                target="worldmodel-demo",
                scenario="imagination_poison_signal",
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(
                    0,
                    "subgoal_request",
                    "world_model",
                    request_ref="requests/000000.json",
                    source="forced_invoke",
                    requested_motion={"type": "yaw", "magnitude_deg": 30},
                )
                recorder.event(
                    0,
                    "world_model_output",
                    "imagined_evidence",
                    image_sha256="sha256:clean",
                    generated_image_ref="generated/000000.png",
                    request_ref="requests/000000.json",
                    provenance={"world_model": "worldmodel-demo"},
                    consistency={"pose": 0.91, "content": 0.88},
                    plausibility={"lpips": 0.04},
                )

            report = validate_trace_file(trace)
            event_types = [json.loads(line)["event_type"] for line in trace.read_text().splitlines()]

        self.assertTrue(report.valid, [str(issue) for issue in report.issues])
        self.assertEqual(report.issues, [])
        self.assertEqual(event_types, ["subgoal_request", "world_model_output"])

    def test_builtin_challenge_includes_imagined_evidence_integrity(self):
        challenge = load_builtin_challenge("robotprobe-v2-imagined-evidence-integrity")

        self.assertEqual(challenge.diagnosis["layer"], "imagined_evidence")
        self.assertEqual(challenge.diagnosis["contract"], "ImaginedEvidenceIntegrityContract")
        self.assertEqual(challenge.diagnosis["fault"], "world_model_output_shift")
        self.assertEqual(challenge.required_evidence["world_model_output_changed"], True)
        self.assertEqual(challenge.required_evidence["plausibility_preserved"], True)
        self.assertIn("on_subgoal_request", challenge.metadata["hook_points"])
        self.assertIn("on_world_model_output", challenge.metadata["hook_points"])


if __name__ == "__main__":
    unittest.main()
