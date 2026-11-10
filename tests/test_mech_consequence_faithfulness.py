import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.core import ReplayLevel
from robotprobe.mech.collect import collect_activations
from robotprobe.mech.faithfulness import compute_consequence_faithfulness
from robotprobe.record import TraceRecorder
from robotprobe.summarize import summarize_trace_pair


class MechConsequenceFaithfulnessTests(unittest.TestCase):
    def test_offline_consequence_faithfulness_stays_below_causal_promotion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mech_dir, summary_path = _write_mech_dir_and_summary(root)
            out_path = mech_dir / "consequence_aopc.json"

            result = compute_consequence_faithfulness(
                mech_dir=mech_dir,
                summary_path=summary_path,
                target_metric="raw_action_delta",
                out_path=out_path,
                max_k=2,
            )
            wrote_output = out_path.exists()

        self.assertEqual(result["format"], "robotprobe.mech.consequence_aopc.v0")
        self.assertEqual(result["mode"], "offline_proxy")
        self.assertEqual(result["target_metric"], "raw_action_delta")
        self.assertEqual(result["evidence_level"], 0)
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["intervention_executed"])
        self.assertEqual(result["components"][0]["top_dimensions"], [0, 1])
        self.assertGreater(result["components"][0]["aopc_score"], 0.0)
        self.assertTrue(wrote_output)

    def test_offline_consequence_faithfulness_handles_no_consequence_delta(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mech_dir, summary_path = _write_mech_dir_and_summary(root, raw_action_changed=False)

            result = compute_consequence_faithfulness(
                mech_dir=mech_dir,
                summary_path=summary_path,
                target_metric="raw_action_delta",
                max_k=2,
            )

        self.assertEqual(result["consequence_score"], 0.0)
        self.assertEqual(result["evidence_level"], 0)
        self.assertEqual(result["components"][0]["aopc_score"], 0.0)

    def test_cli_faithfulness_outputs_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mech_dir, summary_path = _write_mech_dir_and_summary(root)
            out_path = mech_dir / "consequence_aopc.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "mech",
                    "faithfulness",
                    "--mech-dir",
                    str(mech_dir),
                    "--summary",
                    str(summary_path),
                    "--target-metric",
                    "raw_action_delta",
                    "--max-k",
                    "2",
                    "--out",
                    str(out_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout)
            wrote_output = out_path.exists()

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["format"], "robotprobe.mech.consequence_aopc.v0")
        self.assertEqual(result["evidence_level"], 0)
        self.assertTrue(wrote_output)


def _write_mech_dir_and_summary(root: Path, *, raw_action_changed: bool = True) -> tuple[Path, Path]:
    clean, perturbed = _write_trace_pair(root, raw_action_changed=raw_action_changed)
    config = _write_capture_config(root)
    mech_dir = root / "mech"
    collect_activations(
        target="synthetic",
        trace_pair=f"{clean},{perturbed}",
        capture_config=config,
        out_dir=mech_dir,
    )
    summary = summarize_trace_pair(clean, perturbed)
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return mech_dir, summary_path


def _write_trace_pair(root: Path, *, raw_action_changed: bool) -> tuple[Path, Path]:
    clean = root / "clean.jsonl"
    perturbed = root / "perturbed.jsonl"
    perturbed_final = [0.0, 1.0, 0.0] if raw_action_changed else [1.0, 0.0, 0.0]
    with TraceRecorder(
        clean,
        run_id="clean_001",
        target="synthetic-vla",
        scenario="instruction_swap",
        replay_level=ReplayLevel.TRACE_PLAYBACK,
    ) as recorder:
        recorder.event(10, "raw_action", "policy_forward", value=[1.0, 0.0, 0.0])
        recorder.event(14, "raw_action", "policy_forward", value=[1.0, 0.0, 0.0])
    with TraceRecorder(
        perturbed,
        run_id="instruction_swap_001",
        target="synthetic-vla",
        scenario="instruction_swap",
        baseline_run_id="clean_001",
        perturbation_group_id="instruction_swap_001",
        replay_level=ReplayLevel.TRACE_PLAYBACK,
    ) as recorder:
        recorder.event(10, "raw_action", "policy_forward", value=[1.0, 0.0, 0.0])
        recorder.event(14, "raw_action", "policy_forward", value=perturbed_final)
    return clean, perturbed


def _write_capture_config(root: Path) -> Path:
    path = root / "capture.json"
    path.write_text(
        json.dumps(
            {
                "format": "robotprobe.mech.capture_config.v0",
                "id": "synthetic_instruction_swap",
                "target": "synthetic",
                "model_access": "white_box",
                "layers": [
                    {
                        "name": "policy.backbone.layers.18",
                        "kind": "residual_stream",
                        "capture": "output",
                    }
                ],
                "storage": {"format": "safetensors", "dtype": "float16", "max_bytes_per_run": 4096},
                "claim_boundary": {"no_feature_labels": True, "no_causal_claims": True},
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
