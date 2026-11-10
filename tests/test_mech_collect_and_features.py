import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.core import ReplayLevel
from robotprobe.mech.collect import collect_activations
from robotprobe.mech.features import diff_features
from robotprobe.record import TraceRecorder


class MechCollectAndFeatureDiffTests(unittest.TestCase):
    def test_synthetic_collect_writes_activation_index_events_and_diffable_activations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clean, perturbed = _write_trace_pair(root)
            config = _write_capture_config(root)
            mech_dir = root / "mech"

            result = collect_activations(
                target="synthetic",
                trace_pair=f"{clean},{perturbed}",
                capture_config=config,
                out_dir=mech_dir,
            )

        self.assertEqual(result["format"], "robotprobe.mech.collect.v0")
        self.assertEqual(result["target"], "synthetic")
        self.assertEqual(result["status"], "captured")
        self.assertEqual(result["external_first_divergence_step"], 14)
        self.assertEqual(result["activation_count"], 4)
        self.assertTrue(result["activation_index"].endswith("activation_index.json"))
        self.assertEqual(set(result["activation_events"]), {"clean", "perturbed"})
        self.assertTrue(result["activation_events"]["clean"].endswith("model_activation_events.clean.jsonl"))

    def test_cli_collect_synthetic_outputs_json_and_writes_index(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clean, perturbed = _write_trace_pair(root)
            config = _write_capture_config(root)
            mech_dir = root / "mech"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "mech",
                    "collect",
                    "--target",
                    "synthetic",
                    "--trace-pair",
                    f"{clean},{perturbed}",
                    "--capture-config",
                    str(config),
                    "--out",
                    str(mech_dir),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout)

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["status"], "captured")
        self.assertEqual(result["activation_count"], 4)

    def test_feature_diff_defaults_to_candidate_and_stays_below_promotion_level(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clean, perturbed = _write_trace_pair(root)
            config = _write_capture_config(root)
            dictionary = _write_dictionary(root)
            mech_dir = root / "mech"
            collect_activations(
                target="synthetic",
                trace_pair=f"{clean},{perturbed}",
                capture_config=config,
                out_dir=mech_dir,
            )

            report = diff_features(
                mech_dir=mech_dir,
                dictionary_path=dictionary,
                out_path=mech_dir / "feature_diff.json",
            )

        feature = report["features"][0]
        self.assertEqual(report["format"], "robotprobe.mech.feature_diff.v0")
        self.assertEqual(feature["label"], "candidate: target-object selection")
        self.assertEqual(feature["first_feature_divergence_step"], 10)
        self.assertEqual(feature["first_action_divergence_step"], 14)
        self.assertEqual(feature["lead_time_steps"], 4)
        self.assertEqual(feature["evidence_level"], 0)
        self.assertFalse(feature["promotion_allowed"])

    def test_cli_diff_features_outputs_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clean, perturbed = _write_trace_pair(root)
            config = _write_capture_config(root)
            dictionary = _write_dictionary(root)
            mech_dir = root / "mech"
            collect_activations(
                target="synthetic",
                trace_pair=f"{clean},{perturbed}",
                capture_config=config,
                out_dir=mech_dir,
            )
            out_path = mech_dir / "feature_diff.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "mech",
                    "diff-features",
                    "--mech-dir",
                    str(mech_dir),
                    "--dictionary",
                    str(dictionary),
                    "--out",
                    str(out_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout)

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["features"][0]["label"], "candidate: target-object selection")
        self.assertFalse(result["features"][0]["promotion_allowed"])


def _write_trace_pair(root: Path) -> tuple[Path, Path]:
    clean = root / "clean.jsonl"
    perturbed = root / "perturbed.jsonl"
    with TraceRecorder(
        clean,
        run_id="clean_001",
        target="synthetic-vla",
        scenario="instruction_swap",
        replay_level=ReplayLevel.TRACE_PLAYBACK,
    ) as recorder:
        recorder.event(0, "instruction", "instruction", value="pick mug")
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
        recorder.event(0, "instruction", "instruction", before="pick mug", after="pick bowl", value="pick bowl")
        recorder.event(10, "raw_action", "policy_forward", value=[1.0, 0.0, 0.0])
        recorder.event(14, "raw_action", "policy_forward", value=[0.0, 1.0, 0.0])
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
                "layers": [{"name": "policy.backbone.layers.18", "kind": "residual_stream", "capture": "output"}],
                "storage": {"format": "safetensors", "dtype": "float16", "max_bytes_per_run": 4096},
                "claim_boundary": {"no_feature_labels": True, "no_causal_claims": True},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_dictionary(root: Path) -> Path:
    path = root / "dictionary.json"
    path.write_text(
        json.dumps(
            {
                "format": "robotprobe.mech.dictionary.v0",
                "id": "synthetic-target-direction",
                "type": "manual_direction",
                "model_name": "synthetic",
                "features": [
                    {
                        "feature_id": "target_object_direction",
                        "label": "target-object selection",
                        "layer": "policy.backbone.layers.18",
                        "direction": [1.0, 0.0, 0.0],
                    }
                ],
                "claim_boundary": {"labels_are_candidates": True, "causal_without_intervention": False},
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
