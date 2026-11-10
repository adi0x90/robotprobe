import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.core import ReplayLevel
from robotprobe.mech.activation_store import ActivationIndexWriter, ActivationStore
from robotprobe.mech.cache_view import ActivationCacheView, build_cache_summary
from robotprobe.mech.collect import collect_activations
from robotprobe.mech.diff import diff_activations
from robotprobe.record import TraceRecorder


class MechCacheViewTests(unittest.TestCase):
    def test_cache_view_queries_file_backed_activation_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mech_dir = _write_synthetic_mech_dir(root)

            view = ActivationCacheView(mech_dir)
            pair = view["policy.backbone.layers.18", 10]
            perturbed_sequence = view.sequence(
                "policy.backbone.layers.18",
                run_label="perturbed",
            )
            summary = build_cache_summary(mech_dir)

        self.assertEqual(view.incident_id, "synthetic_instruction_swap")
        self.assertEqual(view.component_names, ["policy.backbone.layers.18"])
        self.assertEqual(view.run_labels, ["clean", "perturbed"])
        self.assertEqual(view.timesteps, [5, 10])
        self.assertEqual(pair["clean"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(pair["perturbed"].tolist(), [3.0, 1.0, 1.0])
        self.assertEqual(perturbed_sequence.steps, [5, 10])
        self.assertEqual(perturbed_sequence.values.tolist(), [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
        self.assertEqual(summary["format"], "robotprobe.mech.cache_summary.v0")
        self.assertEqual(summary["capture_count"], 4)
        self.assertEqual(summary["components"][0]["capture_count"], 4)
        self.assertIs(summary["synthetic"], True)
        self.assertIsNone(summary["evidence_eligible"])
        self.assertEqual(summary["evidence_status"], "not_assessed")

    def test_cache_summary_leaves_real_capture_eligibility_to_tensor_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            mech_dir = _write_npy_mech_dir(Path(td))
            summary = build_cache_summary(mech_dir)
            analysis = diff_activations(mech_dir=mech_dir)

        self.assertIs(summary["synthetic"], False)
        self.assertIs(summary["components"][0]["comparison_complete"], True)
        self.assertIsNone(summary["evidence_eligible"])
        self.assertEqual(summary["evidence_status"], "not_assessed")
        self.assertIn("metadata", summary["eligibility_reason"])
        self.assertIn("diff-activations", summary["eligibility_reason"])
        self.assertIs(analysis["evidence_eligible"], True)

    def test_cache_summary_does_not_validate_corrupt_tensor_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            mech_dir = _write_npy_mech_dir(Path(td))
            record = ActivationCacheView(mech_dir).records()[0]
            (mech_dir / record["activation_ref"]).write_bytes(b"invalid numpy payload")

            summary = build_cache_summary(mech_dir)

            self.assertIsNone(summary["evidence_eligible"])
            self.assertEqual(summary["evidence_status"], "not_assessed")
            self.assertEqual(summary["capture_count"], 2)
            self.assertIs(summary["components"][0]["comparison_complete"], True)
            with self.assertRaises(ValueError):
                diff_activations(mech_dir=mech_dir)

    def test_cache_view_rejects_missing_index_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mech_dir = _write_synthetic_mech_dir(root)

            with self.assertRaises(FileNotFoundError):
                ActivationCacheView(root / "missing")

            view = ActivationCacheView(mech_dir)

            with self.assertRaises(KeyError):
                view.get("policy.backbone.layers.18", 999, run_label="clean")
            with self.assertRaises(KeyError):
                view.get("policy.backbone.layers.18", 10, run_label="missing")
            with self.assertRaises(KeyError):
                _ = view["unknown.module", 10]

    def test_cli_cache_info_outputs_summary_without_model_runtime_imports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mech_dir = _write_synthetic_mech_dir(root)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "mech",
                    "cache-info",
                    "--mech-dir",
                    str(mech_dir),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout)

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["format"], "robotprobe.mech.cache_summary.v0")
        self.assertEqual(result["runs"], ["clean", "perturbed"])
        self.assertEqual(result["components"][0]["module"], "policy.backbone.layers.18")
        self.assertIsNone(result["evidence_eligible"])
        self.assertEqual(result["evidence_status"], "not_assessed")


def _write_npy_mech_dir(root: Path) -> Path:
    mech_dir = root / "mech"
    writer = ActivationIndexWriter(
        mech_dir, incident_id="test", clean_run_id="clean", perturbed_run_id="perturbed",
        model={"name": "test-model"}, synthetic=False,
    )
    store = ActivationStore(mech_dir, store_format="npy")
    for label in ("clean", "perturbed"):
        writer.add(store.write_activation(
            run_label=label, run_id=label, step=0, module="test.layer",
            module_kind="activation", activation=np.array([1.0, 2.0]),
        ))
    writer.write()
    return mech_dir


def _write_synthetic_mech_dir(root: Path) -> Path:
    clean, perturbed = _write_trace_pair(root)
    config = _write_capture_config(root)
    mech_dir = root / "mech"
    collect_activations(
        target="synthetic",
        trace_pair=f"{clean},{perturbed}",
        capture_config=config,
        out_dir=mech_dir,
    )
    return mech_dir


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
