import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.core import ReplayLevel
from robotprobe.mech.activation_store import ActivationStore
from robotprobe.mech.schemas import FeatureFinding, MechEvidenceLevel, ModelActivationEvent, validate_capture_config
from robotprobe.packs import validate_pack
from robotprobe.schema import validate_trace_events


ROOT = Path(__file__).resolve().parents[1]


class MechActivationSchemaTests(unittest.TestCase):
    def test_model_activation_event_validates_with_file_reference(self):
        with tempfile.TemporaryDirectory() as td:
            mech_dir = Path(td)
            store = ActivationStore(mech_dir)
            capture = store.write_activation(
                run_label="clean",
                run_id="clean_001",
                step=10,
                module="policy.backbone.layers.18",
                module_kind="residual_stream",
                activation=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )
            event = ModelActivationEvent(
                run_id="clean_001",
                target="pi05-libero",
                scenario="instruction_swap",
                replay_level=ReplayLevel.TRACE_PLAYBACK.value,
                step=10,
                module="policy.backbone.layers.18",
                module_kind="residual_stream",
                activation_ref=capture.activation_ref,
                activation_shape=capture.shape,
                activation_dtype=capture.dtype,
                model_ref={"model_name": "pi0.5", "checkpoint_hash": "sha256:test", "runtime": "torch"},
            )

            trace_event = event.to_trace_event()
            report = validate_trace_events([trace_event])

        self.assertTrue(report.valid, report.issues)
        self.assertEqual(trace_event["event_type"], "model_activation")
        self.assertEqual(trace_event["layer"], "model_activation")
        self.assertEqual(trace_event["activation_shape"], [3])
        self.assertIn("capture", trace_event)

    def test_model_activation_event_requires_activation_ref_shape_dtype_and_model_ref(self):
        event = {
            "run_id": "clean_001",
            "target": "pi05-libero",
            "scenario": "instruction_swap",
            "replay_level": "L0_TRACE_PLAYBACK",
            "step": 10,
            "event_type": "model_activation",
            "layer": "model_activation",
            "module": "policy.backbone.layers.18",
        }

        report = validate_trace_events([event])
        fields = {issue.field for issue in report.issues}

        self.assertFalse(report.valid)
        self.assertIn("activation_ref", fields)
        self.assertIn("activation_shape", fields)
        self.assertIn("activation_dtype", fields)
        self.assertIn("model_ref", fields)

    def test_capture_config_validation_rejects_missing_claim_boundary(self):
        config = {
            "format": "robotprobe.mech.capture_config.v0",
            "id": "pi05_l18",
            "target": "pi05-libero",
            "model_access": "white_box",
            "layers": [{"name": "policy.backbone.layers.18", "kind": "residual_stream", "capture": "output"}],
            "storage": {"format": "safetensors", "dtype": "float16", "max_bytes_per_run": 1024},
        }

        result = validate_capture_config(config)

        self.assertFalse(result["valid"])
        self.assertIn("claim_boundary", result["errors"])

    def test_feature_findings_default_to_candidate_and_do_not_promote_below_level_three(self):
        finding = FeatureFinding(feature_id="sae_l18_f428", label="target-object selection")

        self.assertEqual(finding.label, "candidate: target-object selection")
        self.assertEqual(finding.evidence_level, MechEvidenceLevel.SEMANTIC_LOOKS_RELEVANT)
        self.assertFalse(finding.promotion_allowed)

    def test_cli_mech_validate_outputs_json(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "capture.json"
            config_path.write_text(
                json.dumps(
                    {
                        "format": "robotprobe.mech.capture_config.v0",
                        "id": "pi05_l18",
                        "target": "pi05-libero",
                        "model_access": "white_box",
                        "layers": [
                            {"name": "policy.backbone.layers.18", "kind": "residual_stream", "capture": "output"}
                        ],
                        "storage": {"format": "safetensors", "dtype": "float16", "max_bytes_per_run": 1024},
                        "claim_boundary": {"no_feature_labels": True, "no_causal_claims": True},
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [sys.executable, "-m", "robotprobe", "mech", "validate", "--capture-config", str(config_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout)

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["format"], "robotprobe.mech.capture_config.validation.v0")

    def test_mechprobe_model_specific_behavior_starts_as_a_pack(self):
        report = validate_pack(ROOT / "examples" / "packs" / "mechprobe-openvla-libero")

        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(report.id, "robotprobe-pack-mechprobe-openvla-libero")


if __name__ == "__main__":
    unittest.main()
