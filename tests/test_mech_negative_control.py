import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.mech.activation_store import ActivationIndexWriter, ActivationStore
from robotprobe.mech.diff import diff_activations
from robotprobe.mech.report import build_mech_report


class MechNegativeControlTests(unittest.TestCase):
    def test_post_policy_negative_control_keeps_internal_activation_stable(self):
        with tempfile.TemporaryDirectory() as td:
            mech_dir = Path(td)
            index_path = _write_negative_control_fixture(mech_dir)

            diff = diff_activations(index_path=index_path, out_path=mech_dir / "activation_diff.json")

        layer = diff["layers"][0]
        self.assertIsNone(layer["first_significant_divergence_step"])
        self.assertFalse(layer["activation_diff_significant"])
        self.assertTrue(layer["negative_control_stable"])
        self.assertFalse(diff["hypothesis"]["internal_divergence_precedes_action_divergence"])
        self.assertEqual(diff["hypothesis"]["recommended_attribution"], "external_or_downstream")

    def test_mech_report_keeps_negative_control_below_model_root_cause_claim(self):
        summary = {
            "root_cause_hypothesis": {"layer": "action_postprocess", "fault": "gripper_freeze_mid"},
            "evidence": {
                "raw_policy_action_changed": False,
                "postprocessed_action_changed": True,
                "first_divergence_step": 35,
            },
        }
        activation_diff = {
            "format": "robotprobe.mech.activation_diff.v0",
            "synthetic": False,
            "evidence_eligible": True,
            "comparison_complete": True,
            "incident_id": "gripper_freeze_negative_control",
            "layers": [
                {
                    "module": "policy.backbone.layers.18",
                    "paired_step_count": 1,
                    "comparison_complete": True,
                    "activation_diff_significant": False,
                    "negative_control_stable": True,
                    "first_significant_divergence_step": None,
                    "external_first_divergence_step": 35,
                    "lead_time_steps": None,
                }
            ],
            "hypothesis": {
                "internal_divergence_precedes_action_divergence": False,
                "recommended_attribution": "external_or_downstream",
            },
        }

        report = build_mech_report(summary=summary, activation_diff=activation_diff)

        self.assertEqual(report["format"], "robotprobe.mech.report.v0")
        self.assertEqual(report["evidence_level"], 1)
        self.assertFalse(report["model_root_cause_claim_allowed"])
        self.assertIn("do not blame model internals", report["claim_boundary"])

    def test_cli_mech_report_writes_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summary_path = root / "summary.json"
            diff_path = root / "activation_diff.json"
            out_path = root / "mech_report.md"
            summary_path.write_text(
                json.dumps(
                    {
                        "root_cause_hypothesis": {"layer": "action_postprocess", "fault": "gripper_freeze_mid"},
                        "evidence": {"first_divergence_step": 35},
                    }
                ),
                encoding="utf-8",
            )
            diff_path.write_text(
                json.dumps(
                    {
                        "format": "robotprobe.mech.activation_diff.v0",
                        "synthetic": False,
                        "evidence_eligible": True,
                        "comparison_complete": True,
                        "layers": [
                            {
                                "module": "policy.backbone.layers.18",
                                "paired_step_count": 1,
                                "comparison_complete": True,
                                "activation_diff_significant": False,
                                "negative_control_stable": True,
                                "first_significant_divergence_step": None,
                                "external_first_divergence_step": 35,
                                "lead_time_steps": None,
                            }
                        ],
                        "hypothesis": {"recommended_attribution": "external_or_downstream"},
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "mech",
                    "report",
                    "--summary",
                    str(summary_path),
                    "--activation-diff",
                    str(diff_path),
                    "--out",
                    str(out_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            text = out_path.read_text(encoding="utf-8")

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("MechProbe Incident Report", text)
        self.assertIn("What This Does Not Prove", text)
        self.assertIn("do not blame model internals", text)


def _write_negative_control_fixture(mech_dir: Path) -> Path:
    store = ActivationStore(mech_dir)
    writer = ActivationIndexWriter(
        mech_dir,
        incident_id="gripper_freeze_negative_control",
        clean_run_id="clean_001",
        perturbed_run_id="gripper_freeze_001",
        model={"name": "pi0.5", "checkpoint_hash": "sha256:test", "adapter": "mechprobe.synthetic.v0"},
        external_first_divergence_step=35,
        negative_control=True,
    )
    for run_label, run_id in [("clean", "clean_001"), ("perturbed", "gripper_freeze_001")]:
        for step in [10, 35]:
            writer.add(
                store.write_activation(
                    run_label=run_label,
                    run_id=run_id,
                    step=step,
                    module="policy.backbone.layers.18",
                    module_kind="residual_stream",
                    activation=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                )
            )
    return writer.write()


if __name__ == "__main__":
    unittest.main()
