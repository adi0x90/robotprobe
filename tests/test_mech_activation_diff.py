import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.mech.activation_store import ActivationIndexWriter, ActivationStore
from robotprobe.mech.diff import diff_activations


class MechActivationDiffTests(unittest.TestCase):
    def test_activation_diff_finds_internal_lead_time_before_external_action_divergence(self):
        with tempfile.TemporaryDirectory() as td:
            mech_dir = Path(td)
            index_path = _write_instruction_swap_fixture(mech_dir)

            diff = diff_activations(index_path=index_path, out_path=mech_dir / "activation_diff.json")
            wrote_diff = (mech_dir / "activation_diff.json").exists()

        layer = diff["layers"][0]
        self.assertEqual(diff["format"], "robotprobe.mech.activation_diff.v0")
        self.assertEqual(layer["module"], "policy.backbone.layers.18")
        self.assertEqual(layer["first_significant_divergence_step"], 10)
        self.assertEqual(layer["external_first_divergence_step"], 14)
        self.assertEqual(layer["lead_time_steps"], 4)
        self.assertTrue(diff["hypothesis"]["internal_divergence_precedes_action_divergence"])
        self.assertGreater(layer["max_l2_delta"], 1.0)
        self.assertTrue(wrote_diff)

    def test_cli_diff_activations_reads_mech_dir_index(self):
        with tempfile.TemporaryDirectory() as td:
            mech_dir = Path(td)
            _write_instruction_swap_fixture(mech_dir)
            out_path = mech_dir / "activation_diff.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "mech",
                    "diff-activations",
                    "--mech-dir",
                    str(mech_dir),
                    "--out",
                    str(out_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout)

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["layers"][0]["first_significant_divergence_step"], 10)
        self.assertEqual(result["layers"][0]["lead_time_steps"], 4)


def _write_instruction_swap_fixture(mech_dir: Path) -> Path:
    store = ActivationStore(mech_dir)
    writer = ActivationIndexWriter(
        mech_dir,
        incident_id="instruction_swap_001",
        clean_run_id="clean_001",
        perturbed_run_id="instruction_swap_001",
        model={"name": "pi0.5", "checkpoint_hash": "sha256:test", "adapter": "mechprobe.synthetic.v0"},
        external_first_divergence_step=14,
    )
    for run_label, run_id, direction in [
        ("clean", "clean_001", np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        ("perturbed", "instruction_swap_001", np.array([0.0, 0.0, 0.0], dtype=np.float32)),
    ]:
        writer.add(
            store.write_activation(
                run_label=run_label,
                run_id=run_id,
                step=5,
                module="policy.backbone.layers.18",
                module_kind="residual_stream",
                activation=direction,
            )
        )
    writer.add(
        store.write_activation(
            run_label="clean",
            run_id="clean_001",
            step=10,
            module="policy.backbone.layers.18",
            module_kind="residual_stream",
            activation=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
    )
    writer.add(
        store.write_activation(
            run_label="perturbed",
            run_id="instruction_swap_001",
            step=10,
            module="policy.backbone.layers.18",
            module_kind="residual_stream",
            activation=np.array([3.0, 1.0, 1.0], dtype=np.float32),
        )
    )
    return writer.write()


if __name__ == "__main__":
    unittest.main()
