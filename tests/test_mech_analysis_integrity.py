import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.mech.activation_store import ActivationIndexWriter, ActivationStore
from robotprobe.mech.diff import diff_activations
from robotprobe.mech.faithfulness import compute_consequence_faithfulness
from robotprobe.mech.features import diff_features
from robotprobe.mech.report import build_mech_report, render_mech_report_markdown
from robotsec_range.challenge_spec import ChallengeSpec
from robotsec_range.mech_checker import check_mech_report


class MechAnalysisIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.summary = {
            "root_cause_hypothesis": {"layer": "instruction"},
            "evidence": {"raw_policy_action_changed": True, "first_divergence_step": 0},
        }
        self.summary_path = self.root / "summary.json"
        self.summary_path.write_text(json.dumps(self.summary))
        self.dictionary = self.root / "dictionary.json"
        self.dictionary.write_text(json.dumps({
            "id": "manual", "type": "manual_direction",
            "features": [{"feature_id": "f", "label": "direction", "layer": "m", "direction": [1, 0, 0]}],
        }))

    def cache(self, records=None, *, synthetic=False, negative_control=False):
        records = records or [
            ("clean", "m", 10, np.array([0.0, 0.0, 0.0])),
            ("perturbed", "m", 10, np.array([2.0, 1.0, 0.0])),
        ]
        folder = self.root / "mech"
        writer = ActivationIndexWriter(
            folder, incident_id="incident", clean_run_id="clean", perturbed_run_id="perturbed",
            model={"name": "test-model"}, external_first_divergence_step=14,
            synthetic=synthetic, negative_control=negative_control,
        )
        store = ActivationStore(folder, store_format="npy")
        for label, module, step, array in records:
            writer.add(store.write_activation(
                run_label=label, run_id=label, step=step, module=module,
                module_kind="activation", activation=array,
            ))
        writer.write()
        return folder

    def analyze_all(self, folder):
        activation = diff_activations(mech_dir=folder)
        features = diff_features(mech_dir=folder, dictionary_path=self.dictionary)
        faithfulness = compute_consequence_faithfulness(
            mech_dir=folder, summary_path=self.summary_path, target_metric="raw_action_delta",
        )
        report = build_mech_report(summary=self.summary, activation_diff=activation, feature_diff=features)
        return activation, features, faithfulness, report

    def test_synthetic_provenance_survives_every_analysis_stage(self):
        outputs = self.analyze_all(self.cache(synthetic=True))
        for output in outputs:
            with self.subTest(format=output["format"]):
                self.assertIs(output.get("synthetic"), True)
                self.assertIs(output.get("evidence_eligible"), False)
                self.assertEqual(output.get("evidence_level"), 0)
        self.assertEqual(outputs[1]["features"][0]["evidence_level"], 0)
        self.assertIn("synthetic", render_mech_report_markdown(outputs[-1]).lower())

    def test_real_offline_timing_is_correlational_not_predictive(self):
        outputs = self.analyze_all(self.cache())
        for output in outputs:
            with self.subTest(format=output["format"]):
                self.assertIs(output.get("evidence_eligible"), True)
                self.assertEqual(output.get("evidence_level"), 1)
        self.assertEqual(outputs[1]["features"][0]["evidence_level"], 1)
        self.assertNotIn("predicts", outputs[1]["features"][0]["reason"])
        self.assertNotIn("predictive", outputs[-1]["claim_boundary"])

    def test_missing_provenance_is_not_assumed_real(self):
        folder = self.cache()
        path = folder / "activation_index.json"
        index = json.loads(path.read_text())
        del index["synthetic"]
        path.write_text(json.dumps(index))
        for output in self.analyze_all(folder):
            self.assertIsNone(output.get("synthetic"))
            self.assertIs(output.get("evidence_eligible"), False)
            self.assertEqual(output.get("evidence_level"), 0)

    def test_unpaired_steps_are_not_a_stable_control(self):
        folder = self.cache([
            ("clean", "m", 10, np.zeros(3)), ("perturbed", "m", 11, np.zeros(3)),
        ], negative_control=True)
        diff = diff_activations(mech_dir=folder)
        layer = diff["layers"][0]
        self.assertFalse(layer["negative_control_stable"])
        self.assertFalse(layer["comparison_complete"])
        self.assertEqual(layer["paired_step_count"], 0)
        self.assertIsNone(layer["stability_score"])
        self.assertFalse(diff["evidence_eligible"])

    def test_partially_paired_steps_cannot_establish_a_stable_control(self):
        folder = self.cache([
            ("clean", "m", 10, np.zeros(3)), ("perturbed", "m", 10, np.zeros(3)),
            ("clean", "m", 11, np.zeros(3)),
        ], negative_control=True)
        diff = diff_activations(mech_dir=folder)
        self.assertFalse(diff["layers"][0]["negative_control_stable"])
        self.assertFalse(diff["comparison_complete"])
        self.assertFalse(diff["evidence_eligible"])

    def test_absent_required_module_makes_comparison_incomplete(self):
        folder = self.cache()
        path = folder / "activation_index.json"
        index = json.loads(path.read_text())
        index["required_modules"] = ["m", "missing"]
        path.write_text(json.dumps(index))
        diff = diff_activations(mech_dir=folder)
        self.assertFalse(diff["comparison_complete"])
        self.assertFalse(diff["evidence_eligible"])

    def test_all_layers_must_pass_negative_control(self):
        folder = self.cache([
            ("clean", "stable", 10, np.zeros(3)), ("perturbed", "stable", 10, np.zeros(3)),
            ("clean", "m", 10, np.zeros(3)), ("perturbed", "m", 10, np.ones(3)),
        ], negative_control=True)
        report = build_mech_report(
            summary={"root_cause_hypothesis": {"layer": "action_postprocess"}},
            activation_diff=diff_activations(mech_dir=folder),
        )
        self.assertFalse(report["negative_control_stable"])
        self.assertNotIn("negative control stable", report["claim_boundary"])

    def test_nonfinite_activations_are_rejected_by_each_analysis(self):
        folder = self.cache([
            ("clean", "m", 10, np.zeros(3)), ("perturbed", "m", 10, np.array([np.nan, 0, 0])),
        ], negative_control=True)
        for operation in (
            lambda: diff_activations(mech_dir=folder),
            lambda: diff_features(mech_dir=folder, dictionary_path=self.dictionary),
            lambda: compute_consequence_faithfulness(
                mech_dir=folder, summary_path=self.summary_path, target_metric="raw_action_delta"),
        ):
            with self.assertRaisesRegex(ValueError, "finite"):
                operation()

    def test_original_tensor_shape_is_checked_before_flattening(self):
        folder = self.cache([
            ("clean", "m", 10, np.arange(6).reshape(2, 3)),
            ("perturbed", "m", 10, np.arange(6).reshape(3, 2)),
        ])
        for operation in (
            lambda: diff_activations(mech_dir=folder),
            lambda: compute_consequence_faithfulness(
                mech_dir=folder, summary_path=self.summary_path, target_metric="raw_action_delta"),
        ):
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                operation()

    def test_capture_metadata_must_match_stored_tensor(self):
        folder = self.cache()
        path = folder / "activation_index.json"
        index = json.loads(path.read_text())
        index["captures"][0]["shape"] = [1, 3]
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "shape"):
            diff_activations(mech_dir=folder)

    def test_faithfulness_requires_consistent_original_shape_across_steps(self):
        folder = self.cache([
            ("clean", "m", 10, np.zeros((2, 3))), ("perturbed", "m", 10, np.ones((2, 3))),
            ("clean", "m", 11, np.zeros((3, 2))), ("perturbed", "m", 11, np.ones((3, 2))),
        ])
        with self.assertRaisesRegex(ValueError, "shape changes across steps"):
            compute_consequence_faithfulness(
                mech_dir=folder, summary_path=self.summary_path, target_metric="raw_action_delta",
            )

    def test_faithfulness_rejects_malformed_summary_mappings(self):
        folder = self.cache()
        for summary in ({"evidence": []}, {"metrics": []}, {"metrics": {"delta": []}},
                        {"evidence": {"outcome_delta": []}}):
            self.summary_path.write_text(json.dumps(summary))
            with self.subTest(summary=summary), self.assertRaisesRegex(ValueError, "object"):
                compute_consequence_faithfulness(
                    mech_dir=folder, summary_path=self.summary_path, target_metric="raw_action_delta",
                )

    def test_markdown_includes_features_and_distinguishes_action_timing(self):
        _, _, _, report = self.analyze_all(self.cache())
        self.assertEqual(report["first_external_divergence_step"], 0)
        self.assertEqual(report["first_action_divergence_step"], 14)
        markdown = render_mech_report_markdown(report)
        self.assertIn("candidate: direction", markdown)
        self.assertIn("First action divergence step: `14`", markdown)

    def test_report_rejects_malformed_summary_mappings(self):
        activation = diff_activations(mech_dir=self.cache())
        for summary in ({"root_cause_hypothesis": [], "evidence": {}}, {"evidence": []}):
            with self.subTest(summary=summary), self.assertRaisesRegex(ValueError, "objects"):
                build_mech_report(summary=summary, activation_diff=activation)

    def test_checker_rejects_forged_eligible_synthetic_and_unknown_reports(self):
        challenge = ChallengeSpec(id="gate", diagnosis={}, required_evidence={"min_evidence_level": 2})
        for provenance in (True, None):
            report = {"evidence_level": 4, "synthetic": provenance, "evidence_eligible": True}
            result = check_mech_report(challenge, report)
            self.assertFalse(result.passed)
            self.assertTrue(any("provenance" in error or "synthetic" in error for error in result.failures))

    def test_max_k_must_be_positive_integer(self):
        folder = self.cache()
        for max_k in (0, -1, True):
            with self.subTest(max_k=max_k), self.assertRaisesRegex(ValueError, "max_k"):
                compute_consequence_faithfulness(
                    mech_dir=folder, summary_path=self.summary_path,
                    target_metric="raw_action_delta", max_k=max_k,
                )

    def test_faithfulness_prefix_sum_preserves_score(self):
        result = compute_consequence_faithfulness(
            mech_dir=self.cache(), summary_path=self.summary_path, target_metric="raw_action_delta",
        )
        component = result["components"][0]
        np.testing.assert_allclose(component["metric_deltas"], [2 / 3, 1, 1])
        self.assertAlmostEqual(component["aopc_score"], 11 / 6)

    def test_default_faithfulness_handles_large_activation_without_quadratic_work(self):
        folder = self.cache([
            ("clean", "m", 10, np.zeros(32768)), ("perturbed", "m", 10, np.ones(32768)),
        ])
        code = (
            "from robotprobe.mech.faithfulness import compute_consequence_faithfulness; "
            "import sys; r=compute_consequence_faithfulness(mech_dir=sys.argv[1], "
            "summary_path=sys.argv[2], target_metric='raw_action_delta'); "
            "assert len(r['components'][0]['metric_deltas']) == 32768"
        )
        try:
            process = subprocess.run(
                [sys.executable, "-c", code, str(folder), str(self.summary_path)],
                text=True, capture_output=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            self.fail("faithfulness exceeded 10 seconds for only 32768 dimensions")
        self.assertEqual(process.returncode, 0, process.stderr)
