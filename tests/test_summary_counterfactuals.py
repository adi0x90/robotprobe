import json
from pathlib import Path
import tempfile
import unittest

from robotprobe.summarize import summarize_trace_pair


class SummaryCounterfactualTests(unittest.TestCase):
    def summarize_outcomes(self, clean_success, perturbed_success):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for run, success in (("clean", clean_success), ("perturbed", perturbed_success)):
                path = Path(directory) / f"{run}.jsonl"
                outcome = {
                    "event_type": "outcome",
                    "layer": "outcome",
                    "run_id": run,
                    "replay_level": "L0_TRACE_PLAYBACK",
                    "step": 10,
                    "steps": 10,
                    "success": success,
                }
                path.write_text(json.dumps(outcome) + "\n", encoding="utf-8")
                paths.append(path)
            return summarize_trace_pair(*paths)

    def assert_recovery_is_untested(self, summary):
        # A clean/fault pair never contains a fault-removal run, so the summary must not
        # carry a counterfactual block that implies one was tested.
        self.assertNotIn("counterfactuals", summary)
        self.assertEqual(summary["replay_level"], "L0_TRACE_PLAYBACK")

    def test_two_failed_episodes_do_not_establish_fault_removal_recovery(self):
        summary = self.summarize_outcomes(False, False)

        self.assert_recovery_is_untested(summary)
        self.assertIs(summary["evidence"]["outcome_delta"]["success_changed"], False)

    def test_clean_success_and_fault_failure_are_contrast_not_recovery(self):
        summary = self.summarize_outcomes(True, False)

        self.assert_recovery_is_untested(summary)
        self.assertIs(summary["evidence"]["outcome_delta"]["success_changed"], True)
