import json
from pathlib import Path
import tempfile
import unittest

from robotprobe.summarize import summarize_trace_pair


ROOT = Path(__file__).resolve().parents[1]


class SummaryInstructionTests(unittest.TestCase):
    def summarize_instruction(self, instruction, *, clean_steps=2, perturbed_steps=2, manifest=None):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for run, steps, fields in (
                ("clean", clean_steps, {"value": "pick up the bowl"}),
                ("perturbed", perturbed_steps, instruction),
            ):
                events = [{"event_type": "instruction", "step": 0, **fields}]
                events.extend(
                    {"event_type": "raw_action", "step": step, "value": [0.25]}
                    for step in range(steps)
                )
                events.append({"event_type": "outcome", "step": steps, "steps": steps, "success": True})
                path = Path(directory) / f"{run}.jsonl"
                path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
                paths.append(path)
            return summarize_trace_pair(*paths, intervention_manifest_ref=manifest)

    def test_plain_instruction_declaration_is_not_a_mutation(self):
        for clean_steps, perturbed_steps in ((2, 2), (74, 75)):
            with self.subTest(clean_steps=clean_steps, perturbed_steps=perturbed_steps):
                summary = self.summarize_instruction(
                    {"value": "pick up the bowl"},
                    clean_steps=clean_steps,
                    perturbed_steps=perturbed_steps,
                )

                self.assertIs(summary["evidence"]["instruction_changed"], False)
                self.assertIs(summary["evidence"]["raw_policy_drift_vs_clean"], False)

    def test_unknown_before_or_after_does_not_establish_a_mutation(self):
        for instruction in (
            {"after": "pick up the bowl"},
            {"before": None, "after": "pick up the bowl"},
            {"before": "pick up the bowl"},
            {"before": "pick up the bowl", "after": None},
        ):
            with self.subTest(instruction=instruction):
                summary = self.summarize_instruction(instruction)

                self.assertIs(summary["evidence"]["instruction_changed"], False)

    def test_explicit_unchanged_instruction_is_not_a_mutation(self):
        summary = self.summarize_instruction({"before": "pick up the bowl", "after": "pick up the bowl"})

        self.assertIs(summary["evidence"]["instruction_changed"], False)

    def test_known_before_after_or_value_mutation_remains_detected(self):
        for instruction in (
            {"before": "pick up the bowl", "after": "pick up the plate"},
            {"before": "pick up the bowl", "value": "pick up the plate"},
            {"before": "", "after": "pick up the bowl"},
        ):
            with self.subTest(instruction=instruction):
                summary = self.summarize_instruction(
                    instruction, manifest=ROOT / "interventions" / "instruction_swap.yaml"
                )

                self.assertIs(summary["evidence"]["instruction_changed"], True)
                self.assertEqual(summary["evidence"]["first_divergence_step"], 0)
                self.assertEqual(summary["root_cause_hypothesis"]["layer"], "instruction")
                self.assertEqual(summary["root_cause_hypothesis"]["rule_match_score"], 0.87)
