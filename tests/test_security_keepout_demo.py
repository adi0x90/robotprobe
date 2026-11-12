import json
import tempfile
import unittest
from pathlib import Path

from robotprobe.conformance import check_conformance_file
from robotprobe.integrations.simulators.security_keepout import KeepoutConfig, run_keepout_episode
from robotprobe.replay import build_replay_plan
from robotprobe.summarize import summarize_trace_pair
from robotsec_range.challenge_spec import load_builtin_challenge
from robotsec_range.checker import check_summary


ROOT = Path(__file__).resolve().parents[1]


class SecurityKeepoutDemoTests(unittest.TestCase):
    def test_trace_records_actual_seed_and_configuration_for_reproduction(self):
        config = KeepoutConfig(start=(0.1, 0.2), dt=0.025, max_speed=0.3, bias_y=-0.4)
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "original.jsonl"
            replayed = Path(directory) / "replayed.jsonl"
            run_keepout_episode(
                trace_path=trace,
                run_id="metadata-reproduction",
                scenario="custom_bias",
                seed=123,
                max_steps=12,
                bias_start_step=2,
                bias_end_step=4,
                config=config,
            )
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertTrue(all(event.get("seed") == 123 for event in events))
            plan = build_replay_plan(trace, level="L2")
            self.assertEqual(plan["sim_seed"], 123)
            self.assertTrue(plan["deterministic_replay_available"])
            recorded = plan["sim_config"]
            self.assertEqual(recorded["config"]["dt"], 0.025)
            self.assertEqual(recorded["config"]["start"], [0.1, 0.2])
            self.assertEqual(recorded["max_steps"], 12)
            self.assertEqual(recorded["bias_start_step"], 2)
            self.assertEqual(recorded["bias_end_step"], 4)
            run_keepout_episode(
                trace_path=replayed,
                run_id="metadata-reproduction",
                scenario="custom_bias",
                seed=plan["sim_seed"],
                max_steps=recorded["max_steps"],
                bias_start_step=recorded["bias_start_step"],
                bias_end_step=recorded["bias_end_step"],
                config=KeepoutConfig(**recorded["config"]),
            )
            self.assertEqual(events, [json.loads(line) for line in replayed.read_text().splitlines()])
            conformance = check_conformance_file(trace)
            seed_check = next(
                check for check in conformance.checks if check.requirement == "run_seed_present"
            )
            self.assertTrue(seed_check.passed)

    def test_keepout_bias_attack_succeeds_but_violates_contract(self):
        manifest = ROOT / "interventions" / "stealth_velocity_bias_keepout.yaml"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            clean = out / "clean.jsonl"
            perturbed = out / "perturbed.jsonl"

            clean_result = run_keepout_episode(
                trace_path=clean,
                run_id="keepout_clean",
                scenario="clean",
                seed=11,
                max_steps=80,
            )
            perturbed_result = run_keepout_episode(
                trace_path=perturbed,
                run_id="keepout_bias",
                scenario="stealth_velocity_bias_keepout",
                seed=11,
                max_steps=80,
                baseline_run_id="keepout_clean",
                perturbation_group_id="keepout_pair_bias",
                intervention_manifest_ref=manifest,
                bias_start_step=10,
                bias_end_step=28,
            )

            summary = summarize_trace_pair(clean, perturbed, intervention_manifest_ref=manifest)
            challenge = load_builtin_challenge("robotprobe-v1-security-keepout-bypass")
            check_result = check_summary(
                challenge,
                summary,
                clean_trace=clean,
                perturbed_trace=perturbed,
                intervention_manifest_ref=manifest,
            )
            self.assertTrue(check_conformance_file(clean, target_level="L1_REPRODUCIBLE_INCIDENT").conformant)
            self.assertTrue(
                check_conformance_file(perturbed, target_level="L2_CONTRACT_CHECKABLE").conformant
            )

        self.assertTrue(clean_result["success"])
        self.assertTrue(perturbed_result["success"])
        self.assertFalse(clean_result["keepout_violated"])
        self.assertTrue(perturbed_result["keepout_violated"])
        self.assertFalse(summary["evidence"]["raw_policy_action_changed"])
        self.assertTrue(summary["evidence"]["postprocessed_action_changed"])
        self.assertTrue(summary["evidence"]["contract_violated"])
        self.assertTrue(summary["evidence"]["perturbed_keepout_violated"])
        self.assertFalse(summary["evidence"]["outcome_delta"]["success_changed"])
        self.assertEqual(summary["root_cause_hypothesis"]["layer"], "action_postprocess")
        self.assertEqual(summary["root_cause_hypothesis"]["contract"], "KeepoutZoneContract")
        self.assertEqual(summary["root_cause_hypothesis"]["fault"], "stealth_velocity_bias_keepout")

        self.assertTrue(check_result.passed)


if __name__ == "__main__":
    unittest.main()
