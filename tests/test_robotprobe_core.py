import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.core import ReplayLevel, load_intervention_manifest
from robotprobe.hooks import HookContext, HookRegistry
from robotprobe.record import TraceRecorder
from robotprobe.summarize import summarize_trace_pair
from robotsec_range.challenge_spec import ChallengeSpec
from robotsec_range.checker import check_summary


ROOT = Path(__file__).resolve().parents[1]


class RobotProbeCoreTests(unittest.TestCase):
    def test_hook_registry_applies_hooks_in_deterministic_order(self):
        registry = HookRegistry()
        calls = []

        @registry.hook("instruction", order=20)
        def suffix(ctx: HookContext, text: str) -> str:
            calls.append(("suffix", ctx.step))
            return f"{text} now"

        @registry.hook("instruction", order=10)
        def replace(ctx: HookContext, _text: str) -> str:
            calls.append(("replace", ctx.step))
            return "pick up the blue block"

        out = registry.apply("instruction", "pick up the red block", HookContext(step=7))

        self.assertEqual(out, "pick up the blue block now")
        self.assertEqual(calls, [("replace", 7), ("suffix", 7)])

    def test_intervention_manifest_is_loaded_as_reproducible_contract(self):
        manifest = load_intervention_manifest(ROOT / "interventions" / "gripper_freeze_mid.yaml")

        self.assertEqual(manifest.id, "ACTION.GRIPPER.FREEZE.MID.001")
        self.assertEqual(manifest.name, "gripper_freeze_mid")
        self.assertEqual(manifest.contract, "ActionSpaceContract")
        self.assertEqual(manifest.target_layers, ["action_postprocess"])
        self.assertEqual(manifest.window, {"start_step": 35, "end_step": 120})
        self.assertFalse(manifest.safety["hardware_allowed"])
        self.assertTrue(manifest.digest.startswith("sha256:"))

    def test_trace_recorder_writes_run_pair_and_contract_check_events(self):
        manifest_ref = ROOT / "interventions" / "gripper_freeze_mid.yaml"
        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "gripper_freeze.jsonl"

            with TraceRecorder(
                trace_path,
                run_id="rp_run_gripper_001",
                target="pi05-libero",
                scenario="gripper_freeze_mid",
                baseline_run_id="rp_run_clean_001",
                perturbation_group_id="rp_pair_gripper_001",
                intervention_manifest_ref=manifest_ref,
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(
                    step=35,
                    event_type="postprocessed_action",
                    layer="action_postprocess",
                    value=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
                )
                recorder.contract_check(
                    step=35,
                    contract="ActionSpaceContract",
                    check="raw_vs_postprocessed_gripper_freshness",
                    status="violated",
                    severity=0.68,
                    evidence_refs=["actions/raw/000035.npy", "actions/post/000035.npy"],
                )

            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertEqual(events[0]["run_id"], "rp_run_gripper_001")
            self.assertEqual(events[0]["baseline_run_id"], "rp_run_clean_001")
            self.assertEqual(events[0]["perturbation_group_id"], "rp_pair_gripper_001")
            self.assertEqual(events[0]["intervention_manifest_ref"], str(manifest_ref))
            self.assertEqual(events[0]["intervention_manifest_hash"], load_intervention_manifest(manifest_ref).digest)
            self.assertNotIn("expected_layer_of_effect", events[0])
            self.assertNotIn("expected_downstream_layers", events[0])
            self.assertEqual(events[1]["event_type"], "contract_check")
            self.assertEqual(events[1]["status"], "violated")
            self.assertEqual(events[1]["severity"], 0.68)

    def test_causal_summary_identifies_post_policy_action_fault(self):
        manifest_ref = ROOT / "interventions" / "gripper_freeze_mid.yaml"
        with tempfile.TemporaryDirectory() as td:
            clean_path = Path(td) / "clean.jsonl"
            perturbed_path = Path(td) / "gripper_freeze.jsonl"
            clean_action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            frozen_action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]

            with TraceRecorder(
                clean_path,
                run_id="rp_run_clean_001",
                target="pi05-libero",
                scenario="clean",
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(35, "raw_action", "policy_forward", value=clean_action)
                recorder.event(35, "postprocessed_action", "action_postprocess", value=clean_action)
                recorder.event(153, "outcome", "outcome", success=True, steps=77)

            with TraceRecorder(
                perturbed_path,
                run_id="rp_run_gripper_001",
                target="pi05-libero",
                scenario="gripper_freeze_mid",
                baseline_run_id="rp_run_clean_001",
                perturbation_group_id="rp_pair_gripper_001",
                intervention_manifest_ref=manifest_ref,
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(35, "raw_action", "policy_forward", value=clean_action)
                recorder.event(35, "postprocessed_action", "action_postprocess", value=frozen_action)
                recorder.event(153, "outcome", "outcome", success=True, steps=153)

            summary = summarize_trace_pair(clean_path, perturbed_path, intervention_manifest_ref=manifest_ref)

        self.assertEqual(summary["root_cause_hypothesis"]["layer"], "action_postprocess")
        self.assertEqual(summary["root_cause_hypothesis"]["contract"], "ActionSpaceContract")
        self.assertEqual(summary["root_cause_hypothesis"]["fault"], "gripper_freeze_mid")
        self.assertAlmostEqual(summary["root_cause_hypothesis"]["rule_match_score"], 0.87)
        self.assertFalse(summary["evidence"]["raw_policy_action_changed"])
        self.assertTrue(summary["evidence"]["postprocessed_action_changed"])
        self.assertEqual(summary["evidence"]["first_divergence_step"], 35)
        self.assertFalse(summary["evidence"]["outcome_delta"]["success_changed"])
        self.assertAlmostEqual(summary["evidence"]["outcome_delta"]["step_inflation"], 153 / 77)
        self.assertEqual(summary["replay_level"], "L0_TRACE_PLAYBACK")

    def test_action_postprocess_summary_uses_intra_trace_action_mutation_not_cross_run_drift(self):
        manifest_ref = ROOT / "interventions" / "gripper_freeze_mid.yaml"
        with tempfile.TemporaryDirectory() as td:
            clean_path = Path(td) / "clean.jsonl"
            perturbed_path = Path(td) / "gripper_freeze.jsonl"

            with TraceRecorder(
                clean_path,
                run_id="rp_run_clean_001",
                target="pi05-libero",
                scenario="clean",
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(0, "raw_action", "policy_forward", value=[[1.0, 0.0, 0.7]])
                recorder.event(0, "postprocessed_action", "action_postprocess", value=[[1.0, 0.0, 0.7]])
                recorder.event(35, "raw_action", "policy_forward", value=[[1.0, 0.0, 0.8]])
                recorder.event(35, "postprocessed_action", "action_postprocess", value=[[1.0, 0.0, 0.8]])
                recorder.event(74, "outcome", "outcome", success=True, steps=74)

            with TraceRecorder(
                perturbed_path,
                run_id="rp_run_gripper_001",
                target="pi05-libero",
                scenario="gripper_freeze_mid",
                baseline_run_id="rp_run_clean_001",
                perturbation_group_id="rp_pair_gripper_001",
                intervention_manifest_ref=manifest_ref,
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                # Real rollout smoke showed cross-run raw drift at step 0. This must not be
                # interpreted as the intervention mutating raw policy output.
                recorder.event(0, "raw_action", "policy_forward", value=[[2.0, 0.0, 0.4]])
                recorder.event(0, "postprocessed_action", "action_postprocess", value=[[2.0, 0.0, 0.4]])
                recorder.event(35, "raw_action", "policy_forward", value=[[2.0, 0.0, 0.2]])
                recorder.event(35, "postprocessed_action", "action_postprocess", value=[[2.0, 0.0, 0.4]])
                recorder.event(154, "outcome", "outcome", success=True, steps=154)

            summary = summarize_trace_pair(clean_path, perturbed_path, intervention_manifest_ref=manifest_ref)

        self.assertFalse(summary["evidence"]["raw_policy_action_changed"])
        self.assertTrue(summary["evidence"]["postprocessed_action_changed"])
        self.assertEqual(summary["evidence"]["first_divergence_step"], 35)
        self.assertAlmostEqual(summary["root_cause_hypothesis"]["rule_match_score"], 0.87)

    def test_instruction_summary_uses_intra_trace_instruction_change(self):
        manifest_ref = ROOT / "interventions" / "instruction_swap.yaml"
        with tempfile.TemporaryDirectory() as td:
            clean_path = Path(td) / "clean.jsonl"
            perturbed_path = Path(td) / "instruction_swap.jsonl"

            with TraceRecorder(
                clean_path,
                run_id="rp_run_clean_001",
                target="pi05-libero",
                scenario="clean",
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(0, "instruction", "instruction", before="red", after="red", value="red")
                recorder.event(20, "outcome", "outcome", success=True, steps=20)

            with TraceRecorder(
                perturbed_path,
                run_id="rp_run_instruction_001",
                target="pi05-libero",
                scenario="instruction_swap",
                baseline_run_id="rp_run_clean_001",
                perturbation_group_id="rp_pair_instruction_001",
                intervention_manifest_ref=manifest_ref,
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(0, "instruction", "instruction", before="red", after="blue", value="blue")
                recorder.event(30, "outcome", "outcome", success=True, steps=30)

            summary = summarize_trace_pair(clean_path, perturbed_path, intervention_manifest_ref=manifest_ref)

        self.assertTrue(summary["evidence"]["instruction_changed"])
        self.assertEqual(summary["evidence"]["first_divergence_step"], 0)
        self.assertEqual(summary["root_cause_hypothesis"]["layer"], "instruction")
        self.assertAlmostEqual(summary["root_cause_hypothesis"]["rule_match_score"], 0.87)

    def test_observation_summary_detects_stale_replay_inside_perturbed_trace(self):
        manifest_ref = ROOT / "interventions" / "stale_observation_replay.yaml"
        with tempfile.TemporaryDirectory() as td:
            clean_path = Path(td) / "clean.jsonl"
            perturbed_path = Path(td) / "stale.jsonl"

            with TraceRecorder(
                clean_path,
                run_id="rp_run_clean_001",
                target="pi05-libero",
                scenario="clean",
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(20, "outcome", "outcome", success=True, steps=20)

            with TraceRecorder(
                perturbed_path,
                run_id="rp_run_stale_001",
                target="pi05-libero",
                scenario="stale_observation_replay",
                baseline_run_id="rp_run_clean_001",
                perturbation_group_id="rp_pair_stale_001",
                intervention_manifest_ref=manifest_ref,
                replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as recorder:
                recorder.event(0, "observation", "observation", image_sha256="sha256:a")
                recorder.event(1, "observation", "observation", image_sha256="sha256:b")
                recorder.event(2, "observation", "observation", image_sha256="sha256:b")
                recorder.event(30, "outcome", "outcome", success=True, steps=30)

            summary = summarize_trace_pair(clean_path, perturbed_path, intervention_manifest_ref=manifest_ref)

        self.assertTrue(summary["evidence"]["observation_replayed"])
        self.assertEqual(summary["evidence"]["first_divergence_step"], 2)
        self.assertEqual(summary["root_cause_hypothesis"]["layer"], "observation")
        self.assertAlmostEqual(summary["root_cause_hypothesis"]["rule_match_score"], 0.87)

    def test_challenge_checker_validates_diagnosis_evidence_not_flags(self):
        challenge = ChallengeSpec(
            id="challenge-003",
            diagnosis={
                "layer": "action_postprocess",
                "contract": "ActionSpaceContract",
                "fault": "gripper_freeze_mid",
            },
            required_evidence={
                "raw_policy_action_changed": False,
                "postprocessed_action_changed": True,
                "min_step_inflation": 1.5,
            },
        )
        summary = {
            "root_cause_hypothesis": {
                "layer": "action_postprocess",
                "contract": "ActionSpaceContract",
                "fault": "gripper_freeze_mid",
                "rule_match_score": 0.87,
            },
            "evidence": {
                "raw_policy_action_changed": False,
                "postprocessed_action_changed": True,
                "outcome_delta": {"step_inflation": 1.99},
            },
        }

        result = check_summary(challenge, summary)

        self.assertTrue(result.passed)
        self.assertEqual(result.diagnosis["contract"], "ActionSpaceContract")
        self.assertAlmostEqual(result.rule_match_score, 0.87)

    def test_robotprobe_summarize_cli_outputs_causal_diff_json(self):
        manifest_ref = ROOT / "interventions" / "gripper_freeze_mid.yaml"
        with tempfile.TemporaryDirectory() as td:
            clean_path, perturbed_path = self._write_gripper_trace_pair(Path(td), manifest_ref)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "summarize",
                    "--clean",
                    str(clean_path),
                    "--perturbed",
                    str(perturbed_path),
                    "--manifest",
                    str(manifest_ref),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

        summary = json.loads(proc.stdout)
        self.assertEqual(summary["root_cause_hypothesis"]["layer"], "action_postprocess")
        self.assertEqual(summary["evidence"]["first_divergence_step"], 35)

    def test_robotsec_range_check_cli_outputs_diagnosis_result(self):
        challenge = {
            "id": "challenge-003",
            "diagnosis": {
                "layer": "action_postprocess",
                "contract": "ActionSpaceContract",
                "fault": "gripper_freeze_mid",
            },
            "required_evidence": {
                "raw_policy_action_changed": False,
                "postprocessed_action_changed": True,
                "min_step_inflation": 1.5,
            },
        }
        summary = {
            "root_cause_hypothesis": {
                "layer": "action_postprocess",
                "contract": "ActionSpaceContract",
                "fault": "gripper_freeze_mid",
                "rule_match_score": 0.87,
            },
            "evidence": {
                "raw_policy_action_changed": False,
                "postprocessed_action_changed": True,
                "outcome_delta": {"step_inflation": 1.99},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            challenge_path = Path(td) / "challenge.json"
            summary_path = Path(td) / "summary.json"
            challenge_path.write_text(json.dumps(challenge), encoding="utf-8")
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotsec_range",
                    "check",
                    "--challenge",
                    str(challenge_path),
                    "--summary",
                    str(summary_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

        result = json.loads(proc.stdout)
        self.assertTrue(result["passed"])
        self.assertEqual(result["diagnosis"]["fault"], "gripper_freeze_mid")

    def _write_gripper_trace_pair(self, directory: Path, manifest_ref: Path):
        clean_path = directory / "clean.jsonl"
        perturbed_path = directory / "gripper_freeze.jsonl"
        clean_action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        frozen_action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]

        with TraceRecorder(
            clean_path,
            run_id="rp_run_clean_001",
            target="pi05-libero",
            scenario="clean",
            replay_level=ReplayLevel.TRACE_PLAYBACK,
        ) as recorder:
            recorder.event(35, "raw_action", "policy_forward", value=clean_action)
            recorder.event(35, "postprocessed_action", "action_postprocess", value=clean_action)
            recorder.event(153, "outcome", "outcome", success=True, steps=77)

        with TraceRecorder(
            perturbed_path,
            run_id="rp_run_gripper_001",
            target="pi05-libero",
            scenario="gripper_freeze_mid",
            baseline_run_id="rp_run_clean_001",
            perturbation_group_id="rp_pair_gripper_001",
            intervention_manifest_ref=manifest_ref,
            replay_level=ReplayLevel.TRACE_PLAYBACK,
        ) as recorder:
            recorder.event(35, "raw_action", "policy_forward", value=clean_action)
            recorder.event(35, "postprocessed_action", "action_postprocess", value=frozen_action)
            recorder.event(153, "outcome", "outcome", success=True, steps=153)

        return clean_path, perturbed_path


if __name__ == "__main__":
    unittest.main()
