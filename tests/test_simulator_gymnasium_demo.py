import tempfile
import unittest
from pathlib import Path

from robotprobe.integrations.simulators.gymnasium_cartpole import (
    action_flip_postprocess,
    run_episode_trace,
)
from robotprobe.summarize import summarize_trace_pair
from robotsec_range.challenge_spec import load_builtin_challenge
from robotsec_range.checker import check_summary


ROOT = Path(__file__).resolve().parents[1]


class FakeDiscreteActionSpace:
    n = 2


class FakeGymnasiumEnv:
    action_space = FakeDiscreteActionSpace()

    def __init__(self):
        self.step_calls = 0

    def reset(self, *, seed=None):
        self.step_calls = 0
        return [0.0, 0.0, 0.01, 0.0], {"seed": seed}

    def step(self, action):
        self.step_calls += 1
        sign = 1.0 if action else -1.0
        obs = [float(self.step_calls), sign, 0.01 * sign, float(action)]
        terminated = self.step_calls >= 6
        return obs, 1.0, terminated, False, {"action": action}

    def close(self):
        pass


def angle_policy(observation):
    return 1 if observation[2] >= 0 else 0


class SimulatorGymnasiumDemoTests(unittest.TestCase):
    def test_action_flip_postprocess_flips_discrete_action_inside_window(self):
        self.assertEqual(action_flip_postprocess(0, action_space_n=2), 1)
        self.assertEqual(action_flip_postprocess(1, action_space_n=2), 0)
        self.assertEqual(action_flip_postprocess(2, action_space_n=3), 0)

    def test_fake_gymnasium_episode_pair_produces_robotprobe_diagnosis(self):
        manifest = ROOT / "interventions" / "action_flip_discrete.yaml"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            clean = out / "clean.jsonl"
            perturbed = out / "action_flip.jsonl"

            run_episode_trace(
                env=FakeGymnasiumEnv(),
                policy=angle_policy,
                trace_path=clean,
                run_id="sim_clean",
                scenario="clean",
                target="fake-gymnasium",
                seed=7,
                max_steps=6,
            )
            run_episode_trace(
                env=FakeGymnasiumEnv(),
                policy=angle_policy,
                trace_path=perturbed,
                run_id="sim_action_flip",
                scenario="action_flip_discrete",
                target="fake-gymnasium",
                seed=7,
                max_steps=6,
                baseline_run_id="sim_clean",
                perturbation_group_id="sim_pair_action_flip",
                intervention_manifest_ref=manifest,
                fault_start_step=2,
            )

            summary = summarize_trace_pair(clean, perturbed, intervention_manifest_ref=manifest)

        self.assertEqual(summary["root_cause_hypothesis"]["layer"], "action_postprocess")
        self.assertEqual(summary["root_cause_hypothesis"]["contract"], "DiscreteActionContract")
        self.assertEqual(summary["root_cause_hypothesis"]["fault"], "action_flip_discrete")
        self.assertFalse(summary["evidence"]["raw_policy_action_changed"])
        self.assertTrue(summary["evidence"]["postprocessed_action_changed"])
        self.assertEqual(summary["evidence"]["first_divergence_step"], 2)
        challenge = load_builtin_challenge("robotprobe-v1-gymnasium-cartpole-action-flip")
        self.assertTrue(check_summary(challenge, summary).passed)


if __name__ == "__main__":
    unittest.main()
