import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.integrations.vla_eval.robotprobe_adapter import (
    RobotProbeVlaEvalConfig,
    RobotProbeVlaEvalRunner,
    RobotProbeSyncEpisodeRunner,
)
from robotprobe.attacks import ActionFaultAttack


ROOT = Path(__file__).resolve().parents[1]


class VlaEvalRunnerTests(unittest.TestCase):
    def test_runner_wraps_episode_with_observation_action_traces_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = ROOT / "interventions" / "gripper_freeze_mid.yaml"
            config = RobotProbeVlaEvalConfig(
                trace_dir=Path(td) / "traces",
                result_dir=Path(td) / "results",
                target="vla-eval/libero",
                scenario="gripper_freeze_mid",
                perturbation_group_id="rp_pair_vla_eval_001",
                intervention_manifest_ref=manifest,
                challenge="robotprobe-v0-gripper-freeze",
                metrics=("action_energy", "trajectory_deviation"),
            )
            runner = RobotProbeVlaEvalRunner(config)
            action_attack = ActionFaultAttack(
                name="gripper_freeze_mid",
                mode="hold_previous_dims",
                dims=(2,),
                start=1,
                end=3,
            )

            result = runner.run_pair(
                clean_episode=lambda episode: _fake_episode(episode, steps=4),
                perturbed_episode=lambda episode: _fake_episode(episode, steps=7, action_attack=action_attack),
            )

            self.assertTrue(result.clean_trace.exists())
            self.assertTrue(result.perturbed_trace.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.check_path.exists())
            self.assertTrue(result.report_path.exists())
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["root_cause_hypothesis"]["layer"], "action_postprocess")
            self.assertIn("metrics", summary)
            check = json.loads(result.check_path.read_text(encoding="utf-8"))
        self.assertTrue(check["passed"])

    def test_sync_episode_runner_records_actual_vla_eval_loop_shape(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = ROOT / "interventions" / "gripper_freeze_mid.yaml"
            config = RobotProbeVlaEvalConfig(
                trace_dir=Path(td) / "traces",
                result_dir=Path(td) / "results",
                target="vla-eval/sync-runner",
                scenario="gripper_freeze_mid",
                perturbation_group_id="rp_pair_sync_runner_001",
                intervention_manifest_ref=manifest,
            )
            action_attack = ActionFaultAttack(
                name="gripper_freeze_mid",
                mode="hold_previous_dims",
                dims=(2,),
                start=1,
                end=3,
            )
            runner = RobotProbeSyncEpisodeRunner(config, action_attack=action_attack)

            result = _run_async(
                runner.run_episode(
                    _AsyncStubBenchmark(done_at_step=3),
                    {"name": "stub_task"},
                    _AsyncStubConnection(),
                    max_steps=10,
                )
            )

            self.assertTrue(result["metrics"]["success"])
            self.assertEqual(result["steps"], 3)
            self.assertTrue(runner.last_trace_path.exists())
            events = [json.loads(line) for line in runner.last_trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["event_type"], "instruction")
            self.assertTrue(any(event["event_type"] == "raw_action" for event in events))
            mutated = [
                event for event in events
                if event["event_type"] == "postprocessed_action" and event["step"] == 1
            ][0]
            self.assertAlmostEqual(mutated["value"][0][2], 0.7)


def _fake_episode(episode, *, steps: int, action_attack=None):
    action_seed = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    for step in range(3):
        obs = {
            "images": {"agentview": np.full((2, 2, 3), step, dtype=np.uint8)},
            "task_description": "pick up the red bowl",
            "state": np.zeros(7),
        }
        episode.pre_inference_observation(obs, step=step)
        raw_action = {"actions": action_seed + np.array([[step / 10, 0.0, step / 10]], dtype=np.float32)}
        episode.post_inference_action(raw_action, step=step, action_attack=action_attack)
    episode.record_outcome(success=True, steps=steps)


class _AsyncStubBenchmark:
    def __init__(self, *, done_at_step: int):
        self.done_at_step = done_at_step
        self.step = 0
        self.task = {}

    async def start_episode(self, task, recorder=None):
        self.task = dict(task)
        self.step = 0

    async def get_observation(self):
        return {
            "images": {"agentview": np.full((2, 2, 3), self.step, dtype=np.uint8)},
            "task_description": "pick up the red bowl",
            "state": np.array([self.step], dtype=np.float32),
        }

    async def apply_action(self, action):
        self.step += 1

    async def is_done(self):
        return self.step >= self.done_at_step

    async def get_time(self):
        return float(self.step) / 10.0

    async def get_result(self):
        return {"success": self.step >= self.done_at_step}


class _AsyncStubConnection:
    def __init__(self):
        self.started = False

    async def start_episode(self, payload):
        self.started = True

    async def act(self, obs):
        step = int(obs["state"][0])
        return {"actions": np.array([[0.1 + step, 0.2, 0.7 + step]], dtype=np.float32)}

    async def end_episode(self, result):
        self.ended = result


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
