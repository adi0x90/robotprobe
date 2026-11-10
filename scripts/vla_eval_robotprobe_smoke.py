from __future__ import annotations

import argparse
import asyncio
import json
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

from robotprobe.integrations.vla_eval.robotprobe_adapter import RobotProbeSyncEpisodeRunner, RobotProbeVlaEvalConfig
from robotprobe.attacks import ActionFaultAttack
from robotprobe.summarize import summarize_trace_pair


class SmokeEchoServer:
    def __new__(cls):
        from vla_eval.model_servers.predict import PredictModelServer

        class _SmokeEchoServer(PredictModelServer):
            def predict(self, obs: dict[str, Any], ctx: Any) -> dict[str, Any]:
                step = int(np.asarray(obs.get("state", [0]))[0])
                return {"actions": np.array([[0.1 + step, 0.2, 0.7 + step]], dtype=np.float32)}

        return _SmokeEchoServer()


class SmokeBenchmark:
    def __new__(cls, *, done_at_step: int):
        from vla_eval.benchmarks.base import StepBenchmark, StepResult

        class _SmokeBenchmark(StepBenchmark):
            def __init__(self, *, done_at_step: int):
                super().__init__()
                self.done_at_step = done_at_step
                self.step_count = 0

            def get_tasks(self) -> list[dict[str, Any]]:
                return [{"name": "robotprobe_smoke"}]

            def reset(self, task: dict[str, Any]) -> Any:
                self.step_count = 0
                return None

            def step(self, action: dict[str, Any]) -> StepResult:
                self.step_count += 1
                done = self.step_count >= self.done_at_step
                return StepResult(obs=None, reward=1.0 if done else 0.0, done=done, info={})

            def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
                return {
                    "images": {
                        "agentview": np.full((8, 8, 3), self.step_count, dtype=np.uint8),
                        "wrist": np.zeros((8, 8, 3), dtype=np.uint8),
                    },
                    "task_description": "pick up the red bowl",
                    "state": np.array([self.step_count], dtype=np.float32),
                }

            def get_step_result(self, step_result: Any) -> dict[str, Any]:
                return {"success": bool(step_result.done)}

        return _SmokeBenchmark(done_at_step=done_at_step)


async def run_smoke(out_dir: Path, manifest: Path) -> dict[str, Any]:
    from vla_eval.connection import Connection
    from vla_eval.model_servers.serve import serve_async

    out_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    server = SmokeEchoServer()
    server_task = asyncio.create_task(serve_async(server, host="127.0.0.1", port=port))
    await _wait_for_tcp(port)

    try:
        clean_config = RobotProbeVlaEvalConfig(
            trace_dir=out_dir / "traces",
            result_dir=out_dir / "results",
            target="allenai-vla-eval/smoke",
            scenario="clean",
            perturbation_group_id="rp_vla_eval_smoke_pair",
            clean_run_id="rp_vla_eval_smoke_clean",
            perturbed_run_id="rp_vla_eval_smoke_clean",
        )
        perturbed_config = RobotProbeVlaEvalConfig(
            trace_dir=out_dir / "traces",
            result_dir=out_dir / "results",
            target="allenai-vla-eval/smoke",
            scenario="gripper_freeze_mid",
            perturbation_group_id="rp_vla_eval_smoke_pair",
            intervention_manifest_ref=manifest,
            clean_run_id="rp_vla_eval_smoke_clean_robotprobe_smoke",
            perturbed_run_id="rp_vla_eval_smoke_gripper",
        )
        action_attack = ActionFaultAttack(
            name="gripper_freeze_mid",
            mode="hold_previous_dims",
            dims=(2,),
            start=1,
            end=3,
        )

        async with Connection(f"ws://127.0.0.1:{port}") as conn:
            clean_runner = RobotProbeSyncEpisodeRunner(clean_config)
            clean_result = await clean_runner.run_episode(
                SmokeBenchmark(done_at_step=3),
                {"name": "robotprobe_smoke"},
                conn,
                max_steps=10,
            )

        async with Connection(f"ws://127.0.0.1:{port}") as conn:
            perturbed_runner = RobotProbeSyncEpisodeRunner(perturbed_config, action_attack=action_attack)
            perturbed_result = await perturbed_runner.run_episode(
                SmokeBenchmark(done_at_step=3),
                {"name": "robotprobe_smoke"},
                conn,
                max_steps=10,
            )

        assert clean_runner.last_trace_path is not None
        assert perturbed_runner.last_trace_path is not None
        summary = summarize_trace_pair(
            clean_runner.last_trace_path,
            perturbed_runner.last_trace_path,
            intervention_manifest_ref=manifest,
            metrics=["action_energy", "jerk", "trajectory_deviation", "recovery_latency", "near_miss"],
        )
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "clean_result": clean_result,
            "perturbed_result": perturbed_result,
            "clean_trace": str(clean_runner.last_trace_path),
            "perturbed_trace": str(perturbed_runner.last_trace_path),
            "summary": str(summary_path),
            "root_layer": summary["root_cause_hypothesis"]["layer"],
            "first_divergence_step": summary["evidence"]["first_divergence_step"],
            "postprocessed_action_changed": summary["evidence"]["postprocessed_action_changed"],
        }
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_tcp(port: int) -> None:
    import websockets

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}"):
                pass
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise TimeoutError(f"server did not open port {port}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default="interventions/gripper_freeze_mid.yaml")
    args = parser.parse_args()
    result = asyncio.run(run_smoke(Path(args.out), Path(args.manifest)))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
