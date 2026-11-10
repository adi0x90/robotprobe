from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from robotprobe.integrations.vla_eval.robotprobe_adapter import RobotProbeSyncEpisodeRunner, RobotProbeVlaEvalConfig
from robotprobe.attacks import ActionFaultAttack
from robotprobe.summarize import summarize_trace_pair


@dataclass(frozen=True)
class VlaEvalServeSpec:
    config_path: Path
    cwd: Path
    port: int
    host: str = "127.0.0.1"
    vla_eval_executable: str = "vla-eval"
    arg_overrides: tuple[str, ...] = ()


def build_serve_command(spec: VlaEvalServeSpec) -> list[str]:
    command = [
        spec.vla_eval_executable,
        "serve",
        "--config",
        str(spec.config_path),
        "--arg",
        f"port={spec.port}",
        "--arg",
        f"host={spec.host}",
    ]
    for override in spec.arg_overrides:
        command.extend(["--arg", override])
    return command


async def run_real_model_smoke(
    *,
    model_name: str,
    server_config: Path,
    vla_eval_cwd: Path,
    out_dir: Path,
    manifest: Path,
    vla_eval_executable: str = "vla-eval",
    host: str = "127.0.0.1",
    port: int | None = None,
    startup_timeout: float = 900.0,
    episode_timeout: float = 120.0,
    max_steps: int = 3,
    arg_overrides: Sequence[str] = (),
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen_port = port or free_port(host)
    spec = VlaEvalServeSpec(
        config_path=server_config,
        cwd=vla_eval_cwd,
        host=host,
        port=chosen_port,
        vla_eval_executable=vla_eval_executable,
        arg_overrides=tuple(arg_overrides),
    )
    command = build_serve_command(spec)
    log_path = out_dir / "model-server.log"

    proc: subprocess.Popen | None = None
    try:
        proc = start_server_process(
            command,
            cwd=vla_eval_cwd,
            log_path=log_path,
            env=_process_env(env_overrides),
        )
        await wait_for_tcp(host, chosen_port, proc=proc, timeout=startup_timeout)

        artifacts = await _run_robotprobe_pair(
            model_name=model_name,
            host=host,
            port=chosen_port,
            out_dir=out_dir,
            manifest=manifest,
            episode_timeout=episode_timeout,
            max_steps=max_steps,
        )
        result_path = write_attempt_result(
            out_dir=out_dir,
            status="passed",
            model_name=model_name,
            server_config=server_config,
            vla_eval_cwd=vla_eval_cwd,
            command=command,
            log_path=log_path,
            artifacts=artifacts,
        )
        result = _read_json(result_path)
        result["result_path"] = str(result_path)
        return result
    except Exception as exc:
        stage = "startup"
        if proc is not None and proc.poll() is None and is_tcp_open(host, chosen_port):
            stage = "episode"
        result_path = write_attempt_result(
            out_dir=out_dir,
            status="failed",
            model_name=model_name,
            server_config=server_config,
            vla_eval_cwd=vla_eval_cwd,
            command=command,
            log_path=log_path,
            stage=stage,
            error=str(exc),
        )
        result = _read_json(result_path)
        result["result_path"] = str(result_path)
        return result
    finally:
        if proc is not None:
            terminate_process(proc)


def write_attempt_result(
    *,
    out_dir: Path,
    status: str,
    model_name: str,
    server_config: Path,
    vla_eval_cwd: Path,
    command: Sequence[str],
    log_path: Path,
    stage: str | None = None,
    error: str | None = None,
    artifacts: dict[str, Any] | None = None,
    log_tail_lines: int = 80,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": status,
        "model_server": {
            "name": model_name,
            "config": str(server_config),
            "vla_eval_cwd": str(vla_eval_cwd),
        },
        "command": list(command),
        "server_log": str(log_path),
        "server_log_tail": tail_lines(log_path, log_tail_lines),
        "artifacts": artifacts or {},
    }
    if status != "passed":
        result["failure"] = {"stage": stage, "error": error}
    result_path = out_dir / "model-server-smoke-result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_path


async def wait_for_tcp(
    host: str,
    port: int,
    *,
    proc: subprocess.Popen | None = None,
    timeout: float = 120.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"model server exited with code {proc.returncode} before opening {host}:{port}")
        if is_tcp_open(host, port):
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(f"model server did not open {host}:{port} within {timeout:.1f}s")


def is_tcp_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def start_server_process(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            start_new_session=True,
        )
        # Keep the owned group identity even if its leader exits before cleanup.
        proc._robotprobe_process_group_id = proc.pid
        return proc
    finally:
        log.close()


def terminate_process(proc: subprocess.Popen, *, timeout: float = 20.0) -> None:
    pgid = _process_group_id(proc)
    if pgid is not None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            try:
                _wait_for_process_group(proc, pgid, timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                if sig == signal.SIGKILL:
                    raise
        proc.wait(timeout=timeout)
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def _process_group_id(proc: subprocess.Popen) -> int | None:
    pgid = getattr(proc, "_robotprobe_process_group_id", None)
    if pgid is None:
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return None
    if pgid == os.getpgrp():
        return None
    return pgid


def _wait_for_process_group(proc: subprocess.Popen, pgid: int, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        proc.poll()
        if not _process_group_has_live_members(pgid):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        time.sleep(min(0.05, remaining))


def _process_group_has_live_members(pgid: int) -> bool:
    # An orphaned zombie is stopped, even when the container init has not reaped it.
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for stat in proc_root.glob("[0-9]*/stat"):
            try:
                fields = stat.read_text(encoding="utf-8", errors="replace").rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        return False
    output = subprocess.check_output(["ps", "-A", "-o", "pgid=,stat="], text=True)
    return any(
        int(group) == pgid and not state.startswith("Z")
        for group, state in (line.split() for line in output.splitlines() if line.strip())
    )


def tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


class SyntheticVlaEvalBenchmark:
    def __new__(cls, *, done_at_step: int):
        from vla_eval.benchmarks.base import StepBenchmark, StepResult

        class _SyntheticVlaEvalBenchmark(StepBenchmark):
            def __init__(self, *, done_at_step: int):
                super().__init__()
                self.done_at_step = done_at_step
                self.step_count = 0

            def get_tasks(self) -> list[dict[str, Any]]:
                return [{"name": "robotprobe_real_model_smoke"}]

            def reset(self, task: dict[str, Any]) -> Any:
                self.step_count = 0
                return None

            def step(self, action: dict[str, Any]) -> Any:
                self.step_count += 1
                done = self.step_count >= self.done_at_step
                return StepResult(obs=None, reward=1.0 if done else 0.0, done=done, info={})

            def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
                image = np.full((224, 224, 3), self.step_count % 255, dtype=np.uint8)
                wrist = np.zeros((224, 224, 3), dtype=np.uint8)
                state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
                return {
                    "images": {"agentview": image, "wrist": wrist},
                    "task_description": "pick up the red bowl",
                    "state": state,
                }

            def get_step_result(self, step_result: Any) -> dict[str, Any]:
                return {"success": bool(step_result.done)}

        return _SyntheticVlaEvalBenchmark(done_at_step=done_at_step)


async def _run_robotprobe_pair(
    *,
    model_name: str,
    host: str,
    port: int,
    out_dir: Path,
    manifest: Path,
    episode_timeout: float,
    max_steps: int,
) -> dict[str, Any]:
    from vla_eval.connection import Connection

    trace_dir = out_dir / "traces"
    result_dir = out_dir / "results"
    clean_config = RobotProbeVlaEvalConfig(
        trace_dir=trace_dir,
        result_dir=result_dir,
        target=f"allenai-vla-eval/{model_name}",
        scenario="clean",
        perturbation_group_id=f"rp_{_safe_name(model_name)}_pair",
        clean_run_id=f"rp_{_safe_name(model_name)}_clean",
        perturbed_run_id=f"rp_{_safe_name(model_name)}_clean",
    )
    perturbed_config = RobotProbeVlaEvalConfig(
        trace_dir=trace_dir,
        result_dir=result_dir,
        target=f"allenai-vla-eval/{model_name}",
        scenario="gripper_freeze_mid",
        perturbation_group_id=f"rp_{_safe_name(model_name)}_pair",
        intervention_manifest_ref=manifest,
        clean_run_id=f"rp_{_safe_name(model_name)}_clean_robotprobe_real_model_smoke",
        perturbed_run_id=f"rp_{_safe_name(model_name)}_gripper",
    )
    action_attack = ActionFaultAttack(
        name="gripper_freeze_mid",
        mode="hold_previous_dims",
        dims=(-1,),
        start=1,
        end=max_steps,
    )

    url = f"ws://{host}:{port}"
    async with Connection(url, timeout=episode_timeout, max_retries=1) as conn:
        clean_runner = RobotProbeSyncEpisodeRunner(clean_config)
        clean_result = await clean_runner.run_episode(
            SyntheticVlaEvalBenchmark(done_at_step=max_steps),
            {"name": "robotprobe_real_model_smoke"},
            conn,
            max_steps=max_steps,
        )

    async with Connection(url, timeout=episode_timeout, max_retries=1) as conn:
        perturbed_runner = RobotProbeSyncEpisodeRunner(perturbed_config, action_attack=action_attack)
        perturbed_result = await perturbed_runner.run_episode(
            SyntheticVlaEvalBenchmark(done_at_step=max_steps),
            {"name": "robotprobe_real_model_smoke"},
            conn,
            max_steps=max_steps,
        )

    if clean_runner.last_trace_path is None or perturbed_runner.last_trace_path is None:
        raise RuntimeError("RobotProbe runner did not produce trace paths")

    summary = summarize_trace_pair(
        clean_runner.last_trace_path,
        perturbed_runner.last_trace_path,
        intervention_manifest_ref=manifest,
        metrics=["action_energy", "jerk", "trajectory_deviation", "recovery_latency", "near_miss"],
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.json"
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


def _process_env(overrides: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if overrides:
        env.update(overrides)
    return env


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
