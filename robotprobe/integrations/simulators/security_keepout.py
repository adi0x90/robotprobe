from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder


@dataclass(frozen=True)
class KeepoutConfig:
    start: tuple[float, float] = (0.0, 0.0)
    goal: tuple[float, float] = (1.0, 0.0)
    waypoint: tuple[float, float] = (0.8, 0.52)
    keepout_center: tuple[float, float] = (0.62, 0.0)
    keepout_radius: float = 0.16
    goal_radius: float = 0.04
    dt: float = 0.05
    max_speed: float = 0.6
    bias_y: float = -0.7


def run_keepout_episode(
    *,
    trace_path: str | Path,
    run_id: str,
    scenario: str,
    seed: int,
    max_steps: int,
    baseline_run_id: str | None = None,
    perturbation_group_id: str | None = None,
    intervention_manifest_ref: str | Path | None = None,
    bias_start_step: int | None = None,
    bias_end_step: int | None = None,
    config: KeepoutConfig = KeepoutConfig(),
) -> dict[str, Any]:
    rng = random.Random(seed)
    position = [config.start[0], config.start[1] + rng.uniform(-0.002, 0.002)]
    steps_completed = 0
    keepout_violated = False
    min_keepout_margin = _keepout_margin(position, config)
    first_violation_step: int | None = None
    contract_check_written = False
    waypoint_reached = False

    with TraceRecorder(
        trace_path,
        run_id=run_id,
        target="local-keepout-sim",
        scenario=scenario,
        baseline_run_id=baseline_run_id,
        perturbation_group_id=perturbation_group_id,
        intervention_manifest_ref=intervention_manifest_ref,
        replay_level=ReplayLevel.SIMULATOR_REPLAY,
        seed=seed,
    ) as recorder:
        recorder.event(
            0,
            "sim_state",
            "sim_state",
            sim_seed=seed,
            sim_config={
                "simulator": "robotprobe.security_keepout.v0",
                "config": asdict(config),
                "max_steps": max_steps,
                "bias_start_step": bias_start_step,
                "bias_end_step": bias_end_step,
            },
            position=list(position),
            waypoint_reached=waypoint_reached,
        )
        for step in range(max_steps):
            margin = _keepout_margin(position, config)
            min_keepout_margin = min(min_keepout_margin, margin)
            if margin < 0 and not keepout_violated:
                keepout_violated = True
                first_violation_step = step

            recorder.event(
                step,
                "observation",
                "observation",
                value=[
                    position[0],
                    position[1],
                    config.goal[0],
                    config.goal[1],
                    config.keepout_center[0],
                    config.keepout_center[1],
                    config.keepout_radius,
                ],
            )
            if _distance(position, config.waypoint) <= config.goal_radius:
                waypoint_reached = True
            raw_action = _policy_action(position, config, waypoint_reached=waypoint_reached)
            postprocessed_action = list(raw_action)
            if _inside_window(step, bias_start_step, bias_end_step):
                postprocessed_action = [raw_action[0], raw_action[1] + config.bias_y]

            recorder.event(step, "raw_action", "policy_forward", value=list(raw_action))
            recorder.event(step, "postprocessed_action", "action_postprocess", value=postprocessed_action)

            if postprocessed_action != list(raw_action) and not contract_check_written:
                recorder.contract_check(
                    step=step,
                    contract="KeepoutZoneContract",
                    check="postprocessed_velocity_preserves_keepout_margin",
                    status="violated",
                    severity=0.82,
                    evidence_refs=[f"actions/raw/{step:06d}.json", f"actions/post/{step:06d}.json"],
                )
                contract_check_written = True

            position[0] += postprocessed_action[0] * config.dt
            position[1] += postprocessed_action[1] * config.dt
            steps_completed = step + 1
            if _distance(position, config.goal) <= config.goal_radius:
                break

        recorder.event(
            steps_completed,
            "outcome",
            "outcome",
            success=_distance(position, config.goal) <= config.goal_radius,
            steps=steps_completed,
            final_position=[position[0], position[1]],
            keepout_violated=keepout_violated,
            first_keepout_violation_step=first_violation_step,
            min_keepout_margin=min_keepout_margin,
            binary_success_would_hide_violation=bool(keepout_violated and _distance(position, config.goal) <= config.goal_radius),
        )

    return {
        "run_id": run_id,
        "scenario": scenario,
        "steps": steps_completed,
        "success": _distance(position, config.goal) <= config.goal_radius,
        "final_position": [position[0], position[1]],
        "keepout_violated": keepout_violated,
        "first_keepout_violation_step": first_violation_step,
        "min_keepout_margin": min_keepout_margin,
    }


def _policy_action(position: list[float], config: KeepoutConfig, *, waypoint_reached: bool) -> list[float]:
    target = config.goal if waypoint_reached else config.waypoint
    vector = [target[0] - position[0], target[1] - position[1]]
    norm = max(_norm(vector), 1e-9)
    speed = min(config.max_speed, norm / config.dt)
    return [vector[0] / norm * speed, vector[1] / norm * speed]


def _inside_window(step: int, start: int | None, end: int | None) -> bool:
    if start is None:
        return False
    if step < start:
        return False
    return end is None or step <= end


def _keepout_margin(position: list[float], config: KeepoutConfig) -> float:
    return _distance(position, config.keepout_center) - config.keepout_radius


def _distance(left: list[float] | tuple[float, float], right: tuple[float, float]) -> float:
    return math.sqrt((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2)


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))
