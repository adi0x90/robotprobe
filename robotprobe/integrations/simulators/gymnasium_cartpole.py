from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder


Policy = Callable[[Any], int]


def action_flip_postprocess(action: int, *, action_space_n: int) -> int:
    if action_space_n <= 1:
        return int(action)
    return (int(action) + 1) % int(action_space_n)


def run_episode_trace(
    *,
    env: Any,
    policy: Policy,
    trace_path: str | Path,
    run_id: str,
    scenario: str,
    target: str,
    seed: int,
    max_steps: int,
    baseline_run_id: str | None = None,
    perturbation_group_id: str | None = None,
    intervention_manifest_ref: str | Path | None = None,
    fault_start_step: int | None = None,
    fault_end_step: int | None = None,
) -> dict[str, Any]:
    observation, _info = env.reset(seed=seed)
    action_space_n = int(getattr(env.action_space, "n", 0) or 0)
    steps_completed = 0
    terminated = False
    truncated = False
    contract_check_written = False

    with TraceRecorder(
        trace_path,
        run_id=run_id,
        target=target,
        scenario=scenario,
        baseline_run_id=baseline_run_id,
        perturbation_group_id=perturbation_group_id,
        intervention_manifest_ref=intervention_manifest_ref,
        replay_level=ReplayLevel.SIMULATOR_REPLAY,
    ) as recorder:
        for step in range(max_steps):
            recorder.event(
                step,
                "observation",
                "observation",
                value=_flatten_observation(observation),
            )
            raw_action = int(policy(observation))
            postprocessed_action = raw_action
            if _inside_fault_window(step, fault_start_step, fault_end_step):
                postprocessed_action = action_flip_postprocess(raw_action, action_space_n=action_space_n)

            recorder.event(step, "raw_action", "policy_forward", value=[raw_action])
            recorder.event(step, "postprocessed_action", "action_postprocess", value=[postprocessed_action])

            if postprocessed_action != raw_action and not contract_check_written:
                recorder.contract_check(
                    step=step,
                    contract="DiscreteActionContract",
                    check="raw_vs_postprocessed_action_identity",
                    status="violated",
                    severity=0.55,
                    evidence_refs=[f"actions/raw/{step:06d}.json", f"actions/post/{step:06d}.json"],
                )
                contract_check_written = True

            observation, reward, terminated, truncated, info = env.step(postprocessed_action)
            steps_completed = step + 1
            if terminated or truncated:
                break

        recorder.event(
            steps_completed,
            "outcome",
            "outcome",
            success=steps_completed >= max_steps,
            steps=steps_completed,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    close = getattr(env, "close", None)
    if callable(close):
        close()

    return {
        "run_id": run_id,
        "scenario": scenario,
        "steps": steps_completed,
        "success": steps_completed >= max_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def cartpole_angle_policy(observation: Any) -> int:
    values = _flatten_observation(observation)
    angle = values[2] if len(values) > 2 else 0.0
    angular_velocity = values[3] if len(values) > 3 else 0.0
    return 1 if angle + 0.1 * angular_velocity >= 0.0 else 0


def _inside_fault_window(step: int, start: int | None, end: int | None) -> bool:
    if start is None:
        return False
    if step < start:
        return False
    return end is None or step <= end


def _flatten_observation(observation: Any) -> list[float]:
    if hasattr(observation, "tolist"):
        observation = observation.tolist()
    if isinstance(observation, (int, float)):
        return [float(observation)]
    if isinstance(observation, (list, tuple)):
        out: list[float] = []
        for item in observation:
            out.extend(_flatten_observation(item))
        return out
    return []
