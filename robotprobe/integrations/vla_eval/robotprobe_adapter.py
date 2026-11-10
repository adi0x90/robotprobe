from __future__ import annotations

import dataclasses
import hashlib
import json
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotprobe.attacks import Attack, Observation
from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder
from robotprobe.report import build_incident_report
from robotprobe.summarize import summarize_trace_pair
from robotsec_range.challenge_spec import load_builtin_challenge
from robotsec_range.checker import check_summary


def perturb_vla_eval_observation(
    observation: dict[str, Any],
    attack: Attack,
    *,
    step: int,
    camera: str | None = None,
) -> dict[str, Any]:
    """Apply a RobotProbe observation attack to a vla-eval observation payload."""
    images = dict(observation.get("images", {}))
    camera_name = camera or _first_camera(images)
    if camera_name is None:
        raise ValueError("vla-eval observation has no images to perturb")

    original = Observation(
        image=images[camera_name],
        instruction=str(observation.get("task_description", "")),
        state=observation.get("state"),
    )
    perturbed = attack.perturb(original, step=step)

    out = dict(observation)
    out["images"] = images
    out["task_description"] = perturbed.instruction
    if perturbed.image is not original.image:
        out["images"][camera_name] = np.asarray(perturbed.image)
    return out


def perturb_vla_eval_action(
    action_payload: dict[str, Any],
    attack: Attack,
    *,
    step: int,
    action_key: str = "actions",
) -> dict[str, Any]:
    """Apply a RobotProbe action attack to a vla-eval action payload."""
    if action_key not in action_payload:
        raise ValueError(f"vla-eval action payload missing {action_key!r}")
    out = dict(action_payload)
    out[action_key] = attack.perturb_action(np.asarray(action_payload[action_key]), step=step)
    return out


@dataclass(frozen=True)
class RobotProbeVlaEvalConfig:
    trace_dir: Path
    result_dir: Path
    target: str
    scenario: str
    perturbation_group_id: str
    intervention_manifest_ref: str | Path | None = None
    challenge: str | None = None
    metrics: tuple[str, ...] = ()
    replay_level: ReplayLevel = ReplayLevel.TRACE_PLAYBACK
    clean_run_id: str = "rp_vla_eval_clean"
    perturbed_run_id: str = "rp_vla_eval_perturbed"


@dataclass(frozen=True)
class RobotProbeVlaEvalResult:
    clean_trace: Path
    perturbed_trace: Path
    summary_path: Path
    check_path: Path | None
    report_path: Path


class RobotProbeVlaEvalEpisode:
    def __init__(
        self,
        recorder: TraceRecorder,
        *,
        observation_attack: Attack | None = None,
        camera: str | None = None,
    ):
        self.recorder = recorder
        self.observation_attack = observation_attack
        self.camera = camera

    def pre_inference_observation(self, observation: dict[str, Any], *, step: int) -> dict[str, Any]:
        before_instruction = str(observation.get("task_description", ""))
        out = observation
        if self.observation_attack is not None:
            out = perturb_vla_eval_observation(
                observation,
                self.observation_attack,
                step=step,
                camera=self.camera,
            )
        after_instruction = str(out.get("task_description", ""))
        image_sha256 = _hash_observation_image(out, camera=self.camera)
        self.recorder.event(
            step,
            "instruction",
            "instruction",
            before=before_instruction,
            after=after_instruction,
            value=after_instruction,
        )
        self.recorder.event(
            step,
            "observation",
            "observation",
            image_sha256=image_sha256,
        )
        self.recorder.event(
            step,
            "policy_input",
            "policy_forward",
            observation_ref=f"step:{step}",
        )
        return out

    def post_inference_action(
        self,
        action_payload: dict[str, Any],
        *,
        step: int,
        action_attack: Attack | None = None,
        action_key: str = "actions",
    ) -> dict[str, Any]:
        raw = np.asarray(action_payload[action_key])
        self.recorder.event(step, "raw_action", "policy_forward", value=_jsonable_array(raw))
        out = action_payload
        if action_attack is not None:
            out = perturb_vla_eval_action(action_payload, action_attack, step=step, action_key=action_key)
        post = np.asarray(out[action_key])
        self.recorder.event(step, "postprocessed_action", "action_postprocess", value=_jsonable_array(post))
        return out

    def record_outcome(self, *, success: bool, steps: int, **fields: Any) -> None:
        self.recorder.event(steps, "outcome", "outcome", success=success, steps=steps, **fields)


class RobotProbeVlaEvalRunner:
    def __init__(self, config: RobotProbeVlaEvalConfig):
        self.config = config

    def run_pair(self, *, clean_episode, perturbed_episode) -> RobotProbeVlaEvalResult:
        self.config.trace_dir.mkdir(parents=True, exist_ok=True)
        self.config.result_dir.mkdir(parents=True, exist_ok=True)
        clean_trace = self.config.trace_dir / f"{self.config.clean_run_id}.jsonl"
        perturbed_trace = self.config.trace_dir / f"{self.config.perturbed_run_id}.jsonl"

        with TraceRecorder(
            clean_trace,
            run_id=self.config.clean_run_id,
            target=self.config.target,
            scenario="clean",
            replay_level=self.config.replay_level,
        ) as recorder:
            clean_episode(RobotProbeVlaEvalEpisode(recorder))

        with TraceRecorder(
            perturbed_trace,
            run_id=self.config.perturbed_run_id,
            target=self.config.target,
            scenario=self.config.scenario,
            baseline_run_id=self.config.clean_run_id,
            perturbation_group_id=self.config.perturbation_group_id,
            intervention_manifest_ref=self.config.intervention_manifest_ref,
            replay_level=self.config.replay_level,
        ) as recorder:
            perturbed_episode(RobotProbeVlaEvalEpisode(recorder))

        summary = summarize_trace_pair(
            clean_trace,
            perturbed_trace,
            intervention_manifest_ref=self.config.intervention_manifest_ref,
            metrics=self.config.metrics,
        )
        summary_path = self.config.result_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        check_path = None
        check_data = None
        if self.config.challenge is not None:
            check = check_summary(load_builtin_challenge(self.config.challenge), summary)
            check_data = dataclasses.asdict(check)
            check_path = self.config.result_dir / "check.json"
            check_path.write_text(json.dumps(check_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        report_path = self.config.result_dir / "report.md"
        report_path.write_text(
            build_incident_report(summary, check=check_data, title=f"RobotProbe {self.config.scenario}"),
            encoding="utf-8",
        )
        return RobotProbeVlaEvalResult(
            clean_trace=clean_trace,
            perturbed_trace=perturbed_trace,
            summary_path=summary_path,
            check_path=check_path,
            report_path=report_path,
        )


class RobotProbeSyncEpisodeRunner:
    """RobotProbe-instrumented sync runner for AllenAI vla-evaluation-harness.

    This follows the current `SyncEpisodeRunner` episode flow while adding RobotProbe
    observation, raw-action, postprocessed-action, and outcome trace events.
    """

    def __init__(
        self,
        config: RobotProbeVlaEvalConfig,
        *,
        observation_attack: Attack | None = None,
        action_attack: Attack | None = None,
        camera: str | None = None,
    ):
        self.config = config
        self.observation_attack = observation_attack
        self.action_attack = action_attack
        self.camera = camera
        self.last_trace_path: Path | None = None

    async def run_episode(
        self,
        benchmark: Any,
        task: dict[str, Any],
        conn: Any,
        *,
        max_steps: int | None = None,
        recorder: Any | None = None,
    ) -> dict[str, Any]:
        self.config.trace_dir.mkdir(parents=True, exist_ok=True)
        task_name = _safe_task_name(task)
        run_id = f"{self.config.perturbed_run_id}_{task_name}"
        trace_path = self.config.trace_dir / f"{run_id}.jsonl"
        self.last_trace_path = trace_path

        await benchmark.start_episode(task, recorder=recorder)
        obs_dict = await benchmark.get_observation()
        task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
        ep_payload: dict[str, Any] = {"task": task_info}
        if recorder is not None and getattr(recorder, "is_active", False):
            ep_payload["recording"] = {
                "sid": recorder.sid,
                "eid": recorder.eid,
                "eval_id": recorder.eval_id,
                "db_path": recorder.db_path,
            }
        await conn.start_episode(ep_payload)

        with TraceRecorder(
            trace_path,
            run_id=run_id,
            target=self.config.target,
            scenario=self.config.scenario,
            baseline_run_id=self.config.clean_run_id,
            perturbation_group_id=self.config.perturbation_group_id,
            intervention_manifest_ref=self.config.intervention_manifest_ref,
            replay_level=self.config.replay_level,
        ) as trace:
            episode = RobotProbeVlaEvalEpisode(
                trace,
                observation_attack=self.observation_attack,
                camera=self.camera,
            )
            steps = range(max_steps) if max_steps is not None else itertools.count()
            step = -1
            for step in steps:
                obs_for_policy = episode.pre_inference_observation(obs_dict, step=step)
                raw_action = await conn.act(obs_for_policy)
                action = episode.post_inference_action(
                    raw_action,
                    step=step,
                    action_attack=self.action_attack,
                )
                await benchmark.apply_action(action)
                if await benchmark.is_done():
                    break
                obs_dict = await benchmark.get_observation()

            elapsed = await benchmark.get_time()
            metrics = await benchmark.get_result()
            episode_result: dict[str, Any] = {
                "metrics": metrics,
                "steps": step + 1,
                "elapsed_sec": round(float(elapsed), 3),
            }
            episode.record_outcome(
                success=bool(metrics.get("success", False)),
                steps=step + 1,
                elapsed_sec=episode_result["elapsed_sec"],
            )

        await conn.end_episode(episode_result)
        return episode_result


def _first_camera(images: dict[str, Any]) -> str | None:
    if not images:
        return None
    return sorted(images)[0]


def _hash_observation_image(observation: dict[str, Any], *, camera: str | None) -> str | None:
    images = observation.get("images", {})
    if not images:
        return None
    camera_name = camera or _first_camera(images)
    if camera_name is None or camera_name not in images:
        return None
    arr = np.asarray(images[camera_name])
    return "sha256:" + hashlib.sha256(arr.tobytes()).hexdigest()


def _jsonable_array(value: np.ndarray) -> Any:
    return value.tolist()


def _safe_task_name(task: dict[str, Any]) -> str:
    raw = str(task.get("name", "episode"))
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw) or "episode"
