from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder


ROBOFAC_REPO = "MINT-SJTU/RoboFAC-dataset"
DEFAULT_ROBOFAC_LEAF = "simulation_data/LiftPegUpright-box/view0"


def download_robofac_seed(
    out_dir: str | Path,
    *,
    leaf: str = DEFAULT_ROBOFAC_LEAF,
    include_h5: bool = True,
) -> dict[str, Any]:
    """Download one small RoboFAC leaf directory seed, not the full dataset snapshot."""
    hf_hub_download = _hf_hub_download()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    leaf_path = leaf.strip("/")
    name = Path(leaf_path).name
    files = ["README.md", f"{leaf_path}/{name}.json"]
    if include_h5:
        files.append(f"{leaf_path}/{name}.h5")

    downloaded = []
    for filename in files:
        path = hf_hub_download(
            repo_id=ROBOFAC_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=out,
        )
        downloaded.append(path)
    return {
        "dataset": "RoboFAC",
        "repo": ROBOFAC_REPO,
        "out": str(out),
        "downloaded": downloaded,
        "h5_included": include_h5,
    }


def convert_robofac_episodes(
    metadata_path: str | Path,
    out_dir: str | Path,
    *,
    h5_path: str | Path | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    run_prefix: str = "robofac",
    target: str = "robofac",
    scenario: str | None = None,
) -> dict[str, Any]:
    source = Path(metadata_path)
    out = Path(out_dir)
    traces_dir = out / "traces"
    golden_dir = out / "golden"
    traces_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("RoboFAC metadata file must contain a JSON object")
    episodes = metadata.get("episodes") or []
    if not isinstance(episodes, list):
        raise ValueError("RoboFAC metadata field 'episodes' must be a list")
    selected = episodes[:limit] if limit is not None else episodes
    env_info = metadata.get("env_info") if isinstance(metadata.get("env_info"), dict) else {}
    env_id = str(env_info.get("env_id") or source.parent.name)
    trace_scenario = scenario or _slug(env_id)

    h5_source = Path(h5_path) if h5_path is not None else None
    traces: list[dict[str, Any]] = []
    for index, episode in enumerate(selected):
        if not isinstance(episode, dict):
            continue
        episode_id = int(episode.get("episode_id", index))
        unique_id = str(episode.get("unique_id") or f"episode_{episode_id}")
        trace_path = traces_dir / f"{_slug(env_id)}_episode_{episode_id}.jsonl"
        diagnosis_path = golden_dir / f"{_slug(env_id)}_episode_{episode_id}.diagnosis.json"
        run_id = f"{run_prefix}_{_slug(env_id)}_episode_{episode_id}"
        trajectory = _load_h5_trajectory(h5_source, episode_id) if h5_source is not None else None
        actions = trajectory.get("actions", []) if trajectory else []
        states = trajectory.get("states", []) if trajectory else []
        rewards = trajectory.get("rewards", []) if trajectory else []
        h5_success = trajectory.get("success") if trajectory else None
        success = bool(episode.get("success", h5_success if h5_success is not None else False))
        step_count = _step_count(episode, actions)
        emitted_steps = min(step_count, max_steps) if max_steps is not None else step_count

        with TraceRecorder(
            trace_path,
            run_id=run_id,
            target=target,
            scenario=trace_scenario,
            replay_level=ReplayLevel.TRACE_PLAYBACK,
        ) as recorder:
            recorder.event(
                0,
                "instruction",
                "instruction",
                value=_task_text(env_id, metadata),
                dataset="RoboFAC",
                source_metadata=str(source),
                source_h5=str(h5_source) if h5_source is not None else None,
                episode_id=episode_id,
                unique_id=unique_id,
            )
            for step in range(emitted_steps):
                if step < len(states):
                    recorder.event(step, "robot_state", "dataset_state", value=states[step])
                if step < len(actions):
                    action = actions[step]
                    recorder.event(step, "raw_action", "dataset_policy_action", value=action)
                    recorder.event(step, "postprocessed_action", "dataset_applied_action", value=action)
                elif not actions and step == 0:
                    recorder.event(
                        step,
                        "observation",
                        "dataset_observation",
                        value={
                            "episode_id": episode_id,
                            "unique_id": unique_id,
                            "control_mode": episode.get("control_mode"),
                            "episode_seed": episode.get("episode_seed"),
                        },
                    )
                if step < len(rewards):
                    recorder.event(step, "sim_state", "dataset_reward", value={"reward": rewards[step]})
            if not success:
                recorder.contract_check(
                    step=max(emitted_steps - 1, 0),
                    contract="RoboFACFailureOutcome",
                    check="episode_marked_unsuccessful",
                    status="violated",
                    severity=0.72,
                    evidence_refs=[str(source), str(h5_source) if h5_source is not None else f"episode:{episode_id}"],
                )
            recorder.event(
                emitted_steps,
                "outcome",
                "outcome",
                success=success,
                steps=step_count,
                emitted_steps=emitted_steps,
                dataset="RoboFAC",
                env_id=env_id,
                episode_id=episode_id,
                unique_id=unique_id,
                control_mode=episode.get("control_mode"),
                source_type=metadata.get("source_type"),
            )

        diagnosis = _diagnosis(
            trace_path=trace_path,
            env_id=env_id,
            episode=episode,
            success=success,
            h5_used=trajectory is not None,
        )
        diagnosis_path.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        traces.append(
            {
                "trace": str(trace_path),
                "diagnosis": str(diagnosis_path),
                "run_id": run_id,
                "episode_id": episode_id,
                "unique_id": unique_id,
                "success": success,
                "h5_used": trajectory is not None,
            }
        )

    index_doc = {
        "format": "robotprobe.public_incident_pack.v0",
        "dataset": "RoboFAC",
        "source": str(source),
        "h5": str(h5_source) if h5_source is not None else None,
        "trace_count": len(traces),
        "traces": traces,
        "limitations": [
            "RoboFAC metadata traces are annotated episode traces, not clean-vs-perturbed causal pairs.",
            "H5 action/state conversion is best-effort and depends on the leaf directory schema.",
        ],
    }
    (out / "index.json").write_text(json.dumps(index_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index_doc


def _hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install the incident-datasets extra: python -m pip install -e '.[incident-datasets]'") from exc
    return hf_hub_download


def _load_h5_trajectory(h5_path: Path | None, episode_id: int) -> dict[str, Any] | None:
    if h5_path is None:
        return None
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("RoboFAC H5 conversion requires h5py; install python -m pip install -e '.[incident-datasets]'") from exc
    group_name = f"traj_{episode_id}"
    with h5py.File(h5_path, "r") as f:
        if group_name not in f:
            return None
        group = f[group_name]
        actions = _dataset_list(group, "actions")
        rewards = _dataset_list(group, "rewards")
        success_values = _dataset_list(group, "success")
        states = _first_state_dataset(group)
    return {
        "actions": actions,
        "states": states,
        "rewards": rewards,
        "success": bool(success_values[-1]) if success_values else None,
    }


def _dataset_list(group: Any, key: str) -> list[Any]:
    if key not in group:
        return []
    data = group[key][()]
    return [_jsonable(item) for item in data]


def _first_state_dataset(group: Any) -> list[Any]:
    for path in (
        "env_states/articulations/panda",
        "env_states/actors/peg",
    ):
        current = group
        found = True
        for part in path.split("/"):
            if part not in current:
                found = False
                break
            current = current[part]
        if found:
            data = current[()]
            return [_jsonable(item) for item in data]
    return []


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def _step_count(episode: dict[str, Any], actions: list[Any]) -> int:
    if actions:
        return len(actions)
    elapsed = episode.get("elapsed_steps")
    try:
        return int(elapsed)
    except (TypeError, ValueError):
        return 1


def _task_text(env_id: str, metadata: dict[str, Any]) -> str:
    source_desc = metadata.get("source_desc")
    if source_desc:
        return f"{env_id}: {source_desc}"
    return f"RoboFAC episode in {env_id}"


def _diagnosis(*, trace_path: Path, env_id: str, episode: dict[str, Any], success: bool, h5_used: bool) -> dict[str, Any]:
    return {
        "format": "robotprobe.public_incident_diagnosis.v0",
        "dataset": "RoboFAC",
        "trace": str(trace_path),
        "diagnosis": {
            "layer": "dataset_annotation",
            "contract": "RoboFACFailureOutcome",
            "fault": "episode_marked_unsuccessful" if not success else "none",
            "env_id": env_id,
            "episode_id": episode.get("episode_id"),
        },
        "evidence": {
            "success": success,
            "elapsed_steps": episode.get("elapsed_steps"),
            "unique_id": episode.get("unique_id"),
            "h5_used": h5_used,
        },
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "item"
