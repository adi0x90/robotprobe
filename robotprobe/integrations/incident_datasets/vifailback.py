from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder


VIFAILBACK_REPO = "sii-rhos-ai/ViFailback-Dataset"
DEFAULT_VIFAILBACK_TASK = "cut_sausage"


def download_vifailback_seed(
    out_dir: str | Path,
    *,
    task: str = DEFAULT_VIFAILBACK_TASK,
    include_vqa: bool = True,
    include_large_hdf5: bool = False,
    episode_ids: list[int] | tuple[int, ...] = (),
) -> dict[str, Any]:
    """Download a small ViFailback seed, not the full dataset snapshot."""
    hf_hub_download = _hf_hub_download()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [
        "README.md",
        f"annotated_data/{task}/{task}_annotations.json",
    ]
    if include_vqa:
        files.append("VQA/ViFailback_VQA_benchmark.json")
    if include_large_hdf5:
        for episode_id in episode_ids:
            files.append(f"raw_data/{task}/episode_{episode_id}.hdf5")

    downloaded = []
    for filename in files:
        path = hf_hub_download(
            repo_id=VIFAILBACK_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=out,
        )
        downloaded.append(path)
    return {
        "dataset": "ViFailback",
        "repo": VIFAILBACK_REPO,
        "out": str(out),
        "downloaded": downloaded,
        "large_hdf5_included": include_large_hdf5,
    }


def convert_vifailback_annotations(
    annotation_path: str | Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    run_prefix: str = "vifailback",
    target: str = "vifailback",
    scenario: str | None = None,
) -> dict[str, Any]:
    source = Path(annotation_path)
    out = Path(out_dir)
    traces_dir = out / "traces"
    golden_dir = out / "golden"
    traces_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("ViFailback annotation file must contain a list of annotation records")

    selected = data[:limit] if limit is not None else data
    traces: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            continue
        video_ref = str(item.get("video", ""))
        task_slug = _task_slug_from_annotation(source, video_ref)
        episode_slug = _episode_slug(video_ref, fallback=index)
        trace_path = traces_dir / f"{task_slug}_{episode_slug}.jsonl"
        diagnosis_path = golden_dir / f"{task_slug}_{episode_slug}.diagnosis.json"
        run_id = f"{run_prefix}_{task_slug}_{episode_slug}"
        trace_scenario = scenario or task_slug
        failure_detected = _as_failure(item.get("failure_detection"))
        failure_step = _failure_step(item)
        failure_type = str(item.get("failure_type") or "unknown")

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
                value=str(item.get("task") or ""),
                dataset="ViFailback",
                source_annotation=str(source),
                source_video=video_ref,
                source_index=item.get("source_index", index),
            )
            for frame_index, image_ref in enumerate(item.get("images", []) or []):
                if not isinstance(image_ref, str):
                    continue
                recorder.event(
                    _image_step(image_ref, fallback=frame_index),
                    "observation",
                    "dataset_observation",
                    observation_ref=image_ref,
                    dataset="ViFailback",
                    image_role=_image_role(image_ref),
                )
            if failure_detected:
                recorder.contract_check(
                    step=failure_step,
                    contract="DatasetFailureAnnotation",
                    check=_normalize_check(failure_type),
                    status="violated",
                    severity=_failure_severity(failure_type),
                    evidence_refs=_evidence_refs(item),
                )
            recorder.event(
                _outcome_step(item, fallback=failure_step + 1),
                "outcome",
                "outcome",
                success=not failure_detected,
                steps=len(item.get("images", []) or []),
                dataset="ViFailback",
                source_video=video_ref,
                failure_detected=failure_detected,
                failure_type=failure_type if failure_detected else None,
                failure_subtask=str(item.get("failure_subtask")) if item.get("failure_subtask") is not None else None,
                failure_step=failure_step if failure_detected else None,
                avoidance_command=_first_low_level(item.get("avoidance")),
                correction_command=_first_low_level(item.get("correction")),
            )

        diagnosis = _diagnosis(item, trace_path=trace_path, failure_step=failure_step)
        diagnosis_path.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        traces.append(
            {
                "trace": str(trace_path),
                "diagnosis": str(diagnosis_path),
                "run_id": run_id,
                "source_video": video_ref,
                "failure_detected": failure_detected,
                "failure_type": failure_type if failure_detected else None,
            }
        )

    index_doc = {
        "format": "robotprobe.public_incident_pack.v0",
        "dataset": "ViFailback",
        "source": str(source),
        "trace_count": len(traces),
        "traces": traces,
        "limitations": [
            "Annotation traces are not clean-vs-perturbed causal pairs.",
            "Image and video refs remain external dataset refs unless separately downloaded.",
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


def _task_slug_from_annotation(source: Path, video_ref: str) -> str:
    match = re.search(r"annotated_data/([^/]+)/", video_ref)
    if match:
        return _slug(match.group(1))
    stem = source.stem.replace("_annotations", "")
    return _slug(stem or "vifailback")


def _episode_slug(video_ref: str, *, fallback: int) -> str:
    match = re.search(r"(episode[_-]?\d+)", video_ref)
    if match:
        return _slug(match.group(1))
    return f"episode_{fallback}"


def _as_failure(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "failed", "failure"}
    return bool(value)


def _failure_step(item: dict[str, Any]) -> int:
    for keyframe in item.get("keyframe", []) or []:
        if isinstance(keyframe, dict):
            avoid = keyframe.get("avoid_keyframe")
            if isinstance(avoid, str):
                return _image_step(avoid, fallback=0)
    for image_ref in item.get("images", []) or []:
        if isinstance(image_ref, str) and "avoid_keyframe" in image_ref:
            return _image_step(image_ref, fallback=0)
    return 0


def _image_step(image_ref: str, *, fallback: int) -> int:
    name = Path(image_ref).name
    match = re.match(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return fallback


def _image_role(image_ref: str) -> str:
    name = Path(image_ref).name
    if "avoid_keyframe" in name:
        return "avoid_keyframe"
    if "correct_keyframe" in name:
        return "correct_keyframe"
    return "timeline_frame"


def _normalize_check(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "annotated_failure"


def _failure_severity(failure_type: str) -> float:
    if failure_type.lower() == "human intervention":
        return 0.9
    if "planning" in failure_type.lower():
        return 0.78
    return 0.74


def _evidence_refs(item: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for keyframe in item.get("keyframe", []) or []:
        if not isinstance(keyframe, dict):
            continue
        for key in ("avoid_keyframe", "correct_keyframe"):
            value = keyframe.get(key)
            if isinstance(value, str):
                refs.append(value)
    if not refs:
        refs.extend(ref for ref in item.get("images", []) or [] if isinstance(ref, str))
    return refs[:8]


def _outcome_step(item: dict[str, Any], *, fallback: int) -> int:
    image_steps = [_image_step(ref, fallback=idx) for idx, ref in enumerate(item.get("images", []) or []) if isinstance(ref, str)]
    return (max(image_steps) + 1) if image_steps else fallback


def _first_low_level(value: Any) -> str | None:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        low_level = value[0].get("low_level")
        return str(low_level) if low_level is not None else None
    return None


def _diagnosis(item: dict[str, Any], *, trace_path: Path, failure_step: int) -> dict[str, Any]:
    failure_type = str(item.get("failure_type") or "unknown")
    return {
        "format": "robotprobe.public_incident_diagnosis.v0",
        "dataset": "ViFailback",
        "trace": str(trace_path),
        "diagnosis": {
            "layer": "dataset_annotation",
            "contract": "DatasetFailureAnnotation",
            "fault": _normalize_check(failure_type),
            "failure_type": failure_type,
            "failure_step": failure_step,
            "failure_subtask": item.get("failure_subtask"),
        },
        "evidence": {
            "source_video": item.get("video"),
            "evidence_refs": _evidence_refs(item),
            "avoidance_command": _first_low_level(item.get("avoidance")),
            "correction_command": _first_low_level(item.get("correction")),
        },
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "item"
