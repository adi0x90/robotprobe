from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robotprobe.core import ReplayLevel
from robotprobe.mech.activation_hooks import CapturePolicy
from robotprobe.mech.activation_store import ActivationIndexWriter, ActivationStore
from robotprobe.mech.schemas import ModelActivationEvent, load_capture_config, validate_capture_config
from robotprobe.summarize import read_trace


def collect_activations(
    *,
    target: str,
    capture_config: str | Path,
    out_dir: str | Path,
    trace_pair: str | None = None,
    clean_trace: str | Path | None = None,
    perturbed_trace: str | Path | None = None,
) -> dict[str, Any]:
    config = load_capture_config(capture_config)
    validation = validate_capture_config(config)
    if not validation["valid"]:
        return {
            "format": "robotprobe.mech.collect.v0",
            "status": "invalid_config",
            "validation": validation,
        }
    if target not in {"synthetic", "toy"}:
        return {
            "format": "robotprobe.mech.collect.v0",
            "status": "not_implemented",
            "target": target,
            "reason": "live adapters use the Python capture_pair/persist_capture_pair API; CLI model loading is not implemented",
        }
    clean_path, perturbed_path = _resolve_trace_pair(trace_pair, clean_trace, perturbed_trace)
    return _collect_synthetic(
        target=target,
        config=config,
        clean_trace=clean_path,
        perturbed_trace=perturbed_path,
        out_dir=out_dir,
    )


def persist_capture_pair(
    *,
    capture_pair: dict[str, Any],
    capture_config: dict[str, Any],
    out_dir: str | Path,
    incident_id: str,
    clean_run_id: str,
    perturbed_run_id: str,
    model: dict[str, Any],
    target: str,
    scenario: str,
    replay_level: str,
    external_first_divergence_step: int | None = None,
    negative_control: bool = False,
    synthetic: bool = False,
    synthetic_lead_time_steps: int | None = None,
) -> dict[str, Any]:
    """Persist adapter snapshots with explicit incident identity and per-run event traces.

    Sampling, stored dtype, and serialized tensor-file budgets are enforced here even
    when an adapter has already applied the corresponding in-memory capture settings.
    """
    validation = validate_capture_config(capture_config)
    if not validation["valid"]:
        raise ValueError(f"invalid capture config: {validation['errors']}")
    if replay_level not in {level.value for level in ReplayLevel}:
        raise ValueError(f"invalid replay_level: {replay_level}")
    if not all(isinstance(value, str) and value for value in (incident_id, clean_run_id, perturbed_run_id, target, scenario)):
        raise ValueError("incident, run, target, and scenario identifiers must be nonempty strings")
    if clean_run_id == perturbed_run_id:
        raise ValueError("clean and perturbed run IDs must differ")
    if not isinstance(capture_pair, dict) or not all(isinstance(capture_pair.get(label), list) for label in ("clean", "perturbed")):
        raise ValueError("capture_pair must contain clean and perturbed capture lists")
    if type(synthetic) is not bool or type(negative_control) is not bool:
        raise ValueError("synthetic and negative_control must be boolean")
    if external_first_divergence_step is not None and (
        type(external_first_divergence_step) is not int or external_first_divergence_step < 0
    ):
        raise ValueError("external_first_divergence_step must be a nonnegative integer or None")
    mech_dir = Path(out_dir)
    if mech_dir.exists() and any(mech_dir.iterdir()):
        raise ValueError("capture output directory must be empty; choose a fresh directory")
    storage = capture_config["storage"]
    policy = CapturePolicy.from_config(capture_config)
    layers = {layer["name"]: layer for layer in capture_config["layers"]}
    store = ActivationStore(
        mech_dir, store_format=storage["format"], dtype=storage.get("dtype"),
        max_bytes_per_run=storage["max_bytes_per_run"],
    )
    writer = ActivationIndexWriter(
        mech_dir, incident_id=incident_id, clean_run_id=clean_run_id, perturbed_run_id=perturbed_run_id,
        model=model, external_first_divergence_step=external_first_divergence_step,
        negative_control=negative_control, synthetic=synthetic,
        synthetic_lead_time_steps=synthetic_lead_time_steps, required_modules=list(layers),
    )
    events: dict[str, list[dict[str, Any]]] = {"clean": [], "perturbed": []}
    seen: set[tuple[str, str, int]] = set()
    for run_label, run_id in (("clean", clean_run_id), ("perturbed", perturbed_run_id)):
        for item in capture_pair[run_label]:
            module, step = item["module"], item["step"]
            if module not in layers:
                raise ValueError(f"capture contains unconfigured module: {module}")
            if type(step) is not int or step < 0:
                raise ValueError("capture steps must be nonnegative integers")
            if not policy.should_capture(step):
                continue
            key = (run_label, module, step)
            if key in seen:
                raise ValueError(f"duplicate activation capture: {key}")
            seen.add(key)
            capture = store.write_activation(
                run_label=run_label, run_id=run_id, step=step, module=module,
                module_kind=layers[module].get("kind", "activation"), activation=item["activation"],
                source_dtype=item.get("source_dtype"),
            )
            writer.add(capture)
            events[run_label].append(ModelActivationEvent(
                run_id=run_id, target=target, scenario=scenario, replay_level=replay_level,
                step=step, module=module, module_kind=capture.module_kind,
                activation_ref=capture.activation_ref, activation_shape=capture.shape,
                activation_dtype=capture.dtype, model_ref={"model_name": model.get("name", target), **model},
                baseline_run_id=clean_run_id if run_label == "perturbed" else None,
                perturbation_group_id=incident_id,
                capture={
                    "capture_mode": "synthetic" if synthetic else capture_pair.get("target", "adapter"),
                    "hook_version": "mechprobe.activation_hook.v0", "detached": True, "requires_grad": False,
                    "synthetic": synthetic, "source_dtype": capture.source_dtype,
                },
            ).to_trace_event())
    index_path = writer.write()
    event_paths = {}
    for label, run_events in events.items():
        event_path = mech_dir / f"model_activation_events.{label}.jsonl"
        event_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in run_events), encoding="utf-8")
        event_paths[label] = str(event_path)
    summary_path = mech_dir / "activation_summary.json"
    count = sum(map(len, events.values()))
    result = {
        "format": "robotprobe.mech.collect.v0", "status": "captured" if count else "no_captures",
        "target": target, "synthetic": synthetic, "activation_count": count,
        "activation_index": str(index_path), "activation_events": event_paths,
        "activation_summary": str(summary_path), "external_first_divergence_step": external_first_divergence_step,
    }
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _collect_synthetic(
    *, target: str, config: dict[str, Any], clean_trace: Path, perturbed_trace: Path, out_dir: str | Path,
) -> dict[str, Any]:
    clean_events = read_trace(clean_trace)
    perturbed_events = read_trace(perturbed_trace)
    external_first = _first_action_divergence(clean_events, perturbed_events)
    synthetic_lead_time = 4
    # Artificial fixture placement, never measured model lead time.
    divergence_step = max(0, (external_first if external_first is not None else 14) - synthetic_lead_time)
    stable_step = max(0, divergence_step - 5)
    captures: dict[str, Any] = {"target": "synthetic", "clean": [], "perturbed": []}
    for layer in config["layers"]:
        for label in ("clean", "perturbed"):
            if stable_step != divergence_step:
                captures[label].append({"step": stable_step, "module": layer["name"], "activation": np.ones(3, dtype=np.float32)})
            activation = np.array([3.0 if label == "perturbed" else 1.0, 1.0, 1.0], dtype=np.float32)
            captures[label].append({"step": divergence_step, "module": layer["name"], "activation": activation})
    return persist_capture_pair(
        capture_pair=captures, capture_config=config, out_dir=out_dir, incident_id=config["id"],
        clean_run_id=_run_id(clean_events, fallback="clean"),
        perturbed_run_id=_run_id(perturbed_events, fallback="perturbed"),
        model={"name": target, "checkpoint_hash": "sha256:synthetic", "adapter": "mechprobe.synthetic.v0", "runtime": "numpy"},
        target=target, scenario=config["id"], replay_level="L0_TRACE_PLAYBACK",
        external_first_divergence_step=external_first, synthetic=True, synthetic_lead_time_steps=synthetic_lead_time,
    )


def _resolve_trace_pair(
    trace_pair: str | None, clean_trace: str | Path | None, perturbed_trace: str | Path | None,
) -> tuple[Path, Path]:
    if trace_pair:
        left, separator, right = trace_pair.partition(",")
        if not separator or not left or not right:
            raise ValueError("--trace-pair must be CLEAN,PERTURBED")
        return Path(left), Path(right)
    if clean_trace is None or perturbed_trace is None:
        raise ValueError("trace_pair or clean_trace/perturbed_trace is required")
    return Path(clean_trace), Path(perturbed_trace)


def _run_id(events: list[dict[str, Any]], *, fallback: str) -> str:
    return str(events[0].get("run_id", fallback)) if events else fallback


def _first_action_divergence(clean: list[dict[str, Any]], perturbed: list[dict[str, Any]]) -> int | None:
    first_steps = []
    for event_type in ("raw_action", "postprocessed_action"):
        clean_by_step = _events_by_step(clean, event_type)
        perturbed_by_step = _events_by_step(perturbed, event_type)
        for step in sorted(set(clean_by_step) & set(perturbed_by_step)):
            if clean_by_step[step].get("value") != perturbed_by_step[step].get("value"):
                first_steps.append(step)
                break
    return min(first_steps) if first_steps else None


def _events_by_step(events: list[dict[str, Any]], event_type: str) -> dict[int, dict[str, Any]]:
    return {int(event["step"]): event for event in events if event.get("event_type") == event_type}
