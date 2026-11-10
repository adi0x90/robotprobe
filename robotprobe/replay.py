from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robotprobe.core import ReplayLevel
from robotprobe.summarize import read_trace


LEVEL_ALIASES = {
    "L0": ReplayLevel.TRACE_PLAYBACK,
    "TRACE_PLAYBACK": ReplayLevel.TRACE_PLAYBACK,
    ReplayLevel.TRACE_PLAYBACK.value: ReplayLevel.TRACE_PLAYBACK,
    "L1": ReplayLevel.POLICY_REPLAY,
    "POLICY_REPLAY": ReplayLevel.POLICY_REPLAY,
    ReplayLevel.POLICY_REPLAY.value: ReplayLevel.POLICY_REPLAY,
    "L2": ReplayLevel.SIMULATOR_REPLAY,
    "SIM_REPLAY": ReplayLevel.SIMULATOR_REPLAY,
    "SIMULATOR_REPLAY": ReplayLevel.SIMULATOR_REPLAY,
    ReplayLevel.SIMULATOR_REPLAY.value: ReplayLevel.SIMULATOR_REPLAY,
    "L3": ReplayLevel.SHADOW_HARDWARE_REPLAY,
    "SHADOW_HARDWARE_REPLAY": ReplayLevel.SHADOW_HARDWARE_REPLAY,
    ReplayLevel.SHADOW_HARDWARE_REPLAY.value: ReplayLevel.SHADOW_HARDWARE_REPLAY,
}


def parse_replay_level(value: str | ReplayLevel) -> ReplayLevel:
    if isinstance(value, ReplayLevel):
        return value
    key = value.upper()
    if key not in LEVEL_ALIASES:
        raise ValueError(f"unknown replay level: {value}")
    return LEVEL_ALIASES[key]


def build_replay_plan(
    trace_path: str | Path,
    *,
    level: str | ReplayLevel,
    action_envelope_path: str | Path | None = None,
) -> dict[str, Any]:
    events = read_trace(trace_path)
    replay_level = parse_replay_level(level)
    if replay_level is ReplayLevel.TRACE_PLAYBACK:
        return _l0_trace_playback(trace_path, events)
    if replay_level is ReplayLevel.POLICY_REPLAY:
        return _l1_policy_replay(trace_path, events)
    if replay_level is ReplayLevel.SIMULATOR_REPLAY:
        return _l2_sim_replay(trace_path, events)
    if replay_level is ReplayLevel.SHADOW_HARDWARE_REPLAY:
        envelope = _load_envelope(action_envelope_path)
        return _l3_shadow_hardware(trace_path, events, envelope)
    raise AssertionError(f"unhandled replay level: {replay_level}")


def _l0_trace_playback(trace_path: str | Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    timeline = [
        {
            "step": event.get("step"),
            "event_type": event.get("event_type"),
            "layer": event.get("layer"),
        }
        for event in sorted(events, key=lambda e: (int(e.get("step", 0)), str(e.get("event_type", ""))))
    ]
    return {
        "trace": str(trace_path),
        "level": ReplayLevel.TRACE_PLAYBACK.value,
        "event_count": len(events),
        "timeline": timeline,
    }


def _l1_policy_replay(trace_path: str | Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    policy_inputs = [
        event for event in events
        if event.get("event_type") in {"policy_input", "instruction", "observation"}
    ]
    return {
        "trace": str(trace_path),
        "level": ReplayLevel.POLICY_REPLAY.value,
        "policy_input_count": sum(1 for event in events if event.get("event_type") == "policy_input"),
        "observation_count": sum(1 for event in events if event.get("event_type") == "observation"),
        "instruction_count": sum(1 for event in events if event.get("event_type") == "instruction"),
        "inputs": [
            {
                "step": event.get("step"),
                "event_type": event.get("event_type"),
                "observation_ref": event.get("observation_ref"),
                "image_sha256": event.get("image_sha256"),
                "value": event.get("value"),
            }
            for event in policy_inputs
        ],
    }


def _l2_sim_replay(trace_path: str | Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = _first_with_any(events, ("sim_seed", "sim_config", "benchmark"))
    sim_seed = metadata.get("sim_seed")
    sim_config = metadata.get("sim_config")
    return {
        "trace": str(trace_path),
        "level": ReplayLevel.SIMULATOR_REPLAY.value,
        "sim_seed": sim_seed,
        "sim_config": sim_config,
        "benchmark": metadata.get("benchmark"),
        "deterministic_replay_available": sim_seed is not None and sim_config is not None,
    }


def _l3_shadow_hardware(
    trace_path: str | Path,
    events: list[dict[str, Any]],
    envelope: dict[str, Any],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") not in {"raw_action", "postprocessed_action"}:
            continue
        value = event.get("value")
        if not isinstance(value, list):
            continue
        flattened = _flatten_numbers(value)
        max_abs = envelope.get("max_abs_action")
        if max_abs is not None:
            for dim, action_value in enumerate(flattened):
                if abs(action_value) > float(max_abs):
                    violations.append({
                        "step": event.get("step"),
                        "event_type": event.get("event_type"),
                        "dimension": dim,
                        "value": action_value,
                        "limit": float(max_abs),
                    })
    return {
        "trace": str(trace_path),
        "level": ReplayLevel.SHADOW_HARDWARE_REPLAY.value,
        "shadow_passed": not violations,
        "violations": violations,
    }


def _first_with_any(events: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    for event in events:
        if any(key in event for key in keys):
            return event
    return {}


def _load_envelope(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numbers(item))
        return out
    return []
