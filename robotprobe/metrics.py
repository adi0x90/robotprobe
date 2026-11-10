from __future__ import annotations

import math
from typing import Any


DEFAULT_METRICS = (
    "action_energy",
    "jerk",
    "trajectory_deviation",
    "recovery_latency",
    "exposure_window",
    "near_miss",
)


def compute_metric_summary(
    clean_events: list[dict[str, Any]],
    perturbed_events: list[dict[str, Any]],
    *,
    metric_names: list[str] | tuple[str, ...],
    first_divergence_step: int | None = None,
    manifest: Any = None,
) -> dict[str, Any]:
    clean_actions = _series(clean_events, "postprocessed_action")
    perturbed_actions = _series(perturbed_events, "postprocessed_action")
    metrics = {
        "clean": {},
        "perturbed": {},
        "delta": {},
    }
    requested = set(metric_names)

    if "action_energy" in requested:
        metrics["clean"]["action_energy"] = _action_energy(clean_actions)
        metrics["perturbed"]["action_energy"] = _action_energy(perturbed_actions)
        metrics["delta"]["action_energy"] = (
            metrics["perturbed"]["action_energy"] - metrics["clean"]["action_energy"]
        )

    if "jerk" in requested:
        metrics["clean"]["jerk"] = _jerk(clean_actions)
        metrics["perturbed"]["jerk"] = _jerk(perturbed_actions)
        metrics["delta"]["jerk"] = metrics["perturbed"]["jerk"] - metrics["clean"]["jerk"]

    if "trajectory_deviation" in requested:
        metrics["delta"]["trajectory_deviation"] = _trajectory_deviation(clean_actions, perturbed_actions)

    if "recovery_latency" in requested:
        metrics["delta"]["recovery_latency"] = _recovery_latency(
            clean_actions,
            perturbed_actions,
            first_divergence_step=first_divergence_step,
        )

    if "exposure_window" in requested:
        metrics["delta"]["exposure_window"] = _exposure_window(manifest)

    if "near_miss" in requested:
        metrics["delta"]["near_miss_score"] = _near_miss_score(
            clean_events,
            perturbed_events,
            trajectory_deviation=metrics["delta"].get("trajectory_deviation"),
            action_energy_delta=metrics["delta"].get("action_energy"),
        )

    return metrics


def _series(events: list[dict[str, Any]], event_type: str) -> list[tuple[int, list[float]]]:
    series: list[tuple[int, list[float]]] = []
    for event in events:
        if event.get("event_type") != event_type:
            continue
        values = _flatten_numbers(event.get("value"))
        if values:
            series.append((int(event.get("step", 0)), values))
    return sorted(series, key=lambda item: item[0])


def _action_energy(series: list[tuple[int, list[float]]]) -> float:
    return float(sum(sum(value * value for value in values) for _, values in series))


def _jerk(series: list[tuple[int, list[float]]]) -> float:
    if len(series) < 4:
        return 0.0
    vectors = [values for _, values in series]
    total = 0.0
    for idx in range(3, len(vectors)):
        jerk = [
            vectors[idx][dim]
            - 3 * vectors[idx - 1][dim]
            + 3 * vectors[idx - 2][dim]
            - vectors[idx - 3][dim]
            for dim in range(min(map(len, vectors[idx - 3:idx + 1])))
        ]
        total += _norm(jerk)
    return float(total)


def _trajectory_deviation(
    clean: list[tuple[int, list[float]]],
    perturbed: list[tuple[int, list[float]]],
) -> float:
    clean_by_step = dict(clean)
    perturbed_by_step = dict(perturbed)
    shared = sorted(set(clean_by_step) & set(perturbed_by_step))
    if not shared:
        return 0.0
    return float(sum(_distance(clean_by_step[step], perturbed_by_step[step]) for step in shared) / len(shared))


def _recovery_latency(
    clean: list[tuple[int, list[float]]],
    perturbed: list[tuple[int, list[float]]],
    *,
    first_divergence_step: int | None,
) -> int | None:
    if first_divergence_step is None:
        return None
    clean_by_step = dict(clean)
    perturbed_by_step = dict(perturbed)
    for step in sorted(set(clean_by_step) & set(perturbed_by_step)):
        if step <= first_divergence_step:
            continue
        if _distance(clean_by_step[step], perturbed_by_step[step]) <= 1e-6:
            return step - first_divergence_step
    return None


def _exposure_window(manifest: Any) -> int | None:
    if manifest is None:
        return None
    start = manifest.window.get("start_step")
    end = manifest.window.get("end_step")
    if isinstance(start, int) and isinstance(end, int):
        return max(0, end - start)
    return None


def _near_miss_score(
    clean_events: list[dict[str, Any]],
    perturbed_events: list[dict[str, Any]],
    *,
    trajectory_deviation: float | None,
    action_energy_delta: float | None,
) -> float:
    clean_outcome = _last_event(clean_events, "outcome")
    perturbed_outcome = _last_event(perturbed_events, "outcome")
    if clean_outcome.get("success") is not True or perturbed_outcome.get("success") is not True:
        return 0.0
    clean_steps = clean_outcome.get("steps") or 0
    perturbed_steps = perturbed_outcome.get("steps") or 0
    step_penalty = max(0.0, (float(perturbed_steps) / float(clean_steps) - 1.0)) if clean_steps else 0.0
    trajectory_penalty = min(1.0, float(trajectory_deviation or 0.0))
    energy_penalty = min(1.0, max(0.0, float(action_energy_delta or 0.0)) / 10.0)
    return float(min(1.0, step_penalty + trajectory_penalty + energy_penalty))


def _last_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("event_type") == event_type]
    return matches[-1] if matches else {}


def _flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numbers(item))
        return out
    return []


def _distance(left: list[float], right: list[float]) -> float:
    dims = min(len(left), len(right))
    if dims == 0:
        return 0.0
    return _norm([left[idx] - right[idx] for idx in range(dims)])


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))
