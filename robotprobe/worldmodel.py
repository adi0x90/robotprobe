"""World-model faithfulness verification.

Diagnoses whether *imagined* evidence (a generated subgoal / imagined view emitted as a
``world_model_output`` event) matched reality before a downstream policy or reasoner committed to it.
Closed-loop imagined rollouts are known to suffer hallucination and long-horizon error accumulation
that silently corrupt the signal the policy acts on.

Scope: this compares *declared* fields already present in the trace (``predicted_*`` on the imagined
event versus the realized ``observation`` / ``outcome``). It does not render or compare pixels and
does not prove a generated image is unfaithful by inspection. A divergence is evidence about the
recorded declared fields, not a universal hallucination verdict. Lead time is the window (in steps)
between the imagined evidence and the next policy action: the budget a runtime monitor would have to
catch the divergence before the policy commits.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def check_world_model_faithfulness(
    trace_path: str | Path, *, state_tolerance: float = 1e-3, divergence_threshold: float = 0.5
) -> dict[str, Any]:
    events = _read_trace(trace_path)
    imagined = [e for e in events if e.get("event_type") == "world_model_output"]
    final_outcome = _last(events, "outcome")

    records: list[dict[str, Any]] = []
    for event in imagined:
        step = _as_int(event.get("step"))
        realized_obs = _first_at_or_after(events, "observation", step)
        next_action = _first_at_or_after(events, "postprocessed_action", step)

        signals: list[str] = []

        # 1) predicted vs realized success (boolean prediction correctness).
        if "predicted_success" in event and "success" in final_outcome:
            if bool(event["predicted_success"]) != bool(final_outcome["success"]):
                signals.append("success_mismatch")

        # 2) predicted_state vs realized observation state (numeric L2 divergence).
        state_divergence = None
        predicted_state = _numeric(event.get("predicted_state"))
        realized_state = _numeric(realized_obs.get("state") if realized_obs else None)
        if predicted_state is not None and realized_state is not None:
            state_divergence = _l2(predicted_state, realized_state)
            if state_divergence > state_tolerance:
                signals.append("state_divergence")

        # 3) an explicit upstream-provided divergence scalar, if present.
        declared = event.get("divergence")
        if isinstance(declared, (int, float)) and not isinstance(declared, bool):
            if float(declared) > divergence_threshold:
                signals.append("declared_divergence")

        lead_time = (
            (_as_int(next_action.get("step")) - step)
            if (next_action and step is not None)
            else None
        )
        records.append(
            {
                "step": step,
                "request_ref": event.get("request_ref"),
                "generated_image_ref": event.get("generated_image_ref"),
                "faithful": not signals,
                "divergence_signals": signals,
                "state_divergence": state_divergence,
                "lead_time_steps": lead_time,
                "acts_on_imagined_evidence": next_action is not None,
            }
        )

    divergent = [r for r in records if not r["faithful"]]
    lead_times = [r["lead_time_steps"] for r in divergent if r["lead_time_steps"] is not None]
    n = len(records)
    return {
        "trace_path": str(trace_path),
        "n_imagined_events": n,
        "n_faithful": n - len(divergent),
        "n_divergent": len(divergent),
        # None, not 1.0: a trace with no imagined events has no faithfulness to report.
        "faithfulness_rate": ((n - len(divergent)) / n if n else None),
        "min_divergent_lead_time": (min(lead_times) if lead_times else None),
        "records": records,
        "claim_boundary": (
            "Compares declared predicted_* fields against realized observation/outcome fields in the "
            "recorded trace; does not compare pixels or prove generative hallucination by inspection."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# World-Model Faithfulness Report",
        "",
        f"- trace: `{report['trace_path']}`",
        f"- imagined events: {report['n_imagined_events']}",
        f"- faithful: {report['n_faithful']} / divergent: {report['n_divergent']}",
        f"- faithfulness rate: {report['faithfulness_rate']:.3f}",
        f"- tightest catch window (min divergent lead time): {report['min_divergent_lead_time']}",
        "",
        "| step | faithful | signals | state divergence | lead time | acts on imagined |",
        "|------|----------|---------|------------------|-----------|------------------|",
    ]
    for r in report["records"]:
        lines.append(
            f"| {r['step']} | {r['faithful']} | {','.join(r['divergence_signals']) or '-'} | "
            f"{r['state_divergence']} | {r['lead_time_steps']} | {r['acts_on_imagined_evidence']} |"
        )
    lines += ["", f"> {report['claim_boundary']}", ""]
    return "\n".join(lines)


def _first_at_or_after(events: list[dict[str, Any]], event_type: str, step: int | None) -> dict[str, Any] | None:
    if step is None:
        return None
    for event in events:
        if event.get("event_type") == event_type and _as_int(event.get("step")) is not None:
            if _as_int(event["step"]) >= step:
                return event
    return None


def _last(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    matches = [e for e in events if e.get("event_type") == event_type]
    return matches[-1] if matches else {}


def _numeric(value: Any) -> list[float] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return [float(v) for v in value]
    return None


def _l2(a: list[float], b: list[float]) -> float:
    dims = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dims)))


def _as_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
