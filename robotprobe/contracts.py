from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robotprobe.core import _parse_manifest_yaml


OPERATORS = {
    "lt": lambda left, right: left < right,
    "lte": lambda left, right: left <= right,
    "gt": lambda left, right: left > right,
    "gte": lambda left, right: left >= right,
    "eq": lambda left, right: left == right,
    "neq": lambda left, right: left != right,
}


def check_contract_file(contract_path: str | Path, trace_path: str | Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    events = _read_trace(trace_path)
    return check_contract(contract, events, trace_path=str(trace_path), contract_path=str(contract_path))


def load_contract(path: str | Path) -> dict[str, Any]:
    return _parse_manifest_yaml(Path(path).read_text(encoding="utf-8"))


def check_contract(
    contract: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    trace_path: str = "",
    contract_path: str = "",
) -> dict[str, Any]:
    if contract.get("type") == "temporal_contract":
        return _check_temporal_contract(
            contract,
            events,
            trace_path=trace_path,
            contract_path=contract_path,
        )

    violations: list[dict[str, Any]] = []
    for event in events:
        if not _predicate_matches(contract.get("when", {}), event):
            continue
        if _predicate_matches(contract.get("require", {}), event):
            continue
        violation = dict(contract.get("violation", {}))
        violation.setdefault("contract", str(contract.get("id", "Contract")))
        violation.setdefault("check", str(contract.get("id", "contract_check")))
        violation.setdefault("severity", 1.0)
        violation["step"] = event.get("step")
        violation["event_type"] = event.get("event_type")
        violations.append(violation)

    return {
        "contract_id": contract.get("id"),
        "contract_type": contract.get("type"),
        "contract_path": contract_path,
        "trace_path": trace_path,
        "passed": not violations,
        "violations": violations,
    }


def _check_temporal_contract(
    contract: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    trace_path: str,
    contract_path: str,
) -> dict[str, Any]:
    window = dict(contract.get("window", {}))
    window_events = [event for event in events if _inside_step_window(event, window)]
    violations: list[dict[str, Any]] = []

    if "must_never" in contract:
        predicate = dict(contract["must_never"])
        for event in window_events:
            if _predicate_matches(predicate, event):
                violation = _build_violation(contract, event)
                violation["temporal_operator"] = "must_never"
                violation["window"] = window
                violations.append(violation)

    if "must_always" in contract:
        predicate = dict(contract["must_always"])
        for event in window_events:
            if not _predicate_matches(predicate, event):
                violation = _build_violation(contract, event)
                violation["temporal_operator"] = "must_always"
                violation["window"] = window
                violations.append(violation)

    if "must_eventually" in contract:
        predicate = dict(contract["must_eventually"])
        if not any(_predicate_matches(predicate, event) for event in window_events):
            violation = _build_violation(contract, window_events[-1] if window_events else {})
            violation["temporal_operator"] = "must_eventually"
            violation["window"] = window
            violations.append(violation)

    return {
        "contract_id": contract.get("id"),
        "contract_type": contract.get("type"),
        "contract_path": contract_path,
        "trace_path": trace_path,
        "passed": not violations,
        "violations": violations,
    }


def _inside_step_window(event: dict[str, Any], window: dict[str, Any]) -> bool:
    step = event.get("step")
    if not isinstance(step, (int, float)) or isinstance(step, bool):
        return False
    start = window.get("start_step")
    end = window.get("end_step")
    if isinstance(start, (int, float)) and not isinstance(start, bool) and step < start:
        return False
    if isinstance(end, (int, float)) and not isinstance(end, bool) and step > end:
        return False
    return True


def _build_violation(contract: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    violation = dict(contract.get("violation", {}))
    violation.setdefault("contract", str(contract.get("id", "Contract")))
    violation.setdefault("check", str(contract.get("id", "contract_check")))
    violation.setdefault("severity", 1.0)
    violation["step"] = event.get("step")
    violation["event_type"] = event.get("event_type")
    return violation


def _predicate_matches(predicate: dict[str, Any], event: dict[str, Any]) -> bool:
    if not predicate:
        return True
    field = predicate.get("field")
    if not field:
        raise ValueError("contract predicate requires field")
    value = _field(event, str(field))
    for op_name, expected in predicate.items():
        if op_name == "field":
            continue
        if op_name not in OPERATORS:
            raise ValueError(f"unknown contract operator: {op_name}")
        if value is None:
            return False
        if not OPERATORS[op_name](value, expected):
            return False
    return True


def _field(event: dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    current: Any = event
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, dict) and "value" in current:
            current = current["value"]
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
        return None
    return current


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
