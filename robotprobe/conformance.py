from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robotprobe.schema import validate_trace_file


# Versioned failure-semantics conformance profile.
#
# A frozen, named, versioned contract describing what a RobotProbe trace must carry to be a
# *reproducible, contract-checkable incident* rather than just a well-formed log. Tools can target a
# named level and the validator gives a deterministic verdict.
SPEC_NAME = "robotprobe.failure-semantics"
SPEC_VERSION = "0"
SPEC_ID = f"{SPEC_NAME}/v{SPEC_VERSION}"

# Ordered profile. Each level subsumes the ones before it.
#   L0  WELL_FORMED            - schema-valid, single run, declared replay level.
#   L1  REPRODUCIBLE_INCIDENT  - carries the determinism + provenance + terminal-verdict metadata
#                                needed to deterministically replay and time-travel the episode.
#   L2  CONTRACT_CHECKABLE     - carries (or declares) the spec/contract the incident is judged
#                                against, so the verdict is auditable, not asserted.
CONFORMANCE_LEVELS: tuple[str, ...] = (
    "L0_WELL_FORMED",
    "L1_REPRODUCIBLE_INCIDENT",
    "L2_CONTRACT_CHECKABLE",
)

# The level a trace must reach by default. L0 only means "parseable", which is not what this
# profile exists to certify, so an unspecified target means L1 rather than "anything".
DEFAULT_TARGET_LEVEL = "L1_REPRODUCIBLE_INCIDENT"


@dataclass(frozen=True)
class ConformanceCheck:
    level: str
    requirement: str
    passed: bool
    message: str


@dataclass(frozen=True)
class ConformanceReport:
    path: str
    spec: str
    target_level: str | None
    achieved_level: str | None
    conformant: bool
    run_id: str | None
    event_count: int
    checks: list[ConformanceCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def check_conformance_file(path: str | Path, *, target_level: str | None = None) -> ConformanceReport:
    if target_level is not None and target_level not in CONFORMANCE_LEVELS:
        raise ValueError(f"unknown conformance level: {target_level}")
    target_level = target_level or DEFAULT_TARGET_LEVEL

    events = _read_events(path)
    trace_report = validate_trace_file(path)
    run_id = trace_report.run_ids[0] if len(trace_report.run_ids) == 1 else None

    checks: list[ConformanceCheck] = []
    checks.extend(_check_l0(trace_report))
    checks.extend(_check_l1(events))
    checks.extend(_check_l2(events))

    achieved_level = _achieved_level(checks)
    conformant = _is_conformant(achieved_level, target_level)

    return ConformanceReport(
        path=str(path),
        spec=SPEC_ID,
        target_level=target_level,
        achieved_level=achieved_level,
        conformant=conformant,
        run_id=run_id,
        event_count=trace_report.event_count,
        checks=checks,
    )


def _check_l0(trace_report: Any) -> list[ConformanceCheck]:
    level = "L0_WELL_FORMED"
    error_count = sum(1 for issue in trace_report.issues if issue.severity == "error")
    return [
        ConformanceCheck(
            level,
            "well_formed_trace",
            trace_report.valid,
            "trace has no schema errors" if trace_report.valid else f"{error_count} schema error(s)",
        ),
        ConformanceCheck(
            level,
            "single_run_id",
            len(trace_report.run_ids) == 1,
            f"run_ids={trace_report.run_ids}",
        ),
        ConformanceCheck(
            level,
            "replay_level_declared",
            len(trace_report.replay_levels) >= 1,
            f"replay_levels={trace_report.replay_levels}",
        ),
    ]


def _check_l1(events: list[dict[str, Any]]) -> list[ConformanceCheck]:
    level = "L1_REPRODUCIBLE_INCIDENT"
    has_seed = any(_is_int(event.get("seed")) for event in events)
    manifest_pinned = any(event.get("intervention_manifest_hash") for event in events)
    is_clean_baseline = any(event.get("scenario") == "clean" for event in events)
    provenance = manifest_pinned or is_clean_baseline
    has_outcome = any(event.get("event_type") == "outcome" for event in events)
    monotonic, bad_step = _monotonic_steps(events)
    return [
        ConformanceCheck(
            level,
            "run_seed_present",
            has_seed,
            "integer run seed present for deterministic replay" if has_seed else "no integer run seed found",
        ),
        ConformanceCheck(
            level,
            "provenance_pinned",
            provenance,
            "perturbation provenance pinned (intervention_manifest_hash or clean baseline)"
            if provenance
            else "no intervention_manifest_hash and scenario is not 'clean'",
        ),
        ConformanceCheck(
            level,
            "terminal_outcome_present",
            has_outcome,
            "terminal outcome event present" if has_outcome else "no outcome event (no terminal verdict)",
        ),
        ConformanceCheck(
            level,
            "monotonic_steps",
            monotonic,
            "step indices are non-decreasing (time-travel well-defined)"
            if monotonic
            else f"step decreases to {bad_step}",
        ),
    ]


def _check_l2(events: list[dict[str, Any]]) -> list[ConformanceCheck]:
    level = "L2_CONTRACT_CHECKABLE"
    has_contract_event = any(event.get("event_type") == "contract_check" for event in events)
    declares_contracts = any(
        isinstance(event.get("contracts"), list) and event.get("contracts") for event in events
    )
    bound = has_contract_event or declares_contracts
    return [
        ConformanceCheck(
            level,
            "contract_bound",
            bound,
            "incident carries contract_check evidence or a declared contracts list"
            if bound
            else "no contract_check events and no declared contracts list",
        ),
    ]


def _achieved_level(checks: list[ConformanceCheck]) -> str | None:
    achieved: str | None = None
    for level in CONFORMANCE_LEVELS:
        if _level_passed(checks, level):
            achieved = level
        else:
            break
    return achieved


def _level_passed(checks: list[ConformanceCheck], level: str) -> bool:
    level_checks = [check for check in checks if check.level == level]
    return bool(level_checks) and all(check.passed for check in level_checks)


def _is_conformant(achieved_level: str | None, target_level: str | None) -> bool:
    if achieved_level is None:
        return False
    target = target_level or DEFAULT_TARGET_LEVEL
    return CONFORMANCE_LEVELS.index(achieved_level) >= CONFORMANCE_LEVELS.index(target)


def _monotonic_steps(events: list[dict[str, Any]]) -> tuple[bool, int | None]:
    last: int | None = None
    for event in events:
        step = event.get("step")
        if not _is_int(step):
            continue
        if last is not None and step < last:
            return False, step
        last = step
    return True, None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_events(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events
