from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robotsec_range.challenge_spec import ChallengeSpec


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    diagnosis: dict[str, Any]
    rule_match_score: float
    failures: list[str] = field(default_factory=list)


def check_summary(challenge: ChallengeSpec, summary: dict[str, Any]) -> CheckResult:
    failures: list[str] = []
    hypothesis = summary.get("root_cause_hypothesis", {})
    evidence = summary.get("evidence", {})

    for key, expected in challenge.diagnosis.items():
        actual = hypothesis.get(key)
        if actual != expected:
            failures.append(f"diagnosis.{key}: expected {expected!r}, got {actual!r}")

    for key, expected in challenge.required_evidence.items():
        if key == "min_step_inflation":
            actual = evidence.get("outcome_delta", {}).get("step_inflation")
            if actual is None or actual < expected:
                failures.append(f"evidence.step_inflation: expected >= {expected!r}, got {actual!r}")
            continue
        if key.startswith("min_metric_delta_"):
            metric_name = key.removeprefix("min_metric_delta_")
            actual = summary.get("metrics", {}).get("delta", {}).get(metric_name)
            if actual is None or actual < expected:
                failures.append(f"metrics.delta.{metric_name}: expected >= {expected!r}, got {actual!r}")
            continue

        actual = evidence.get(key)
        if actual != expected:
            failures.append(f"evidence.{key}: expected {expected!r}, got {actual!r}")

    return CheckResult(
        passed=not failures,
        diagnosis=dict(hypothesis),
        rule_match_score=float(hypothesis.get("rule_match_score", 0.0)),
        failures=failures,
    )
