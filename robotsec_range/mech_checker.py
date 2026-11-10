from __future__ import annotations

from typing import Any

from robotsec_range.checker import CheckResult
from robotsec_range.challenge_spec import ChallengeSpec


def check_mech_report(challenge: ChallengeSpec, report: dict[str, Any]) -> CheckResult:
    if not isinstance(report, dict):
        raise ValueError("mech report must be an object")
    failures: list[str] = []
    if report.get("synthetic") is not False:
        failures.append("mech.provenance: synthetic or unknown captures are not model evidence")
    if report.get("evidence_eligible") is not True:
        failures.append("mech.evidence_eligible: expected true")
    if report.get("comparison_complete") is not True:
        failures.append("mech.comparison_complete: expected true")
    required = challenge.required_evidence
    for key, expected in required.items():
        if key == "min_evidence_level":
            actual = report.get("evidence_level")
            if actual is None or int(actual) < int(expected):
                failures.append(f"mech.evidence_level: expected >= {expected!r}, got {actual!r}")
            continue
        if key == "internal_divergence_precedes_action_divergence":
            actual = _internal_divergence_precedes_action(report)
            if actual != expected:
                failures.append(f"mech.internal_divergence_precedes_action_divergence: expected {expected!r}, got {actual!r}")
            continue
        if key == "min_internal_lead_time":
            actual = _max_internal_lead_time(report)
            if actual is None or int(actual) < int(expected):
                failures.append(f"mech.internal_lead_time: expected >= {expected!r}, got {actual!r}")
            continue
        if key == "model_root_cause_claim_allowed":
            actual = bool(report.get("model_root_cause_claim_allowed"))
            if actual != expected:
                failures.append(f"mech.model_root_cause_claim_allowed: expected {expected!r}, got {actual!r}")
            continue
        if key == "feature_promotions_allowed":
            actual = _any_feature_promotion_allowed(report)
            if actual != expected:
                failures.append(f"mech.feature_promotions_allowed: expected {expected!r}, got {actual!r}")
            continue
        actual = report.get(key)
        if actual != expected:
            failures.append(f"mech.{key}: expected {expected!r}, got {actual!r}")
    return CheckResult(
        passed=not failures,
        diagnosis={},
        rule_match_score=0.0,
        failures=failures,
    )


def _internal_divergence_precedes_action(report: dict[str, Any]) -> bool:
    for layer in report.get("activation_layers", []):
        lead_time = layer.get("lead_time_steps")
        if layer.get("activation_diff_significant") and lead_time is not None and int(lead_time) > 0:
            return True
    return False


def _max_internal_lead_time(report: dict[str, Any]) -> int | None:
    values = [
        int(layer["lead_time_steps"])
        for layer in report.get("activation_layers", [])
        if layer.get("lead_time_steps") is not None
    ]
    return max(values) if values else None


def _any_feature_promotion_allowed(report: dict[str, Any]) -> bool:
    feature_diff = report.get("feature_diff") or {}
    return any(bool(feature.get("promotion_allowed")) for feature in feature_diff.get("features", []))
