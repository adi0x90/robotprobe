from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robotprobe.schema import validate_trace_file
from robotprobe.summarize import summarize_trace_pair
from robotsec_range.challenge_spec import ChallengeSpec


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    diagnosis: dict[str, Any]
    rule_match_score: float
    failures: list[str] = field(default_factory=list)


def check_summary(
    challenge: ChallengeSpec,
    summary: dict[str, Any],
    *,
    clean_trace: str | Path | None = None,
    perturbed_trace: str | Path | None = None,
    intervention_manifest_ref: str | Path | None = None,
    metrics: list[str] | tuple[str, ...] | None = None,
) -> CheckResult:
    """Recompute a challenge verdict from validated traces and check the supplied summary.

    The traces establish consistency with the recorded evidence, not its authenticity. Trusted
    capture, signing, or another provenance mechanism remains a separate requirement.
    """
    if not isinstance(summary, dict):
        return CheckResult(False, {}, 0.0, ["summary: expected a JSON object"])

    clean_path = _evidence_path(clean_trace, challenge, "clean_trace")
    perturbed_path = _evidence_path(perturbed_trace, challenge, "perturbed_trace")
    missing = [
        f"evidence.{name}: required for independent recomputation"
        for name, path in (("clean_trace", clean_path), ("perturbed_trace", perturbed_path))
        if path is None
    ]
    if missing:
        return CheckResult(False, {}, 0.0, missing)

    failures: list[str] = []
    for label, path in (("clean_trace", clean_path), ("perturbed_trace", perturbed_path)):
        try:
            validation = validate_trace_file(path)
        except OSError as exc:
            failures.append(f"evidence.{label}: {exc}")
            continue
        if not validation.valid:
            failures.extend(
                f"evidence.{label}:{issue.line}.{issue.field}: {issue.message}"
                for issue in validation.issues
                if issue.severity == "error"
            )
    if failures:
        return CheckResult(False, {}, 0.0, failures)

    manifest_path = _manifest_path(intervention_manifest_ref, challenge)
    computed_metrics = _required_metrics(challenge, metrics)
    try:
        computed = summarize_trace_pair(
            clean_path,
            perturbed_path,
            intervention_manifest_ref=manifest_path,
            metrics=sorted(computed_metrics),
        )
    except (OSError, ValueError, KeyError) as exc:
        return CheckResult(False, {}, 0.0, [f"evidence.recompute: {exc}"])

    disagreements: list[str] = []
    for section in ("root_cause_hypothesis", "evidence", "metrics", "replay_level"):
        if section in summary and section in computed:
            _compare_submitted(summary[section], computed[section], section, disagreements)

    hypothesis = computed["root_cause_hypothesis"]
    evidence = computed["evidence"]
    failures.extend(disagreements)
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
            actual = computed.get("metrics", {}).get("delta", {}).get(metric_name)
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


def _evidence_path(
    supplied: str | Path | None,
    challenge: ChallengeSpec,
    key: str,
) -> Path | None:
    if supplied is not None:
        return Path(supplied)
    ref = challenge.metadata.get("fixture_refs", {}).get(key)
    return _PROJECT_ROOT / ref if ref else None


def _manifest_path(
    supplied: str | Path | None,
    challenge: ChallengeSpec,
) -> Path | None:
    if supplied is not None:
        return Path(supplied)
    ref = challenge.metadata.get("fixture_refs", {}).get("intervention_manifest")
    if ref:
        return _PROJECT_ROOT / ref

    fault = challenge.diagnosis.get("fault")
    if not fault:
        return None
    for directory in (
        _PROJECT_ROOT / "interventions",
        _PROJECT_ROOT / "examples" / "packs" / "libero-action-contracts" / "interventions",
    ):
        for candidate in sorted(directory.glob("*.yaml")):
            if f"name: {fault}" in candidate.read_text(encoding="utf-8"):
                return candidate
    return None


def _required_metrics(
    challenge: ChallengeSpec,
    metrics: list[str] | tuple[str, ...] | None,
) -> set[str]:
    required = set(metrics or ())
    for key in challenge.required_evidence:
        if not key.startswith("min_metric_delta_"):
            continue
        metric_name = key.removeprefix("min_metric_delta_")
        if metric_name == "near_miss_score":
            required.update(("near_miss", "trajectory_deviation", "action_energy"))
        else:
            required.add(metric_name)
    return required


def _compare_submitted(
    submitted: Any,
    computed: Any,
    path: str,
    failures: list[str],
) -> None:
    if isinstance(submitted, dict) and isinstance(computed, dict):
        for key, value in submitted.items():
            if key in computed:
                _compare_submitted(value, computed[key], f"{path}.{key}", failures)
        return
    if submitted != computed:
        failures.append(f"summary.{path}: submitted {submitted!r}, but traces compute {computed!r}")
