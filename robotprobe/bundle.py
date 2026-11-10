from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from robotprobe.report import build_incident_report
from robotprobe.summarize import summarize_trace_pair
from robotsec_range.challenge_spec import ChallengeSpec, load_builtin_challenge
from robotsec_range.checker import check_summary


def create_incident_bundle(
    *,
    clean_trace: str | Path,
    perturbed_trace: str | Path,
    manifest: str | Path | None,
    out_dir: str | Path,
    challenge: str | Path | ChallengeSpec | None = None,
    metrics: list[str] | tuple[str, ...] | None = None,
    title: str = "RobotProbe Incident Report",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces_dir = out / "traces"
    manifests_dir = out / "manifests"
    traces_dir.mkdir(exist_ok=True)
    manifests_dir.mkdir(exist_ok=True)

    clean_ref = traces_dir / Path(clean_trace).name
    perturbed_ref = traces_dir / Path(perturbed_trace).name
    shutil.copy2(clean_trace, clean_ref)
    shutil.copy2(perturbed_trace, perturbed_ref)

    manifest_ref = None
    if manifest is not None:
        manifest_ref = manifests_dir / Path(manifest).name
        shutil.copy2(manifest, manifest_ref)
        # The source trace recorded wherever the manifest lived on the author's machine. Inside a
        # bundle the manifest is copied next to the traces, so rewrite the reference to the
        # bundle-relative path rather than shipping an absolute one.
        bundle_relative = f"manifests/{manifest_ref.name}"
        _rewrite_manifest_ref(clean_ref, bundle_relative)
        _rewrite_manifest_ref(perturbed_ref, bundle_relative)

    summary = summarize_trace_pair(
        clean_trace,
        perturbed_trace,
        intervention_manifest_ref=manifest,
        metrics=metrics,
    )
    summary_path = out / "summary.json"
    causal_diff_path = out / "causal_diff.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    causal_diff_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    check_path = None
    check_data = None
    if challenge is not None:
        challenge_spec = _load_challenge(challenge)
        check = check_summary(challenge_spec, summary)
        check_data = {
            "passed": check.passed,
            "diagnosis": check.diagnosis,
            "rule_match_score": check.rule_match_score,
            "failures": check.failures,
        }
        check_path = out / "check.json"
        check_path.write_text(json.dumps(check_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_incident_report(summary, check=check_data, title=title)
    report_path = out / "report.md"
    report_path.write_text(report, encoding="utf-8")

    bundle = {
        "summary_ref": summary_path.name,
        "causal_diff_ref": causal_diff_path.name,
        "check_ref": check_path.name if check_path else None,
        "report_ref": report_path.name,
        "clean_trace_ref": str(Path("traces") / clean_ref.name),
        "perturbed_trace_ref": str(Path("traces") / perturbed_ref.name),
        "manifest_ref": str(Path("manifests") / manifest_ref.name) if manifest_ref else None,
        "replay_level": summary.get("replay_level"),
        "metrics": list(metrics or []),
    }
    bundle_path = out / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def _rewrite_manifest_ref(trace_path: Path, bundle_relative: str) -> None:
    """Replace absolute intervention_manifest_ref values with the bundle-relative path."""
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    changed = False
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        event = json.loads(line)
        if "intervention_manifest_ref" in event:
            event["intervention_manifest_ref"] = bundle_relative
            changed = True
            out.append(json.dumps(event, sort_keys=True))
        else:
            out.append(line)
    if changed:
        trace_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _load_challenge(ref: str | Path | ChallengeSpec) -> ChallengeSpec:
    if isinstance(ref, ChallengeSpec):
        return ref
    path = Path(ref)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChallengeSpec(
            id=data["id"],
            diagnosis=data["diagnosis"],
            required_evidence=data.get("required_evidence", {}),
            metadata=data.get("metadata", {}),
        )
    return load_builtin_challenge(str(ref))
