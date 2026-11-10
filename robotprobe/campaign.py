"""Fault-injection campaign engine.

Maps the distributed-systems fault-injection model (Jepsen) onto RobotProbe's primitives:

- nemesis   -> a scheduled list of sim-only faults applied to a recorded baseline trace;
- history   -> the synthesized perturbed traces;
- checker   -> the contract DSL run post-hoc over each synthesized trace, decoupled from injection.

Scope: faults are applied to the *recorded* baseline action / observation / instruction stream.
Outcomes are not re-simulated. Contract verdicts and consequence metrics describe the synthesized
stream and its diagnosis, not a fresh policy or simulator rollout. Contracts that depend on
sim-derived state (e.g. ``keepout_violated``) only fire if that state is already present in the
baseline.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robotprobe.attacks import ActionFaultAttack
from robotprobe.contracts import check_contract_file
from robotprobe.summarize import summarize_trace_pair

CLAIM_BOUNDARY = (
    "Faults are applied to the recorded baseline action/observation/instruction stream; outcomes "
    "are not re-simulated. Contract verdicts and consequence metrics describe the synthesized "
    "stream and its diagnosis, not a fresh rollout."
)

CHANNEL_LAYER = {"action": "action_postprocess", "instruction": "instruction", "observation": "observation"}


@dataclass(frozen=True)
class FaultSpec:
    name: str
    channel: str = "action"  # action | observation | instruction
    mode: str = "zero_dims"
    dims: tuple[int, ...] = (0, 1, 2)
    scale: float = 0.0
    value: float = 0.0
    offset: float = 0.0
    start: int = 0
    end: int | None = None
    payload: str = ""
    seed: int = 0

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FaultSpec":
        return FaultSpec(
            name=str(data["name"]),
            channel=str(data.get("channel", "action")),
            mode=str(data.get("mode", "zero_dims")),
            dims=tuple(int(d) for d in data.get("dims", (0, 1, 2))),
            scale=float(data.get("scale", 0.0)),
            value=float(data.get("value", 0.0)),
            offset=float(data.get("offset", 0.0)),
            start=int(data.get("start", 0)),
            end=None if data.get("end") is None else int(data["end"]),
            payload=str(data.get("payload", "")),
            seed=int(data.get("seed", 0)),
        )


@dataclass
class CampaignSpec:
    id: str
    baseline: str
    faults: list[FaultSpec]
    contracts: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)


def load_campaign_file(path: str | Path) -> tuple[CampaignSpec, Path]:
    campaign_path = Path(path)
    data = json.loads(campaign_path.read_text(encoding="utf-8"))
    spec = CampaignSpec(
        id=str(data.get("id", campaign_path.stem)),
        baseline=str(data["baseline"]),
        faults=[FaultSpec.from_dict(f) for f in data.get("faults", [])],
        contracts=[str(c) for c in data.get("contracts", [])],
        metrics=[str(m) for m in data.get("metrics", [])],
    )
    return spec, campaign_path.parent


def run_campaign(spec: CampaignSpec, *, base_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    base = Path(base_dir)
    out = Path(out_dir)
    (out / "traces").mkdir(parents=True, exist_ok=True)

    baseline_path = _resolve(base, spec.baseline)
    baseline_events = _read_trace(baseline_path)
    contract_paths = [_resolve(base, c) for c in spec.contracts]

    fault_reports: list[dict[str, Any]] = []
    for fault in spec.faults:
        perturbed_events = _apply_fault(baseline_events, fault)
        perturbed_path = out / "traces" / f"{fault.name}.jsonl"
        _write_trace(perturbed_path, perturbed_events)

        manifest_path = out / "manifests" / f"{fault.name}.yaml"
        _write_fault_manifest(manifest_path, fault)

        summary = summarize_trace_pair(
            baseline_path,
            perturbed_path,
            intervention_manifest_ref=manifest_path,
            metrics=spec.metrics or None,
        )
        contract_results = []
        for contract_path in contract_paths:
            result = check_contract_file(contract_path, perturbed_path)
            violations = result.get("violations", [])
            contract_results.append(
                {
                    "id": result.get("contract_id"),
                    "passed": bool(result.get("passed")),
                    "violations": len(violations),
                    "max_severity": _max_severity(violations),
                }
            )
        violated_any = any(not c["passed"] for c in contract_results)
        fault_reports.append(
            {
                "name": fault.name,
                "channel": fault.channel,
                "trace": str(perturbed_path),
                "manifest": str(manifest_path),
                "diagnosis": {
                    "layer": summary["root_cause_hypothesis"]["layer"],
                    "declared_layer": summary["root_cause_hypothesis"]["declared_layer"],
                    "first_divergence_step": summary["evidence"]["first_divergence_step"],
                    "rule_match_score": summary["root_cause_hypothesis"]["rule_match_score"],
                },
                "consequence_metrics": summary.get("metrics", {}).get("delta", {}),
                "contracts": contract_results,
                "violated_any_contract": violated_any,
            }
        )

    violating = [f for f in fault_reports if f["violated_any_contract"]]
    severities = [
        c["max_severity"]
        for f in fault_reports
        for c in f["contracts"]
        if c["max_severity"] is not None
    ]
    report = {
        "campaign_id": spec.id,
        "baseline": str(baseline_path),
        "n_faults": len(fault_reports),
        "n_faults_violating_contracts": len(violating),
        "violating_faults": [f["name"] for f in violating],
        "max_contract_severity": max(severities) if severities else None,
        "contracts_checked": [str(c) for c in contract_paths],
        "metrics": list(spec.metrics),
        "faults": fault_reports,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (out / "campaign_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "campaign_report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def run_campaign_file(path: str | Path, *, out_dir: str | Path) -> dict[str, Any]:
    spec, base_dir = load_campaign_file(path)
    return run_campaign(spec, base_dir=base_dir, out_dir=out_dir)


def _apply_fault(baseline_events: list[dict[str, Any]], fault: FaultSpec) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    attack = None
    if fault.channel == "action":
        attack = ActionFaultAttack(
            mode=fault.mode,
            dims=fault.dims,
            start=fault.start,
            end=fault.end,
            scale=fault.scale,
            value=fault.value,
            offset=fault.offset,
            seed=fault.seed,
        )
        attack.reset()
    held_obs_hash: str | None = None

    for raw in baseline_events:
        event = dict(raw)
        event["baseline_run_id"] = raw.get("run_id")
        event["run_id"] = f"{raw.get('run_id', 'baseline')}__{fault.name}"
        event["scenario"] = fault.name
        etype = event.get("event_type")
        step = event.get("step", 0)

        if fault.channel == "action" and etype == "postprocessed_action" and isinstance(event.get("value"), list):
            arr = attack.perturb_action(np.asarray(event["value"], dtype=float), step=int(step))
            event["value"] = arr.tolist()
            flat = _flatten(event["value"])
            if flat:
                event["action_l2"] = math.sqrt(sum(x * x for x in flat))
                event["action_max_abs"] = max(abs(x) for x in flat)
        elif fault.channel == "instruction" and etype == "instruction" and _in_window(int(step), fault):
            event["before"] = event.get("value", event.get("after"))
            event["after"] = fault.payload
            event["value"] = fault.payload
        elif fault.channel == "observation" and etype == "observation":
            current = event.get("image_sha256")
            if _in_window(int(step), fault) and held_obs_hash is not None:
                event["image_sha256"] = held_obs_hash
            elif current is not None:
                held_obs_hash = current

        out.append(event)
    return out


def _in_window(step: int, fault: FaultSpec) -> bool:
    if step < fault.start:
        return False
    if fault.end is not None and step >= fault.end:
        return False
    return True


def _flatten(value: Any) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return []


def _max_severity(violations: list[dict[str, Any]]) -> float | None:
    severities = [v.get("severity") for v in violations if isinstance(v.get("severity"), (int, float))]
    return float(max(severities)) if severities else None


def _write_fault_manifest(path: Path, fault: FaultSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    layer = CHANNEL_LAYER.get(fault.channel, "action_postprocess")
    end = fault.end if fault.end is not None else 1000000
    path.write_text(
        f"id: CAMPAIGN.{fault.name.upper()}\n"
        f"name: {fault.name}\n"
        "target_layers:\n"
        f"  - {layer}\n"
        "contract: CampaignFault\n"
        "window:\n"
        f"  start_step: {fault.start}\n"
        f"  end_step: {end}\n"
        "safety:\n"
        "  sim_only: true\n"
        "  hardware_allowed: false\n",
        encoding="utf-8",
    )


def _resolve(base: Path, ref: str) -> Path:
    candidate = Path(ref)
    return candidate if candidate.is_absolute() else (base / candidate)


def _read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_trace(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(e, sort_keys=True) for e in events) + "\n", encoding="utf-8")


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# RobotProbe Fault Campaign: {report['campaign_id']}",
        "",
        f"- baseline: `{report['baseline']}`",
        f"- faults: {report['n_faults']}",
        f"- faults violating a contract: {report['n_faults_violating_contracts']}",
        f"- max contract severity: {report['max_contract_severity']}",
        "",
        "| fault | channel | root layer | first divergence | contract verdict |",
        "|-------|---------|-----------|------------------|------------------|",
    ]
    for fault in report["faults"]:
        verdict = "VIOLATION" if fault["violated_any_contract"] else "ok"
        lines.append(
            f"| {fault['name']} | {fault['channel']} | {fault['diagnosis']['layer']} | "
            f"{fault['diagnosis']['first_divergence_step']} | {verdict} |"
        )
    lines += ["", f"> {report['claim_boundary']}", ""]
    return "\n".join(lines)
