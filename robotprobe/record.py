from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from robotprobe.core import ReplayLevel, load_intervention_manifest


@dataclass(frozen=True)
class TraceSpec:
    path: Path
    run_id: str
    target: str
    scenario: str
    seed: int | None = None
    baseline_run_id: str | None = None
    perturbation_group_id: str | None = None
    intervention_manifest_ref: str | Path | None = None


class TraceRecorder:
    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        target: str,
        scenario: str,
        baseline_run_id: str | None = None,
        perturbation_group_id: str | None = None,
        counterfactual_run_id: str | None = None,
        intervention_manifest_ref: str | Path | None = None,
        replay_level: ReplayLevel = ReplayLevel.TRACE_PLAYBACK,
        seed: int | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = None
        self.context = {
            "run_id": run_id,
            "target": target,
            "scenario": scenario,
            "baseline_run_id": baseline_run_id,
            "perturbation_group_id": perturbation_group_id,
            "counterfactual_run_id": counterfactual_run_id,
            "replay_level": replay_level.value,
        }
        if seed is not None:
            self.context["seed"] = seed
        if intervention_manifest_ref is not None:
            manifest = load_intervention_manifest(intervention_manifest_ref)
            self.context["intervention_manifest_ref"] = str(intervention_manifest_ref)
            self.context["intervention_manifest_hash"] = manifest.digest

    def __enter__(self) -> "TraceRecorder":
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def event(self, step: int, event_type: str, layer: str, **fields: Any) -> None:
        record = {
            **self.context,
            "step": step,
            "event_type": event_type,
            "layer": layer,
            **fields,
        }
        self._write(record)

    def contract_check(
        self,
        *,
        step: int,
        contract: str,
        check: str,
        status: str,
        severity: float,
        evidence_refs: list[str],
    ) -> None:
        self.event(
            step,
            "contract_check",
            "contract",
            contract=contract,
            check=check,
            status=status,
            severity=severity,
            evidence_refs=evidence_refs,
        )

    def _write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("TraceRecorder must be used as a context manager or opened first")
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
