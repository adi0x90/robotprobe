from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotprobe.mech.activation_store import ActivationStore


@dataclass(frozen=True)
class ActivationSequence:
    run_label: str
    module: str
    steps: list[int]
    values: np.ndarray
    records: list[dict[str, Any]]


class ActivationCacheView:
    """File-backed query layer over a MechProbe activation index."""

    def __init__(self, mech_dir: str | Path, *, index_path: str | Path | None = None):
        self.mech_dir = Path(mech_dir)
        self.index_path = Path(index_path) if index_path is not None else self.mech_dir / "activation_index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"missing activation index: {self.index_path}")
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(self.index, dict):
            raise ValueError("activation index must be an object")
        self.store = ActivationStore(self.mech_dir)
        captures = self.index.get("captures", [])
        if not isinstance(captures, list) or not all(isinstance(record, dict) for record in captures):
            raise ValueError("activation index captures must be a list of objects")
        self._records = [dict(record) for record in captures]
        self._by_module_step_run: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
        for record in self._records:
            if not all(key in record for key in ("module", "step", "run_label", "activation_ref", "shape", "dtype")):
                raise ValueError("activation capture is missing module, step, run_label, activation_ref, shape, or dtype")
            if type(record["step"]) is not int or record["step"] < 0:
                raise ValueError("activation capture step must be a nonnegative integer")
            module = str(record["module"])
            step = int(record["step"])
            run_label = str(record["run_label"])
            by_run = self._by_module_step_run.setdefault(module, {}).setdefault(step, {})
            if run_label in by_run:
                raise ValueError(f"duplicate activation capture for {module} step {step} run {run_label}")
            by_run[run_label] = record
        required_modules = self.index.get("required_modules", [])
        if not isinstance(required_modules, list) or not all(isinstance(name, str) for name in required_modules):
            raise ValueError("required_modules must be a list of module names")
        for module in required_modules:
            self._by_module_step_run.setdefault(module, {})

    @property
    def synthetic(self) -> bool | None:
        value = self.index.get("synthetic")
        return value if isinstance(value, bool) else None

    @property
    def format(self) -> str | None:
        value = self.index.get("format")
        return str(value) if value is not None else None

    @property
    def incident_id(self) -> str | None:
        value = self.index.get("incident_id")
        return str(value) if value is not None else None

    @property
    def component_names(self) -> list[str]:
        return sorted(self._by_module_step_run)

    @property
    def run_labels(self) -> list[str]:
        return sorted({str(record["run_label"]) for record in self._records})

    @property
    def timesteps(self) -> list[int]:
        return sorted({int(record["step"]) for record in self._records})

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.index.get("model", {}))

    def __getitem__(self, key: tuple[Any, ...]) -> Any:
        if not isinstance(key, tuple):
            raise KeyError("activation cache key must be a tuple")
        if len(key) == 2:
            module, step_or_slice = key
            if isinstance(step_or_slice, slice):
                return self.paired_sequences(str(module), step_or_slice)
            return self.get_pair(str(module), int(step_or_slice))
        if len(key) == 3:
            run_label, module, step = key
            return self.get(str(module), int(step), run_label=str(run_label))
        raise KeyError(f"unsupported activation cache key: {key!r}")

    def records(self, *, module: str | None = None, run_label: str | None = None) -> list[dict[str, Any]]:
        records = self._records
        if module is not None:
            records = [record for record in records if record.get("module") == module]
        if run_label is not None:
            records = [record for record in records if record.get("run_label") == run_label]
        return [dict(record) for record in records]

    def get(self, module: str, step: int, *, run_label: str) -> np.ndarray:
        record = self.record(module, step, run_label=run_label)
        return self._read_record(record)

    def _read_record(self, record: dict[str, Any]) -> np.ndarray:
        array = self.store.read_activation(str(record["activation_ref"]))
        location = f"{record['module']} step {record['step']} run {record['run_label']}"
        if list(array.shape) != record.get("shape"):
            raise ValueError(f"activation metadata shape mismatch for {location}")
        if str(array.dtype) != record.get("dtype"):
            raise ValueError(f"activation metadata dtype mismatch for {location}")
        if not array.size or array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
            raise ValueError(f"activation must contain nonempty finite real values for {location}")
        return array

    def comparison_coverage(self, module: str) -> dict[str, Any]:
        steps = self._by_module_step_run.get(module, {})
        paired = [step for step, runs in steps.items() if {"clean", "perturbed"} <= runs.keys()]
        unpaired = [step for step, runs in sorted(steps.items()) if not {"clean", "perturbed"} <= runs.keys()]
        return {
            "paired_step_count": len(paired),
            "unpaired_steps": unpaired,
            "comparison_complete": bool(paired) and not unpaired,
        }

    @property
    def comparison_complete(self) -> bool:
        return bool(self.component_names) and all(
            self.comparison_coverage(module)["comparison_complete"] for module in self.component_names
        )

    def paired_arrays(self, module: str, step: int) -> tuple[np.ndarray, np.ndarray]:
        clean = self.get(module, step, run_label="clean")
        perturbed = self.get(module, step, run_label="perturbed")
        if clean.shape != perturbed.shape:
            raise ValueError(f"activation shape mismatch for {module} step {step}: {clean.shape} != {perturbed.shape}")
        return clean, perturbed

    def record(self, module: str, step: int, *, run_label: str) -> dict[str, Any]:
        by_step = self._by_module_step_run.get(module)
        if by_step is None:
            raise KeyError(f"unknown activation module: {module}")
        by_run = by_step.get(int(step))
        if by_run is None:
            raise KeyError(f"no activation for module {module} at step {step}")
        record = by_run.get(run_label)
        if record is None:
            raise KeyError(f"no activation for module {module} at step {step} run_label {run_label}")
        return dict(record)

    def get_pair(self, module: str, step: int) -> dict[str, np.ndarray]:
        by_step = self._by_module_step_run.get(module)
        if by_step is None:
            raise KeyError(f"unknown activation module: {module}")
        by_run = by_step.get(int(step))
        if by_run is None:
            raise KeyError(f"no activation for module {module} at step {step}")
        return {
            run_label: self._read_record(record)
            for run_label, record in sorted(by_run.items())
        }

    def sequence(self, module: str, *, run_label: str) -> ActivationSequence:
        records = sorted(
            (
                record
                for record in self._records
                if record.get("module") == module and record.get("run_label") == run_label
            ),
            key=lambda record: int(record["step"]),
        )
        if not records:
            if module not in self._by_module_step_run:
                raise KeyError(f"unknown activation module: {module}")
            raise KeyError(f"no activations for module {module} run_label {run_label}")
        arrays = [self._read_record(record) for record in records]
        return ActivationSequence(
            run_label=run_label,
            module=module,
            steps=[int(record["step"]) for record in records],
            values=np.stack(arrays, axis=0),
            records=[dict(record) for record in records],
        )

    def paired_sequences(self, module: str, step_slice: slice | None = None) -> dict[str, ActivationSequence]:
        result = {}
        for run_label in self.run_labels:
            sequence = self.sequence(module, run_label=run_label)
            if step_slice is not None:
                selected = [
                    (step, value, record)
                    for step, value, record in zip(sequence.steps, sequence.values, sequence.records)
                    if _slice_contains(step_slice, step)
                ]
                if not selected:
                    continue
                steps, values, records = zip(*selected)
                sequence = ActivationSequence(
                    run_label=run_label,
                    module=module,
                    steps=list(steps),
                    values=np.stack(list(values), axis=0),
                    records=[dict(record) for record in records],
                )
            result[run_label] = sequence
        return result

    def paired_steps(self, module: str) -> list[int]:
        by_step = self._by_module_step_run.get(module)
        if by_step is None:
            raise KeyError(f"unknown activation module: {module}")
        return [
            step
            for step, by_run in sorted(by_step.items())
            if "clean" in by_run and "perturbed" in by_run
        ]


def build_cache_summary(mech_dir: str | Path, *, index_path: str | Path | None = None) -> dict[str, Any]:
    view = ActivationCacheView(mech_dir, index_path=index_path)
    components = []
    for module in view.component_names:
        records = view.records(module=module)
        components.append(
            {
                "module": module,
                "module_kinds": sorted({str(record.get("module_kind")) for record in records}),
                "run_labels": sorted({str(record.get("run_label")) for record in records}),
                "timesteps": sorted({int(record.get("step", 0)) for record in records}),
                "capture_count": len(records),
                "shapes": sorted({json.dumps(record.get("shape", [])) for record in records}),
                "dtypes": sorted({str(record.get("dtype")) for record in records}),
                **view.comparison_coverage(module),
            }
        )
    for component in components:
        component["shapes"] = [json.loads(shape) for shape in component["shapes"]]
    return {
        "format": "robotprobe.mech.cache_summary.v0",
        "index_path": str(view.index_path),
        "incident_id": view.incident_id,
        "model": view.model,
        "synthetic": view.synthetic,
        # Cache inspection reports metadata, not a validated tensor comparison.
        "evidence_eligible": None,
        "evidence_status": "not_assessed",
        "eligibility_reason": (
            "Cache inspection reads metadata only; tensor payloads and comparisons have not been validated. "
            "Run diff-activations to assess evidence eligibility."
        ),
        "runs": view.run_labels,
        "timesteps": view.timesteps,
        "component_count": len(components),
        "capture_count": len(view.records()),
        "components": components,
    }


def _slice_contains(step_slice: slice, step: int) -> bool:
    start = step_slice.start if step_slice.start is not None else -10**18
    stop = step_slice.stop if step_slice.stop is not None else 10**18
    stride = step_slice.step
    if step < start or step >= stop:
        return False
    if stride is None:
        return True
    return (step - start) % stride == 0
