from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np


@dataclass(frozen=True)
class ActivationCapture:
    run_label: str
    run_id: str
    step: int
    module: str
    module_kind: str
    activation_ref: str
    shape: list[int]
    dtype: str
    source_dtype: str | None = None

    def to_index_record(self) -> dict[str, Any]:
        return {
            "run_label": self.run_label,
            "run_id": self.run_id,
            "step": self.step,
            "module": self.module,
            "module_kind": self.module_kind,
            "activation_ref": self.activation_ref,
            "shape": self.shape,
            "dtype": self.dtype,
            "source_dtype": self.source_dtype or self.dtype,
        }


class ActivationStore:
    def __init__(
        self, mech_dir: str | Path, *, store_format: str = "safetensors",
        dtype: str | None = None, max_bytes_per_run: int | None = None,
    ):
        if store_format not in ("safetensors", "npy"):
            raise ValueError(f"unsupported activation store format: {store_format}")
        if dtype not in (None, "float16", "float32", "float64"):
            raise ValueError(f"unsupported activation storage dtype: {dtype}")
        if max_bytes_per_run is not None and (type(max_bytes_per_run) is not int or max_bytes_per_run <= 0):
            raise ValueError("max_bytes_per_run must be a positive integer")
        self.mech_dir = Path(mech_dir)
        self.store_format = store_format
        self.dtype = dtype
        self.max_bytes_per_run = max_bytes_per_run
        self._bytes_by_run: dict[str, int] = {}

    def write_activation(
        self,
        *,
        run_label: str,
        run_id: str,
        step: int,
        module: str,
        module_kind: str,
        activation: Any,
        source_dtype: str | None = None,
    ) -> ActivationCapture:
        if run_label not in ("clean", "perturbed"):
            raise ValueError("run_label must be clean or perturbed")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        array = np.asarray(activation, dtype=self.dtype)
        original_dtype = source_dtype or str(np.asarray(activation).dtype)
        suffix = ".safetensors" if self.store_format == "safetensors" else ".npy"
        relative = Path("activations") / run_label / _safe_module_path(module) / f"step_{step:06d}{suffix}"
        path = self.mech_dir / relative
        if self.store_format == "safetensors":
            try:
                from safetensors.numpy import save
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("safetensors storage requires the safetensors dependency") from exc
            payload = save({"activation": np.ascontiguousarray(array)})
        else:
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            payload = buffer.getvalue()
        used = self._bytes_by_run.get(run_label)
        if used is None:
            used = sum(file.stat().st_size for file in (self.mech_dir / "activations" / run_label).rglob("*") if file.is_file())
        old_size = path.stat().st_size if path.exists() else 0
        new_used = used - old_size + len(payload)
        if self.max_bytes_per_run is not None and new_used > self.max_bytes_per_run:
            raise ValueError(f"activation storage exceeds max_bytes_per_run for {run_label}: {new_used} > {self.max_bytes_per_run}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self._bytes_by_run[run_label] = new_used
        return ActivationCapture(
            run_label=run_label,
            run_id=run_id,
            step=step,
            module=module,
            module_kind=module_kind,
            activation_ref=relative.as_posix(),
            shape=list(array.shape),
            dtype=str(array.dtype),
            source_dtype=original_dtype,
        )

    def read_activation(self, activation_ref: str) -> np.ndarray:
        path = self.mech_dir / activation_ref
        if path.suffix == ".safetensors":
            try:
                from safetensors.numpy import load_file
            except ImportError as exc:  # pragma: no cover - exercised only without optional extra.
                raise RuntimeError("safetensors storage requires a rebuild with the safetensors dependency present") from exc
            loaded = load_file(str(path))
            return np.asarray(loaded["activation"])
        if path.suffix == ".npy":
            return np.asarray(np.load(path))
        raise ValueError(f"unsupported activation file: {path}")


class ActivationIndexWriter:
    def __init__(
        self,
        mech_dir: str | Path,
        *,
        incident_id: str,
        clean_run_id: str,
        perturbed_run_id: str,
        model: dict[str, Any],
        external_first_divergence_step: int | None = None,
        negative_control: bool = False,
        synthetic: bool = False,
        synthetic_lead_time_steps: int | None = None,
        required_modules: list[str] | None = None,
    ):
        self.mech_dir = Path(mech_dir)
        self.index = {
            "format": "robotprobe.mech.activation_index.v0",
            "incident_id": incident_id,
            "trace_pair": {
                "clean_run_id": clean_run_id,
                "perturbed_run_id": perturbed_run_id,
            },
            "model": model,
            "external_first_divergence_step": external_first_divergence_step,
            "negative_control": negative_control,
            # True when the activations were generated by a fixture rather than captured from a
            # model. Synthetic captures exercise the pipeline; they are not evidence about a model.
            "synthetic": synthetic,
            "synthetic_lead_time_steps": synthetic_lead_time_steps,
            "captures": [],
        }
        if required_modules is not None:
            self.index["required_modules"] = list(required_modules)

    def add(self, capture: ActivationCapture) -> None:
        self.index["captures"].append(capture.to_index_record())

    def write(self, path: str | Path | None = None) -> Path:
        out_path = Path(path) if path is not None else self.mech_dir / "activation_index.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out_path


def _safe_module_path(module: str) -> str:
    return quote(module, safe=".") if module not in ("", ".", "..") else quote(module, safe="").replace(".", "%2E") or "%00"
