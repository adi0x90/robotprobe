"""Capture selected tensor outputs from an NNsight-supported PyTorch model.

The adapter clones outputs inside the trace, before downstream in-place operations,
and returns detached NumPy snapshots with step, module, and source_dtype metadata.
Use collect.persist_capture_pair with explicit incident metadata to write the files
consumed by the activation-diff and report CLI commands.

Scope: verified on a small CPU model. Real VLA capture needs a working model runtime
and remains pack-specific work. This adapter reads activations only; live activation
patching is not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robotprobe.mech.activation_hooks import CapturePolicy, _source_dtype, _to_numpy
from robotprobe.mech.adapters.base import WhiteBoxActivationAdapter


@dataclass
class NNsightActivationAdapter(WhiteBoxActivationAdapter):
    target: str = "nnsight"

    def capture(
        self, model: Any, inputs: Any, module_names: list[str], *, step_offset: int = 0,
        capture_policy: CapturePolicy | None = None, dtype: str | None = None,
        max_bytes_per_run: int | None = None,
    ) -> list[dict[str, Any]]:
        nnsight = _import_nnsight()
        wrapped = model if _is_nnsight(model, nnsight) else nnsight.NNsight(model)
        captures: list[dict[str, Any]] = []
        policy = capture_policy or CapturePolicy()
        if dtype not in (None, "float16", "float32", "float64"):
            raise ValueError(f"unsupported capture dtype: {dtype}")
        if max_bytes_per_run is not None and (type(max_bytes_per_run) is not int or max_bytes_per_run <= 0):
            raise ValueError("max_bytes_per_run must be a positive integer")
        captured_bytes = 0
        for index, single_input in enumerate(inputs):
            step = step_offset + index
            should_capture = policy.should_capture(step)
            saved: dict[str, Any] = {}
            with wrapped.trace(single_input):
                if should_capture:
                    for name in module_names:
                        saved[name] = _resolve_module(wrapped, name).output.detach().clone().save()
            if not should_capture:
                continue
            for name in module_names:
                activation = _to_numpy(saved[name], dtype=dtype)
                captured_bytes += activation.nbytes
                if max_bytes_per_run is not None and captured_bytes > max_bytes_per_run:
                    raise ValueError("activation capture exceeds max_bytes_per_run")
                captures.append(
                    {
                        "step": step,
                        "module": name,
                        "activation": activation,
                        "source_dtype": _source_dtype(saved[name]),
                    }
                )
        return captures

    def capture_pair(
        self,
        model: Any,
        clean_episode: Any,
        perturbed_episode: Any,
        capture_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        module_names = _module_names(capture_config)
        config = capture_config if isinstance(capture_config, dict) else {}
        storage = config.get("storage", {})
        if not isinstance(storage, dict):
            raise ValueError("storage must be a mapping")
        options = {"capture_policy": CapturePolicy.from_config(config), "dtype": storage.get("dtype"),
                   "max_bytes_per_run": storage.get("max_bytes_per_run")}
        return {
            "format": "robotprobe.mech.nnsight.v0",
            "target": self.target,
            "modules": module_names,
            "clean": self.capture(model, clean_episode, module_names, **options),
            "perturbed": self.capture(model, perturbed_episode, module_names, **options),
        }


def _import_nnsight():
    try:
        import nnsight  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "NNsightActivationAdapter requires the optional 'nnsight' extra: "
            "pip install 'robotprobe[nnsight]' (installs nnsight + torch)."
        ) from exc
    return nnsight


def _is_nnsight(model: Any, nnsight: Any) -> bool:
    base = getattr(nnsight, "NNsight", None)
    return base is not None and isinstance(model, base)


def _resolve_module(wrapped: Any, name: str) -> Any:
    proxy = wrapped
    if name == "":
        return proxy
    for part in name.split("."):
        proxy = proxy[int(part)] if part.isdigit() else getattr(proxy, part)
    return proxy


def _module_names(capture_config: dict[str, Any] | None) -> list[str]:
    if capture_config is None:
        return []
    if isinstance(capture_config, list):
        return [str(item) for item in capture_config]
    return [str(layer["name"]) for layer in capture_config.get("layers", [])]
