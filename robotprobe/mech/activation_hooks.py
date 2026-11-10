from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CapturePolicy:
    every_n_steps: int = 1
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if type(self.every_n_steps) is not int or self.every_n_steps <= 0:
            raise ValueError("every_n_steps must be a positive integer")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 0):
            raise ValueError("max_steps must be a nonnegative integer")

    def should_capture(self, step: int) -> bool:
        if self.max_steps is not None and step > self.max_steps:
            return False
        return step % self.every_n_steps == 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CapturePolicy":
        steps = config.get("steps", {})
        if not isinstance(steps, dict) or steps.get("mode", "every_n") != "every_n":
            raise ValueError("steps must specify mode: every_n")
        return cls(every_n_steps=steps.get("n", 1), max_steps=steps.get("max_steps"))


@dataclass(frozen=True)
class ModuleSelector:
    module_names: list[str]

    def select(self, model: Any) -> dict[str, Any]:
        if not hasattr(model, "named_modules"):
            raise ValueError("model must expose named_modules()")
        wanted = set(self.module_names)
        selected = {name: module for name, module in model.named_modules() if name in wanted}
        if missing := wanted - selected.keys():
            raise ValueError(f"unknown activation modules: {sorted(missing)}")
        return selected


class ForwardHookActivationCollector:
    """Capture snapshots with one step per model forward, or caller-supplied episode steps."""

    def __init__(self, *, module_selector: ModuleSelector, capture_policy: CapturePolicy):
        self.module_selector = module_selector
        self.capture_policy = capture_policy
        self.captures: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self._current_step: int | None = None
        self._next_step = 0
        self._explicit_steps = False

    def attach(self, model: Any) -> "ForwardHookActivationCollector":
        if self._handles:
            raise ValueError("collector is already attached")
        selected = self.module_selector.select(model)
        for name, module in selected.items():
            if not hasattr(module, "register_forward_hook"):
                raise ValueError(f"module {name} does not expose register_forward_hook")
        if hasattr(model, "register_forward_pre_hook"):
            self._handles.append(model.register_forward_pre_hook(self._begin_forward))
        for name, module in selected.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def set_step(self, step: int) -> None:
        """Select explicit episode-step mode; call before each forward in that mode."""
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        self._explicit_steps = True
        self._current_step = step

    def _begin_forward(self, _module: Any, _inputs: Any) -> None:
        if not self._explicit_steps:
            self._current_step = self._next_step
            self._next_step += 1

    def __enter__(self) -> "ForwardHookActivationCollector":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            if hasattr(handle, "remove"):
                handle.remove()
        self._handles.clear()

    def _make_hook(self, module_name: str):
        def hook(_module, _inputs, output):
            if self._current_step is None:
                raise ValueError("runtime has no model pre-hook; call collector.set_step(step) before each forward")
            if not self.capture_policy.should_capture(self._current_step):
                return
            self.captures.append(
                {
                    "step": self._current_step,
                    "module": module_name,
                    "source_dtype": _source_dtype(output),
                    "activation": _to_numpy(output),
                }
            )

        return hook


def _to_numpy(value: Any, *, dtype: str | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    # NumPy cannot represent torch.bfloat16. Preserve source_dtype at the capture site.
    if str(getattr(value, "dtype", "")) == "torch.bfloat16":
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.array(value, dtype=dtype, copy=True)


def _source_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return str(dtype if dtype is not None else np.asarray(value).dtype)
