from __future__ import annotations

import fnmatch
import hashlib
import io
import re
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.numpy import save as save_safetensors

from robotprobe.runtime.types import InferenceContext


class CaptureStorageMode(str, Enum):
    SUMMARY = "summary"
    NUMPY = "npy"
    SAFETENSORS = "safetensors"


@dataclass(frozen=True)
class ModuleProbeSpec:
    probe_id: str
    selector: str
    module_kind: str
    phase: str
    storage: CaptureStorageMode = CaptureStorageMode.SUMMARY
    every_n_policy_calls: int = 1
    max_module_calls: int | None = None
    denoise_step_from_module_call: bool = False

    def __post_init__(self) -> None:
        for field_name in ("probe_id", "selector", "module_kind", "phase"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        object.__setattr__(self, "storage", CaptureStorageMode(self.storage))
        if self.every_n_policy_calls <= 0:
            raise ValueError("every_n_policy_calls must be positive")
        if self.max_module_calls is not None and self.max_module_calls <= 0:
            raise ValueError("max_module_calls must be positive when configured")

    def matches(self, module_name: str) -> bool:
        return fnmatch.fnmatchcase(module_name, self.selector)


@dataclass(frozen=True)
class CaptureProfile:
    probes: tuple[ModuleProbeSpec, ...]
    output_dir: Path | None = None
    max_bytes_per_run: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        probes = tuple(self.probes)
        if not probes:
            raise ValueError("capture profile requires at least one probe")
        probe_ids = [probe.probe_id for probe in probes]
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError("capture profile probe_id values must be unique")
        object.__setattr__(self, "probes", probes)
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.max_bytes_per_run <= 0:
            raise ValueError("max_bytes_per_run must be positive")
        requires_output = any(probe.storage is not CaptureStorageMode.SUMMARY for probe in probes)
        if requires_output and self.output_dir is None:
            raise ValueError("output_dir is required for full activation storage")


@dataclass(frozen=True)
class TensorSummary:
    shape: tuple[int, ...]
    dtype: str
    finite_count: int
    non_finite_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    standard_deviation: float | None
    l2_norm: float | None
    sha256: str

    @classmethod
    def from_value(cls, value: Any) -> "TensorSummary":
        array = _as_numpy_array(value)
        finite_mask = np.isfinite(array)
        finite_values = np.asarray(array[finite_mask], dtype=np.float64)
        if finite_values.size:
            minimum = float(np.min(finite_values))
            maximum = float(np.max(finite_values))
            mean = float(np.mean(finite_values))
            standard_deviation = float(np.std(finite_values))
            l2_norm = float(np.linalg.norm(finite_values))
        else:
            minimum = maximum = mean = standard_deviation = l2_norm = None
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.sha256(contiguous.tobytes()).hexdigest()
        return cls(
            shape=tuple(int(value) for value in array.shape),
            dtype=str(array.dtype),
            finite_count=int(np.count_nonzero(finite_mask)),
            non_finite_count=int(array.size - np.count_nonzero(finite_mask)),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            standard_deviation=standard_deviation,
            l2_norm=l2_norm,
            sha256=f"sha256:{digest}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "finite_count": self.finite_count,
            "non_finite_count": self.non_finite_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "l2_norm": self.l2_norm,
            "sha256": self.sha256,
        }


@dataclass
class CaptureBudget:
    max_bytes: int
    used_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("capture budget must be positive")
        if self.used_bytes < 0 or self.used_bytes > self.max_bytes:
            raise ValueError("capture budget used_bytes must be within its limit")

    @property
    def remaining_bytes(self) -> int:
        return self.max_bytes - self.used_bytes

    def reserve(self, byte_count: int) -> bool:
        if byte_count < 0:
            raise ValueError("capture byte count must be non-negative")
        with self._lock:
            if self.used_bytes + byte_count > self.max_bytes:
                return False
            self.used_bytes += byte_count
            return True

    def release(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("capture byte count must be non-negative")
        with self._lock:
            if byte_count > self.used_bytes:
                raise ValueError("cannot release more capture bytes than were reserved")
            self.used_bytes -= byte_count

    def reset(self) -> None:
        with self._lock:
            self.used_bytes = 0


def extract_activation_array(output: Any) -> np.ndarray | None:
    """Return the first tensor-like activation in a nested module output."""
    if isinstance(output, np.ndarray):
        return np.array(output, copy=True)
    if _is_tensor_like(output):
        try:
            return _as_numpy_array(output)
        except (TypeError, ValueError, RuntimeError):
            return None
    if isinstance(output, Mapping):
        for value in output.values():
            array = extract_activation_array(value)
            if array is not None:
                return array
    elif isinstance(output, (tuple, list)):
        for value in output:
            array = extract_activation_array(value)
            if array is not None:
                return array
    return None


class ContextualActivationCollector:
    """Attach bounded forward hooks for one policy-inference context at a time."""

    capabilities = {
        "format": "robotprobe.runtime.capture.v0",
        "component_id": "robotprobe.capture.contextual_activation",
        "capture_mode": "forward_hook",
    }

    def __init__(
        self,
        profile: CaptureProfile,
        *,
        recorder: Any,
        model_ref: Mapping[str, Any],
    ):
        self.profile = profile
        self.recorder = recorder
        self.model_ref = dict(model_ref)
        self.budget = CaptureBudget(profile.max_bytes_per_run)
        self._policy_call_count = 0
        self._session_count = 0
        self._session_lock = threading.Lock()
        self._session_active = False

    def reset_budget(self) -> None:
        self.budget.reset()

    def session(
        self,
        model: Any,
        context: InferenceContext,
    ) -> AbstractContextManager["ContextualActivationCollector"]:
        return _CollectorSession(self, model, context)

    def _enter_session(self, model: Any, context: InferenceContext) -> list[Any]:
        with self._session_lock:
            if self._session_active:
                raise RuntimeError("activation capture sessions cannot overlap")
            self._session_active = True
            self._policy_call_count += 1
            self._session_count += 1
            policy_call_ordinal = self._policy_call_count
            session_ordinal = self._session_count

        handles: list[Any] = []
        try:
            named_modules = list(model.named_modules())
            for probe in self.profile.probes:
                if (policy_call_ordinal - 1) % probe.every_n_policy_calls != 0:
                    continue
                for module_name, module in named_modules:
                    if not probe.matches(module_name):
                        continue
                    counter = [0]

                    def hook(
                        _module: Any,
                        _inputs: Any,
                        output: Any,
                        *,
                        selected_probe: ModuleProbeSpec = probe,
                        selected_module_name: str = module_name,
                        call_counter: list[int] = counter,
                    ) -> None:
                        module_call_index = call_counter[0]
                        call_counter[0] += 1
                        if (
                            selected_probe.max_module_calls is not None
                            and module_call_index >= selected_probe.max_module_calls
                        ):
                            return
                        self._capture(
                            output=output,
                            context=context,
                            probe=selected_probe,
                            module_name=selected_module_name,
                            module_call_index=module_call_index,
                            session_ordinal=session_ordinal,
                        )

                    handles.append(module.register_forward_hook(hook))
            return handles
        except BaseException:
            for handle in reversed(handles):
                handle.remove()
            with self._session_lock:
                self._session_active = False
            raise

    def _exit_session(self, handles: Sequence[Any]) -> None:
        try:
            for handle in reversed(handles):
                handle.remove()
        finally:
            with self._session_lock:
                self._session_active = False

    def _capture(
        self,
        *,
        output: Any,
        context: InferenceContext,
        probe: ModuleProbeSpec,
        module_name: str,
        module_call_index: int,
        session_ordinal: int,
    ) -> None:
        array = extract_activation_array(output)
        if array is None:
            self._record_drop(
                context=context,
                probe=probe,
                module_name=module_name,
                module_call_index=module_call_index,
                reason="unsupported_activation_output",
            )
            return

        summary = TensorSummary.from_value(array)
        activation_ref = f"summary:{summary.sha256}"
        stored_bytes = 0
        if probe.storage is not CaptureStorageMode.SUMMARY:
            payload, suffix = _serialize_activation(array, probe.storage)
            if not self.budget.reserve(len(payload)):
                self._record_drop(
                    context=context,
                    probe=probe,
                    module_name=module_name,
                    module_call_index=module_call_index,
                    reason="capture_budget_exhausted",
                    requested_bytes=len(payload),
                )
                return
            stored_bytes = len(payload)
            try:
                path = self._activation_path(
                    context=context,
                    probe=probe,
                    module_name=module_name,
                    module_call_index=module_call_index,
                    session_ordinal=session_ordinal,
                    suffix=suffix,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                activation_ref = str(path.resolve())
            except BaseException:
                self.budget.release(stored_bytes)
                raise

        capture = {
            "capture_mode": "forward_hook",
            "hook_version": "robotprobe.contextual_activation.v0",
            "detached": True,
            "requires_grad": False,
            "storage_mode": probe.storage.value,
            "tensor_summary": summary.to_dict(),
            "stored_bytes": stored_bytes,
        }
        fields: dict[str, Any] = {
            "probe_id": probe.probe_id,
            "module": module_name,
            "module_kind": probe.module_kind,
            "activation_ref": activation_ref,
            "activation_shape": list(summary.shape),
            "activation_dtype": summary.dtype,
            "model_ref": dict(self.model_ref),
            "capture": capture,
            "policy_call_id": context.policy_call_id,
            "module_call_index": module_call_index,
            "phase": probe.phase,
            "action_chunk_id": context.action_chunk_id,
            "observation_id": context.observation_id,
        }
        if probe.denoise_step_from_module_call:
            fields["denoise_step"] = module_call_index
        self.recorder.event(
            context.env_step,
            "model_activation",
            "model_activation",
            **fields,
        )

    def _record_drop(
        self,
        *,
        context: InferenceContext,
        probe: ModuleProbeSpec,
        module_name: str,
        module_call_index: int,
        reason: str,
        requested_bytes: int | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "reason": reason,
            "probe_id": probe.probe_id,
            "module": module_name,
            "module_kind": probe.module_kind,
            "policy_call_id": context.policy_call_id,
            "module_call_index": module_call_index,
            "phase": probe.phase,
            "action_chunk_id": context.action_chunk_id,
            "observation_id": context.observation_id,
            "storage_mode": probe.storage.value,
            "remaining_bytes": self.budget.remaining_bytes,
        }
        if requested_bytes is not None:
            fields["requested_bytes"] = requested_bytes
        if probe.denoise_step_from_module_call:
            fields["denoise_step"] = module_call_index
        self.recorder.event(
            context.env_step,
            "activation_capture_dropped",
            "model_activation",
            **fields,
        )

    def _activation_path(
        self,
        *,
        context: InferenceContext,
        probe: ModuleProbeSpec,
        module_name: str,
        module_call_index: int,
        session_ordinal: int,
        suffix: str,
    ) -> Path:
        assert self.profile.output_dir is not None
        policy_coordinate = _safe_coordinate(context.policy_call_id)
        module_coordinate = _safe_coordinate(module_name)
        probe_coordinate = _safe_coordinate(probe.probe_id)
        directory = (
            self.profile.output_dir
            / f"policy_{policy_coordinate}"
            / f"session_{session_ordinal:06d}"
            / module_coordinate
            / probe_coordinate
        )
        candidate = directory / f"call_{module_call_index:04d}{suffix}"
        collision = 0
        while candidate.exists():
            collision += 1
            candidate = directory / f"call_{module_call_index:04d}_collision_{collision:04d}{suffix}"
        return candidate


class _CollectorSession(AbstractContextManager[ContextualActivationCollector]):
    def __init__(
        self,
        collector: ContextualActivationCollector,
        model: Any,
        context: InferenceContext,
    ):
        self.collector = collector
        self.model = model
        self.context = context
        self.handles: list[Any] = []

    def __enter__(self) -> ContextualActivationCollector:
        self.handles = self.collector._enter_session(self.model, self.context)
        return self.collector

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.collector._exit_session(self.handles)


def _is_tensor_like(value: Any) -> bool:
    return all(hasattr(value, attribute) for attribute in ("detach", "cpu", "numpy"))


def _as_numpy_array(value: Any) -> np.ndarray:
    candidate = value
    if not isinstance(candidate, np.ndarray):
        if hasattr(candidate, "detach"):
            candidate = candidate.detach()
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        try:
            candidate = candidate.numpy()
        except TypeError:
            if not hasattr(candidate, "float"):
                raise
            candidate = candidate.float().numpy()
    array = np.array(candidate, copy=True)
    if array.dtype.kind not in "biufc":
        raise TypeError("activation must have a numeric or boolean dtype")
    return array


def _serialize_activation(
    array: np.ndarray,
    storage: CaptureStorageMode,
) -> tuple[bytes, str]:
    contiguous = np.ascontiguousarray(array)
    if storage is CaptureStorageMode.NUMPY:
        buffer = io.BytesIO()
        np.save(buffer, contiguous, allow_pickle=False)
        return buffer.getvalue(), ".npy"
    if storage is CaptureStorageMode.SAFETENSORS:
        return save_safetensors({"activation": contiguous}), ".safetensors"
    raise ValueError(f"full serialization is not available for storage mode {storage.value}")


def _safe_coordinate(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return rendered or "unnamed"
