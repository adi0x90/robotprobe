from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.types import (
    AuthorizedAction,
    CanonicalObservation,
    ExecutionMode,
    RuntimePhaseError,
    SafetyDecision,
    ScheduledAction,
)


@dataclass(frozen=True)
class SafetyEnvelope:
    command_space: str
    max_abs: tuple[float, ...]
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    max_observation_age_ms: float = 250.0

    def __post_init__(self) -> None:
        if not self.command_space:
            raise ValueError("command_space must be non-empty")
        if not self.max_abs or not np.isfinite(self.max_abs).all() or np.any(np.asarray(self.max_abs) <= 0):
            raise ValueError("max_abs must contain positive finite limits")
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("workspace_min and workspace_max must be configured together")
        if self.workspace_min is not None:
            lower = np.asarray(self.workspace_min, dtype=float)
            upper = np.asarray(self.workspace_max, dtype=float)
            if lower.shape != (3,) or upper.shape != (3,) or not np.all(lower < upper):
                raise ValueError("workspace bounds must be ordered three-dimensional vectors")
        if (
            self.max_observation_age_ms <= 0
            or not np.isfinite(self.max_observation_age_ms)
        ):
            raise ValueError("max_observation_age_ms must be positive and finite")


class SafetyGateway:
    def __init__(
        self,
        *,
        envelope: SafetyEnvelope,
        execution_mode: ExecutionMode,
        backend_capabilities: RuntimeComponentCapabilities,
    ):
        self.envelope = envelope
        self.execution_mode = ExecutionMode(execution_mode)
        self.backend_capabilities = backend_capabilities

    def authorize(
        self,
        scheduled: ScheduledAction,
        observation: CanonicalObservation,
        *,
        now_ns: int,
    ) -> AuthorizedAction:
        if isinstance(now_ns, bool):
            raise ValueError("now_ns must be a non-negative integer")
        try:
            now_ns = operator.index(now_ns)
        except TypeError as error:
            raise ValueError("now_ns must be a non-negative integer") from error
        if now_ns < 0:
            raise ValueError("now_ns must be a non-negative integer")

        mode_reason = self._mode_rejection()
        if mode_reason is not None:
            return self._result(scheduled, False, "rejected", mode_reason, now_ns)
        if self.execution_mode is ExecutionMode.HARDWARE_EXECUTABLE:
            return self._result(
                scheduled,
                False,
                "rejected",
                "hardware_executable_not_supported",
                now_ns,
            )
        if scheduled.command_space != self.envelope.command_space:
            return self._result(scheduled, False, "rejected", "command_space_mismatch", now_ns)

        value = np.asarray(scheduled.value, dtype=float)
        limits = np.asarray(self.envelope.max_abs, dtype=float)
        if value.shape != limits.shape:
            return self._result(scheduled, False, "rejected", "action_dimension_mismatch", now_ns)
        if not np.isfinite(value).all():
            return self._result(scheduled, False, "rejected", "non_finite_action", now_ns)
        exceeded = np.flatnonzero(np.abs(value) > limits)
        if exceeded.size:
            return self._result(
                scheduled,
                False,
                "rejected",
                "action_limit_exceeded",
                now_ns,
                details={"dimensions": exceeded.tolist()},
            )

        age_ns = now_ns - scheduled.source_observation_captured_at_ns
        if age_ns < 0:
            return self._result(scheduled, False, "rejected", "observation_clock_skew", now_ns)
        if age_ns > self.envelope.max_observation_age_ms * 1_000_000:
            return self._result(
                scheduled,
                False,
                "rejected",
                "observation_stale",
                now_ns,
                details={"age_ns": age_ns},
            )
        if self._outside_workspace(value, observation):
            return self._result(scheduled, False, "rejected", "workspace_violation", now_ns)

        if self.execution_mode is ExecutionMode.HARDWARE_SHADOW:
            return self._result(scheduled, True, "shadow_only", "within_envelope", now_ns)
        return self._result(scheduled, True, "authorized", "within_envelope", now_ns)

    def validate_receipt(self, receipt: Any) -> None:
        if self.execution_mode is ExecutionMode.HARDWARE_SHADOW and bool(
            getattr(receipt, "command_applied", False)
        ):
            backend_id = str(getattr(receipt, "backend_id", "unknown_backend"))
            raise RuntimePhaseError(
                "execution_receipt",
                backend_id,
                ValueError("hardware shadow backend reported an applied command"),
            )

    def _mode_rejection(self) -> str | None:
        backend = self.backend_capabilities
        if self.execution_mode is ExecutionMode.SIMULATION and backend.is_hardware:
            return "execution_mode_backend_mismatch"
        if self.execution_mode is ExecutionMode.HARDWARE_SHADOW and not (
            backend.is_hardware and backend.supports_shadow
        ):
            return "execution_mode_backend_mismatch"
        if self.execution_mode is ExecutionMode.HARDWARE_EXECUTABLE and not backend.is_hardware:
            return "execution_mode_backend_mismatch"
        return None

    def _outside_workspace(
        self,
        value: np.ndarray,
        observation: CanonicalObservation,
    ) -> bool:
        if self.envelope.workspace_min is None:
            return False
        predicted = observation.eef_position + value[:3]
        lower = np.asarray(self.envelope.workspace_min, dtype=float)
        upper = np.asarray(self.envelope.workspace_max, dtype=float)
        return bool(np.any(predicted < lower) or np.any(predicted > upper))

    def _result(
        self,
        scheduled: ScheduledAction,
        accepted: bool,
        disposition: str,
        reason: str,
        checked_at_ns: int,
        *,
        details: dict[str, Any] | None = None,
    ) -> AuthorizedAction:
        return AuthorizedAction(
            scheduled=scheduled,
            decision=SafetyDecision(
                accepted=accepted,
                disposition=disposition,
                reason=reason,
                checked_at_ns=checked_at_ns,
                details=details or {},
            ),
        )
