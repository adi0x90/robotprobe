from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from robotprobe.runtime.capabilities import (
    RuntimeComponentCapabilities,
    check_runtime_compatibility,
)
from robotprobe.runtime.protocols import (
    EmbodimentAdapter,
    PolicyAdapter,
    RuntimeBackend,
    ScenarioProvider,
)
from robotprobe.runtime.safety import SafetyGateway
from robotprobe.runtime.scheduler import ActionScheduler
from robotprobe.runtime.types import (
    CanonicalObservation,
    EpisodeResult,
    ExecutionReceipt,
    InferenceContext,
    ProposedActionChunk,
    RawObservation,
    RuntimePhaseError,
    ScenarioSpec,
)


@dataclass(frozen=True)
class _EpisodeOutcome:
    success: bool
    steps: int
    reason: str


class EpisodeRunner:
    """Run one traced episode over already-instantiated runtime components."""

    def __init__(
        self,
        *,
        policy: PolicyAdapter,
        embodiment: EmbodimentAdapter,
        backend: RuntimeBackend,
        scenario_provider: ScenarioProvider,
        scheduler: ActionScheduler,
        safety_gateway: SafetyGateway,
        recorder: Any,
        clock_ns: Callable[[], int] = time.time_ns,
    ):
        self.policy = policy
        self.embodiment = embodiment
        self.backend = backend
        self.scenario_provider = scenario_provider
        self.scheduler = scheduler
        self.safety_gateway = safety_gateway
        self.recorder = recorder
        self.clock_ns = clock_ns
        self._policy_call_index = 0

    def run(self) -> EpisodeResult:
        self._policy_call_index = 0
        cleanup_stack: list[tuple[str, Any, Callable[[], None]]] = []
        primary_error: RuntimePhaseError | None = None
        cleanup_error: RuntimePhaseError | None = None
        outcome: _EpisodeOutcome | None = None
        scenario: ScenarioSpec | None = None

        try:
            self._validate_compatibility()
            scenario = self._invoke(
                "scenario_build",
                self.scenario_provider,
                self.scenario_provider.build,
            )
            self._require_type(scenario, ScenarioSpec, "scenario_build", self.scenario_provider)

            cleanup_stack.append(("backend", self.backend, self.backend.close))
            self._invoke("backend_connect", self.backend, self.backend.connect)

            cleanup_stack.append(("policy", self.policy, self.policy.close))
            self._invoke("policy_load", self.policy, self.policy.load)

            self._invoke("scheduler_reset", "action_scheduler", self.scheduler.reset)
            self._invoke("policy_reset", self.policy, lambda: self.policy.reset(scenario))
            raw = self._invoke("backend_reset", self.backend, lambda: self.backend.reset(scenario))
            self._require_type(raw, RawObservation, "backend_reset", self.backend)
            outcome = self._run_steps(scenario, raw)
        except Exception as error:
            primary_error = self._bounded_error(error, "runtime", "episode_runner")
            self._record_runtime_error(primary_error)
        finally:
            for _role, component, close in reversed(cleanup_stack):
                try:
                    close()
                except Exception as error:
                    bounded = self._bounded_error(error, "cleanup", _component_id(component))
                    self._record_runtime_error(bounded, secondary=primary_error is not None or cleanup_error is not None)
                    if cleanup_error is None:
                        cleanup_error = bounded

        if primary_error is not None:
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error
        if outcome is None or scenario is None:  # pragma: no cover - defensive invariant.
            raise RuntimeError("episode runner completed without an outcome")

        self.recorder.event(
            outcome.steps,
            "outcome",
            "outcome",
            success=outcome.success,
            steps=outcome.steps,
            reason=outcome.reason,
            scenario_id=scenario.scenario_id,
            execution_mode=self.safety_gateway.execution_mode.value,
        )
        return EpisodeResult(
            run_id=str(self.recorder.context["run_id"]),
            success=outcome.success,
            steps=outcome.steps,
            reason=outcome.reason,
            trace_path=str(self.recorder.path),
            metadata={
                "scenario_id": scenario.scenario_id,
                "execution_mode": self.safety_gateway.execution_mode.value,
            },
        )

    def _run_steps(
        self,
        scenario: ScenarioSpec,
        initial_raw: RawObservation,
    ) -> _EpisodeOutcome:
        raw = initial_raw
        self.recorder.event(
            0,
            "instruction",
            "instruction",
            value=scenario.instruction,
            scenario_id=scenario.scenario_id,
            seed=scenario.seed,
        )

        for env_step in range(scenario.max_steps):
            observation = self._invoke(
                "observation_decode",
                self.embodiment,
                lambda: self.embodiment.canonicalize_observation(raw),
            )
            self._require_type(
                observation,
                CanonicalObservation,
                "observation_decode",
                self.embodiment,
            )
            self.recorder.event(
                env_step,
                "observation",
                "observation",
                observation_ref=observation.observation_id,
                **observation.to_trace_fields(),
            )

            if not self.scheduler.has_pending:
                self._infer_and_schedule(observation, env_step)

            scheduled = self._invoke(
                "action_schedule",
                "action_scheduler",
                self.scheduler.next_action,
            )
            self.recorder.event(
                env_step,
                "scheduled_action",
                "action_schedule",
                **scheduled.to_trace_fields(),
            )

            authorized = self._invoke(
                "safety_authorization",
                "safety_gateway",
                lambda: self.safety_gateway.authorize(
                    scheduled,
                    observation,
                    now_ns=self.clock_ns(),
                ),
            )
            self.recorder.event(
                env_step,
                "safety_decision",
                "safety",
                command_id=scheduled.command_id,
                chunk_id=scheduled.chunk_id,
                policy_call_id=scheduled.policy_call_id,
                source_observation_id=scheduled.source_observation_id,
                **authorized.decision.to_dict(),
            )
            if not authorized.decision.accepted:
                return _EpisodeOutcome(
                    success=False,
                    steps=env_step,
                    reason=f"safety_rejected:{authorized.decision.reason}",
                )

            receipt = self._invoke(
                "backend_execute",
                self.backend,
                lambda: self.backend.execute(authorized),
            )
            self._validate_receipt(receipt, scheduled.command_id)
            self._invoke(
                "execution_receipt",
                self.backend,
                lambda: self.safety_gateway.validate_receipt(receipt),
            )
            receipt_fields = _receipt_trace_fields(receipt)
            self.recorder.event(
                env_step,
                "execution_receipt",
                "backend_execution",
                **receipt_fields,
            )
            self.recorder.event(
                env_step,
                "robot_state",
                "robot_state",
                command_id=receipt.command_id,
                measured_state=receipt_fields["measured_state"],
                timestamp_ns=receipt.timestamp_ns,
            )

            completed_steps = env_step + 1
            if not receipt.accepted:
                reason = str(
                    getattr(receipt, "metadata", {}).get(
                        "reason",
                        "not_accepted",
                    )
                )
                return _EpisodeOutcome(
                    success=False,
                    steps=completed_steps,
                    reason=f"backend_rejected:{reason}",
                )
            if bool(receipt.done):
                return _EpisodeOutcome(
                    success=bool(receipt.success),
                    steps=completed_steps,
                    reason="scenario_success" if receipt.success else "backend_terminated",
                )
            if completed_steps < scenario.max_steps:
                raw = self._invoke("backend_observe", self.backend, self.backend.observe)
                self._require_type(raw, RawObservation, "backend_observe", self.backend)

        return _EpisodeOutcome(
            success=False,
            steps=scenario.max_steps,
            reason="max_steps_exhausted",
        )

    def _infer_and_schedule(
        self,
        observation: CanonicalObservation,
        env_step: int,
    ) -> None:
        policy_call_id = f"policy:{self._policy_call_index:06d}"
        action_chunk_id = f"chunk:{self._policy_call_index:06d}"
        self._policy_call_index += 1
        context = InferenceContext(
            env_step=env_step,
            policy_call_id=policy_call_id,
            action_chunk_id=action_chunk_id,
            observation_id=observation.observation_id,
        )
        self.recorder.event(
            env_step,
            "policy_inference",
            "policy_forward",
            status="started",
            **context.to_dict(),
        )
        proposed = self._invoke(
            "policy_inference",
            self.policy,
            lambda: self.policy.predict(observation, context),
        )
        self._validate_chunk_lineage(proposed, observation, context, phase="policy_output")
        self.recorder.event(
            env_step,
            "policy_inference",
            "policy_forward",
            status="completed",
            **context.to_dict(),
        )

        decoded = self._invoke(
            "action_decode",
            self.embodiment,
            lambda: self.embodiment.decode_action_chunk(proposed),
        )
        self._validate_chunk_lineage(decoded, observation, context, phase="action_decode")
        scheduled_count = self._invoke(
            "action_schedule",
            "action_scheduler",
            lambda: self.scheduler.load(decoded),
        )
        self.recorder.event(
            env_step,
            "action_chunk",
            "policy_output",
            **decoded.to_trace_fields(),
            scheduled_count=scheduled_count,
        )

    def _validate_compatibility(self) -> None:
        try:
            report = check_runtime_compatibility(
                policy=_runtime_capabilities(self.policy),
                embodiment=_runtime_capabilities(self.embodiment),
                backend=_runtime_capabilities(self.backend),
                scenario=_runtime_capabilities(self.scenario_provider),
                execution_mode=self.safety_gateway.execution_mode,
            )
        except Exception as error:
            raise RuntimePhaseError("compatibility", "runtime_components", error) from error
        if not report.valid:
            raise RuntimePhaseError(
                "compatibility",
                "runtime_components",
                ValueError("; ".join(report.issues)),
            )

    def _validate_chunk_lineage(
        self,
        chunk: Any,
        observation: CanonicalObservation,
        context: InferenceContext,
        *,
        phase: str,
    ) -> None:
        if not isinstance(chunk, ProposedActionChunk):
            raise RuntimePhaseError(
                phase,
                _component_id(self.policy if phase == "policy_output" else self.embodiment),
                TypeError("component did not return ProposedActionChunk"),
            )
        expected = {
            "chunk_id": context.action_chunk_id,
            "policy_call_id": context.policy_call_id,
            "source_observation_id": observation.observation_id,
            "source_observation_captured_at_ns": observation.captured_at_ns,
        }
        mismatches = [
            field_name
            for field_name, expected_value in expected.items()
            if getattr(chunk, field_name) != expected_value
        ]
        if mismatches:
            component = self.policy if phase == "policy_output" else self.embodiment
            raise RuntimePhaseError(
                phase,
                _component_id(component),
                ValueError(f"action chunk lineage mismatch: {mismatches}"),
            )

    def _validate_receipt(self, receipt: Any, expected_command_id: str) -> None:
        if not isinstance(receipt, ExecutionReceipt):
            raise RuntimePhaseError(
                "execution_receipt",
                _component_id(self.backend),
                TypeError("backend must return ExecutionReceipt"),
            )
        if receipt.command_id != expected_command_id:
            raise RuntimePhaseError(
                "execution_receipt",
                _component_id(self.backend),
                ValueError("execution receipt command lineage mismatch"),
            )
        expected_backend_id = _component_id(self.backend)
        if receipt.backend_id != expected_backend_id:
            raise RuntimePhaseError(
                "execution_receipt",
                expected_backend_id,
                ValueError("execution receipt backend identity mismatch"),
            )
        receipt_mode = getattr(receipt.execution_mode, "value", receipt.execution_mode)
        if receipt_mode != self.safety_gateway.execution_mode.value:
            raise RuntimePhaseError(
                "execution_receipt",
                _component_id(self.backend),
                ValueError("execution receipt mode mismatch"),
            )
        if not isinstance(receipt.accepted, bool) or not isinstance(
            receipt.command_applied,
            bool,
        ):
            raise RuntimePhaseError(
                "execution_receipt",
                _component_id(self.backend),
                TypeError("execution receipt status fields must be boolean"),
            )
        if receipt.command_applied and not receipt.accepted:
            raise RuntimePhaseError(
                "execution_receipt",
                _component_id(self.backend),
                ValueError("rejected execution receipt cannot report an applied command"),
            )

    def _require_type(
        self,
        value: Any,
        expected_type: type[Any],
        phase: str,
        component: Any,
    ) -> None:
        if not isinstance(value, expected_type):
            raise RuntimePhaseError(
                phase,
                _component_id(component),
                TypeError(f"expected {expected_type.__name__}"),
            )

    def _invoke(
        self,
        phase: str,
        component: Any,
        operation: Callable[[], Any],
    ) -> Any:
        try:
            return operation()
        except RuntimePhaseError:
            raise
        except Exception as error:
            raise RuntimePhaseError(phase, _component_id(component), error) from error

    def _record_runtime_error(
        self,
        error: RuntimePhaseError,
        *,
        secondary: bool = False,
    ) -> None:
        self.recorder.event(
            0,
            "runtime_error",
            "runtime",
            phase=error.phase,
            component_id=error.component_id,
            error_type=error.original_type,
            message="component operation failed",
            secondary=secondary,
        )

    @staticmethod
    def _bounded_error(
        error: Exception,
        default_phase: str,
        default_component_id: str,
    ) -> RuntimePhaseError:
        if isinstance(error, RuntimePhaseError):
            return error
        return RuntimePhaseError(default_phase, default_component_id, error)


def _runtime_capabilities(component: Any) -> RuntimeComponentCapabilities:
    value = component.capabilities
    if isinstance(value, RuntimeComponentCapabilities):
        return value
    if isinstance(value, dict):
        return RuntimeComponentCapabilities.from_dict(value)
    raise TypeError("runtime component capabilities must be a capability object or mapping")


def _component_id(component: Any) -> str:
    if isinstance(component, str):
        return component
    capabilities = getattr(component, "capabilities", None)
    if isinstance(capabilities, RuntimeComponentCapabilities):
        return capabilities.component_id
    if isinstance(capabilities, dict) and capabilities.get("component_id"):
        return str(capabilities["component_id"])
    return type(component).__name__


def _receipt_trace_fields(receipt: ExecutionReceipt) -> dict[str, Any]:
    return receipt.to_trace_fields()
