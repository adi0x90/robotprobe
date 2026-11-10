from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from robotprobe.runtime.types import (
    AuthorizedAction,
    BackendHealth,
    CanonicalObservation,
    ExecutionReceipt,
    InferenceContext,
    ProposedActionChunk,
    RawObservation,
    ScenarioSpec,
)


@runtime_checkable
class PolicyAdapter(Protocol):
    capabilities: Any

    def load(self) -> None: ...

    def reset(self, scenario: ScenarioSpec) -> None: ...

    def predict(
        self,
        observation: CanonicalObservation,
        context: InferenceContext,
    ) -> ProposedActionChunk: ...

    def close(self) -> None: ...


@runtime_checkable
class EmbodimentAdapter(Protocol):
    capabilities: Any

    def canonicalize_observation(self, raw: RawObservation) -> CanonicalObservation: ...

    def decode_action_chunk(self, proposed: ProposedActionChunk) -> ProposedActionChunk: ...


@runtime_checkable
class RuntimeBackend(Protocol):
    capabilities: Any

    def connect(self) -> None: ...

    def reset(self, scenario: ScenarioSpec) -> RawObservation: ...

    def observe(self) -> RawObservation: ...

    def execute(self, action: AuthorizedAction) -> ExecutionReceipt: ...

    def health(self) -> BackendHealth: ...

    def close(self) -> None: ...


@runtime_checkable
class ScenarioProvider(Protocol):
    capabilities: Any

    def build(self) -> ScenarioSpec: ...


@runtime_checkable
class ActivationCapture(Protocol):
    capabilities: Any

    def session(
        self,
        model: Any,
        context: InferenceContext,
    ) -> AbstractContextManager[Any]: ...


@runtime_checkable
class ConfigurableCapturePolicyAdapter(Protocol):
    def configure_capture(
        self,
        *,
        profile_path: Path,
        recorder: Any,
        storage: str | None,
        output_dir: Path | None,
        max_bytes_per_run: int | None,
    ) -> None: ...
