from __future__ import annotations

import fnmatch
import hashlib
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from robotprobe.runtime.capture import (
    CaptureProfile,
    CaptureStorageMode,
    ContextualActivationCollector,
    ModuleProbeSpec,
)


_LAYER_FAMILIES = {
    "vision": re.compile(
        r"(?:^|\.)vision_tower(?:\.vision_model)?\.encoder\.layers\.(\d+)$"
    ),
    "language": re.compile(
        r"(?:^|\.)paligemma\.model\.language_model\.layers\.(\d+)$"
    ),
    "action_expert": re.compile(
        r"(?:^|\.)gemma_expert\.model\.layers\.(\d+)\.mlp$"
    ),
}
_PROJECTIONS = {
    "action_input_projection": ("action_in_proj", "action_projection"),
    "time_mlp_input": ("time_mlp_in", "time_conditioning"),
    "time_mlp_output": ("time_mlp_out", "time_conditioning"),
    "action_output_projection": ("action_out_proj", "action_projection"),
}
_CAPTURE_PROFILE_FORMAT = "robotprobe.runtime.capture_profile.v0"
_SEMANTIC_ROLES = {
    *(f"{family}.{position}" for family in _LAYER_FAMILIES for position in ("early", "middle", "late")),
    *_PROJECTIONS,
}


class Pi05CaptureConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Pi05CaptureSettings:
    profile_id: str
    profile_sha256: str
    semantic_roles: tuple[str, ...]
    user_selectors: tuple[str, ...]
    storage: CaptureStorageMode
    output_dir: Path | None
    max_bytes_per_run: int
    every_n_policy_calls: int
    max_module_calls: int | None


class PI05SemanticActivationCapture:
    """Resolve PI0.5 semantic roles lazily once the selected model is loaded."""

    def __init__(
        self,
        settings: Pi05CaptureSettings,
        *,
        recorder: Any,
        model_ref: Mapping[str, Any],
    ):
        self.settings = settings
        self.recorder = recorder
        self.model_ref = {
            **dict(model_ref),
            "capture_profile_id": settings.profile_id,
            "capture_profile_sha256": settings.profile_sha256,
            "semantic_roles": list(settings.semantic_roles),
            "user_selectors": list(settings.user_selectors),
            "capture_storage": settings.storage.value,
            "capture_max_bytes_per_run": settings.max_bytes_per_run,
        }
        self._collector: ContextualActivationCollector | None = None
        self._model: Any | None = None
        self._lock = threading.Lock()

    def session(self, model: Any, context: Any) -> Any:
        collector = self._collector_for(model)
        return collector.session(model, context)

    def reset_budget(self) -> None:
        if self._collector is not None:
            self._collector.reset_budget()

    def _collector_for(self, model: Any) -> ContextualActivationCollector:
        with self._lock:
            if self._collector is not None and self._model is model:
                return self._collector
            discovery = discover_pi05_semantic_probes(
                model,
                user_selectors=self.settings.user_selectors,
            )
            by_role = {probe.role: probe for probe in discovery.probes}
            missing = [
                role
                for role in self.settings.semantic_roles
                if role not in by_role
            ]
            missing.extend(
                f"user:{selector}"
                for selector in self.settings.user_selectors
                if f"user:{selector}" in discovery.missing_roles
            )
            if missing:
                raise Pi05CaptureConfigurationError(
                    "PI0.5 capture profile roles were not found in the loaded model: "
                    f"{sorted(set(missing))}"
                )
            selected = [
                by_role[role] for role in self.settings.semantic_roles
            ]
            selected.extend(
                probe
                for probe in discovery.probes
                if probe.role.startswith("user.")
            )
            if not selected:
                raise Pi05CaptureConfigurationError(
                    "PI0.5 capture profile resolved no model modules"
                )
            profile = CaptureProfile(
                probes=tuple(
                    probe.to_module_probe_spec(
                        storage=self.settings.storage,
                        every_n_policy_calls=self.settings.every_n_policy_calls,
                        max_module_calls=self.settings.max_module_calls,
                    )
                    for probe in selected
                ),
                output_dir=self.settings.output_dir,
                max_bytes_per_run=self.settings.max_bytes_per_run,
            )
            self._collector = ContextualActivationCollector(
                profile,
                recorder=self.recorder,
                model_ref=self.model_ref,
            )
            self._model = model
            return self._collector


def load_pi05_capture_settings(
    path: str | Path,
    *,
    storage_override: str | CaptureStorageMode | None = None,
    output_dir: str | Path | None = None,
    max_bytes_per_run_override: int | None = None,
) -> Pi05CaptureSettings:
    profile_path = Path(path).resolve()
    try:
        profile_bytes = profile_path.read_bytes()
        profile_text = profile_bytes.decode("utf-8")
        loaded = yaml.safe_load(profile_text)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise Pi05CaptureConfigurationError(
            f"cannot load PI0.5 capture profile: {type(error).__name__}"
        ) from error
    if not isinstance(loaded, dict):
        raise Pi05CaptureConfigurationError(
            "PI0.5 capture profile must be a mapping"
        )
    allowed = {
        "format",
        "id",
        "storage",
        "output_dir",
        "max_bytes_per_run",
        "every_n_policy_calls",
        "max_module_calls",
        "semantic_roles",
        "user_selectors",
        "claim_boundary",
    }
    unknown = sorted(set(loaded) - allowed)
    if unknown:
        raise Pi05CaptureConfigurationError(
            f"unexpected PI0.5 capture profile fields: {unknown}"
        )
    if loaded.get("format") != _CAPTURE_PROFILE_FORMAT:
        raise Pi05CaptureConfigurationError(
            f"capture profile format must be {_CAPTURE_PROFILE_FORMAT}"
        )
    profile_id = loaded.get("id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise Pi05CaptureConfigurationError(
            "capture profile id must be a non-empty string"
        )
    semantic_roles = _string_list(
        loaded.get("semantic_roles"),
        "semantic_roles",
        allow_empty=True,
    )
    unknown_roles = sorted(set(semantic_roles) - _SEMANTIC_ROLES)
    if unknown_roles:
        raise Pi05CaptureConfigurationError(
            f"unknown PI0.5 semantic roles: {unknown_roles}"
        )
    user_selectors = _string_list(
        loaded.get("user_selectors", []),
        "user_selectors",
        allow_empty=True,
    )
    if not semantic_roles and not user_selectors:
        raise Pi05CaptureConfigurationError(
            "capture profile requires semantic_roles or user_selectors"
        )
    storage_value = (
        storage_override
        if storage_override is not None
        else loaded.get("storage", CaptureStorageMode.SUMMARY.value)
    )
    try:
        storage = CaptureStorageMode(storage_value)
    except ValueError as error:
        raise Pi05CaptureConfigurationError(
            "capture storage must be summary, npy, or safetensors"
        ) from error
    configured_output = output_dir
    if configured_output is None and loaded.get("output_dir") is not None:
        candidate = Path(str(loaded["output_dir"]))
        configured_output = (
            candidate
            if candidate.is_absolute()
            else profile_path.parent / candidate
        )
    resolved_output = (
        Path(configured_output).resolve()
        if configured_output is not None
        else None
    )
    if storage is not CaptureStorageMode.SUMMARY and resolved_output is None:
        raise Pi05CaptureConfigurationError(
            "capture output_dir is required for full activation storage"
        )
    max_bytes = (
        max_bytes_per_run_override
        if max_bytes_per_run_override is not None
        else loaded.get("max_bytes_per_run", 64 * 1024 * 1024)
    )
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes <= 0
    ):
        raise Pi05CaptureConfigurationError(
            "capture max_bytes_per_run must be a positive integer"
        )
    every_n = loaded.get("every_n_policy_calls", 1)
    if (
        not isinstance(every_n, int)
        or isinstance(every_n, bool)
        or every_n <= 0
    ):
        raise Pi05CaptureConfigurationError(
            "capture every_n_policy_calls must be a positive integer"
        )
    max_module_calls = loaded.get("max_module_calls")
    if max_module_calls is not None and (
        not isinstance(max_module_calls, int)
        or isinstance(max_module_calls, bool)
        or max_module_calls <= 0
    ):
        raise Pi05CaptureConfigurationError(
            "capture max_module_calls must be a positive integer"
        )
    claim_boundary = loaded.get("claim_boundary")
    if not isinstance(claim_boundary, dict) or not all(
        claim_boundary.get(key) is True
        for key in ("summaries_are_observational", "no_causal_claims")
    ):
        raise Pi05CaptureConfigurationError(
            "capture claim_boundary must preserve observational and no-causal-claim flags"
        )
    return Pi05CaptureSettings(
        profile_id=profile_id,
        profile_sha256=(
            f"sha256:{hashlib.sha256(profile_bytes).hexdigest()}"
        ),
        semantic_roles=semantic_roles,
        user_selectors=user_selectors,
        storage=storage,
        output_dir=resolved_output,
        max_bytes_per_run=max_bytes,
        every_n_policy_calls=every_n,
        max_module_calls=max_module_calls,
    )


@dataclass(frozen=True)
class Pi05SemanticProbe:
    role: str
    module_name: str
    module_kind: str
    phase: str
    layer_index: int | None = None
    denoise_step_from_module_call: bool = False

    def to_module_probe_spec(
        self,
        *,
        storage: CaptureStorageMode = CaptureStorageMode.SUMMARY,
        every_n_policy_calls: int = 1,
        max_module_calls: int | None = None,
    ) -> ModuleProbeSpec:
        return ModuleProbeSpec(
            probe_id=f"pi05_{_safe_role(self.role)}",
            selector=self.module_name,
            module_kind=self.module_kind,
            phase=self.phase,
            storage=storage,
            every_n_policy_calls=every_n_policy_calls,
            max_module_calls=max_module_calls,
            denoise_step_from_module_call=self.denoise_step_from_module_call,
        )


@dataclass(frozen=True)
class Pi05ProbeDiscoveryReport:
    probes: tuple[Pi05SemanticProbe, ...]
    missing_roles: tuple[str, ...]
    available_module_count: int

    @property
    def complete(self) -> bool:
        return not self.missing_roles

    def to_capture_profile(
        self,
        *,
        storage: CaptureStorageMode = CaptureStorageMode.SUMMARY,
        output_dir: str | Path | None = None,
        max_bytes_per_run: int = 64 * 1024 * 1024,
        every_n_policy_calls: int = 1,
        max_module_calls: int | None = None,
    ) -> CaptureProfile:
        return CaptureProfile(
            probes=tuple(
                probe.to_module_probe_spec(
                    storage=storage,
                    every_n_policy_calls=every_n_policy_calls,
                    max_module_calls=max_module_calls,
                )
                for probe in self.probes
            ),
            output_dir=Path(output_dir) if output_dir is not None else None,
            max_bytes_per_run=max_bytes_per_run,
        )


def discover_pi05_semantic_probes(
    model: Any,
    *,
    user_selectors: tuple[str, ...] | list[str] = (),
) -> Pi05ProbeDiscoveryReport:
    """Discover semantic PI0.5 boundaries without assuming one exact wrapper prefix."""
    named_modules = list(model.named_modules())
    module_names = [str(name) for name, _module in named_modules if str(name)]
    probes: list[Pi05SemanticProbe] = []
    missing: list[str] = []

    for family, pattern in _LAYER_FAMILIES.items():
        indexed = sorted(
            (
                (int(match.group(1)), module_name)
                for module_name in module_names
                if (match := pattern.search(module_name)) is not None
            ),
            key=lambda item: item[0],
        )
        selected, family_missing = _select_layer_triad(indexed)
        missing.extend(f"{family}.{position}" for position in family_missing)
        for position, (layer_index, module_name) in selected.items():
            denoising = family == "action_expert"
            probes.append(
                Pi05SemanticProbe(
                    role=f"{family}.{position}",
                    module_name=module_name,
                    module_kind=(
                        "action_expert_mlp"
                        if family == "action_expert"
                        else f"{family}_block"
                    ),
                    phase="denoise" if denoising else "prefix",
                    layer_index=layer_index,
                    denoise_step_from_module_call=denoising,
                )
            )

    for role, (suffix, module_kind) in _PROJECTIONS.items():
        matches = [
            module_name
            for module_name in module_names
            if module_name == suffix or module_name.endswith(f".{suffix}")
        ]
        if len(matches) != 1:
            missing.append(role)
            continue
        probes.append(
            Pi05SemanticProbe(
                role=role,
                module_name=matches[0],
                module_kind=module_kind,
                phase="denoise",
                denoise_step_from_module_call=True,
            )
        )

    for selector_index, selector in enumerate(user_selectors):
        matches = [
            module_name
            for module_name in module_names
            if fnmatch.fnmatchcase(module_name, selector)
        ]
        if not matches:
            missing.append(f"user:{selector}")
            continue
        for match_index, module_name in enumerate(matches):
            probes.append(
                Pi05SemanticProbe(
                    role=f"user.{selector_index:03d}.{match_index:03d}",
                    module_name=module_name,
                    module_kind="user_selected",
                    phase="user",
                    denoise_step_from_module_call=False,
                )
            )

    return Pi05ProbeDiscoveryReport(
        probes=tuple(probes),
        missing_roles=tuple(missing),
        available_module_count=len(module_names),
    )


def _select_layer_triad(
    indexed: list[tuple[int, str]],
) -> tuple[dict[str, tuple[int, str]], tuple[str, ...]]:
    if not indexed:
        return {}, ("early", "middle", "late")
    if len(indexed) == 1:
        return {"early": indexed[0]}, ("middle", "late")
    if len(indexed) == 2:
        return {"early": indexed[0], "late": indexed[-1]}, ("middle",)
    return {
        "early": indexed[0],
        "middle": indexed[(len(indexed) - 1) // 2],
        "late": indexed[-1],
    }, ()


def _safe_role(role: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", role).strip("_")


def _string_list(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Pi05CaptureConfigurationError(
            f"capture {field_name} must be a list of strings"
        )
    if not allow_empty and not value:
        raise Pi05CaptureConfigurationError(
            f"capture {field_name} must not be empty"
        )
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise Pi05CaptureConfigurationError(
            f"capture {field_name} must contain non-empty strings"
        )
    if len(set(value)) != len(value):
        raise Pi05CaptureConfigurationError(
            f"capture {field_name} must not contain duplicates"
        )
    return tuple(value)
