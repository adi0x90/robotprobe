from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any

from robotprobe.plugins import PluginIssue
from robotprobe.runtime.types import ExecutionMode


RUNTIME_CAPABILITY_FORMAT = "robotprobe.runtime.component.v0"


class RuntimeComponentKind(str, Enum):
    POLICY = "policy"
    EMBODIMENT = "embodiment"
    BACKEND = "backend"
    SCENARIO = "scenario"


@dataclass(frozen=True)
class RuntimeComponentCapabilities:
    format: str
    kind: RuntimeComponentKind
    component_id: str
    command_spaces: tuple[str, ...] = ()
    observation_streams: tuple[str, ...] = ()
    runtime_family: str = ""
    is_hardware: bool = False
    supports_reset: bool = False
    supports_ground_truth: bool = False
    supports_shadow: bool = False
    can_apply_hardware_commands: bool = False
    requires_reset: bool = False
    optional_dependencies: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeComponentCapabilities":
        validation = validate_runtime_component_capabilities(data)
        errors = [issue for issue in validation if issue.severity == "error"]
        if errors:
            rendered = "; ".join(f"{issue.field}: {issue.message}" for issue in errors)
            raise ValueError(f"invalid runtime component capabilities: {rendered}")
        return cls(
            format=str(data["format"]),
            kind=RuntimeComponentKind(str(data["kind"])),
            component_id=str(data["component_id"]),
            command_spaces=_string_tuple(data.get("command_spaces", ())),
            observation_streams=_string_tuple(data.get("observation_streams", ())),
            runtime_family=str(data.get("runtime_family", "")),
            is_hardware=bool(data.get("is_hardware", False)),
            supports_reset=bool(data.get("supports_reset", False)),
            supports_ground_truth=bool(data.get("supports_ground_truth", False)),
            supports_shadow=bool(data.get("supports_shadow", False)),
            can_apply_hardware_commands=bool(data.get("can_apply_hardware_commands", False)),
            requires_reset=bool(data.get("requires_reset", False)),
            optional_dependencies=_string_tuple(data.get("optional_dependencies", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["kind"] = self.kind.value
        for key in ("command_spaces", "observation_streams", "optional_dependencies"):
            data[key] = list(data[key])
        return data


@dataclass(frozen=True)
class RuntimeCompatibilityReport:
    valid: bool
    common_command_spaces: tuple[str, ...]
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "common_command_spaces": list(self.common_command_spaces),
            "issues": list(self.issues),
        }


def is_runtime_capability_map(value: dict[str, Any]) -> bool:
    return value.get("format") == RUNTIME_CAPABILITY_FORMAT or {
        "kind",
        "component_id",
        "command_spaces",
    }.issubset(value)


def validate_runtime_component_capabilities(
    data: dict[str, Any],
    *,
    plugin_id: str | None = None,
    release_classification: str | None = None,
    hardware_allowed: bool | None = None,
) -> list[PluginIssue]:
    issues: list[PluginIssue] = []
    if data.get("format") != RUNTIME_CAPABILITY_FORMAT:
        issues.append(
            PluginIssue(
                "capabilities.format",
                f"expected {RUNTIME_CAPABILITY_FORMAT}",
                plugin_id=plugin_id,
            )
        )

    raw_kind = data.get("kind")
    try:
        kind = RuntimeComponentKind(str(raw_kind))
    except ValueError:
        kind = None
        issues.append(
            PluginIssue(
                "capabilities.kind",
                f"kind must be one of {[item.value for item in RuntimeComponentKind]}",
                plugin_id=plugin_id,
            )
        )

    if not isinstance(data.get("component_id"), str) or not str(data.get("component_id")).strip():
        issues.append(
            PluginIssue(
                "capabilities.component_id",
                "component_id must be a non-empty string",
                plugin_id=plugin_id,
            )
        )

    command_spaces = data.get("command_spaces")
    needs_command_space = kind in {
        RuntimeComponentKind.POLICY,
        RuntimeComponentKind.EMBODIMENT,
        RuntimeComponentKind.BACKEND,
    } or kind is None
    if not isinstance(command_spaces, list) or (
        needs_command_space
        and (not command_spaces or not all(isinstance(value, str) and value for value in command_spaces))
    ):
        issues.append(
            PluginIssue(
                "capabilities.command_spaces",
                "command_spaces must be a non-empty list of strings for policy, embodiment, and backend",
                plugin_id=plugin_id,
            )
        )

    for field_name in (
        "is_hardware",
        "supports_reset",
        "supports_ground_truth",
        "supports_shadow",
        "can_apply_hardware_commands",
        "requires_reset",
    ):
        if field_name in data and not isinstance(data[field_name], bool):
            issues.append(
                PluginIssue(
                    f"capabilities.{field_name}",
                    f"{field_name} must be boolean",
                    plugin_id=plugin_id,
                )
            )

    is_hardware = data.get("is_hardware") is True
    supports_shadow = data.get("supports_shadow") is True
    can_apply = data.get("can_apply_hardware_commands") is True
    if supports_shadow and not is_hardware:
        issues.append(
            PluginIssue(
                "capabilities.supports_shadow",
                "supports_shadow requires is_hardware=true",
                plugin_id=plugin_id,
            )
        )
    if can_apply and not is_hardware:
        issues.append(
            PluginIssue(
                "capabilities.can_apply_hardware_commands",
                "hardware command support requires is_hardware=true",
                plugin_id=plugin_id,
            )
        )
    if is_hardware and release_classification not in {None, "hardware_shadow", "hardware_executable"}:
        issues.append(
            PluginIssue(
                "safety.release_classification",
                "hardware backends must be classified hardware_shadow or hardware_executable",
                plugin_id=plugin_id,
            )
        )
    if can_apply and release_classification is not None and release_classification != "hardware_executable":
        issues.append(
            PluginIssue(
                "safety.release_classification",
                "can_apply_hardware_commands requires hardware_executable classification",
                plugin_id=plugin_id,
            )
        )
    if can_apply and hardware_allowed is not None and hardware_allowed is not True:
        issues.append(
            PluginIssue(
                "safety.hardware_allowed",
                "can_apply_hardware_commands requires hardware_allowed=true",
                plugin_id=plugin_id,
            )
        )
    return issues


def check_runtime_compatibility(
    *,
    policy: RuntimeComponentCapabilities,
    embodiment: RuntimeComponentCapabilities,
    backend: RuntimeComponentCapabilities,
    scenario: RuntimeComponentCapabilities,
    execution_mode: ExecutionMode,
) -> RuntimeCompatibilityReport:
    issues: list[str] = []
    expected = {
        "policy": RuntimeComponentKind.POLICY,
        "embodiment": RuntimeComponentKind.EMBODIMENT,
        "backend": RuntimeComponentKind.BACKEND,
        "scenario": RuntimeComponentKind.SCENARIO,
    }
    components = {
        "policy": policy,
        "embodiment": embodiment,
        "backend": backend,
        "scenario": scenario,
    }
    for name, component in components.items():
        if component.kind is not expected[name]:
            issues.append(f"{name} component has kind {component.kind.value}, expected {expected[name].value}")

    common = set(policy.command_spaces) & set(embodiment.command_spaces) & set(backend.command_spaces)
    if not common:
        issues.append("policy, embodiment, and backend do not share a common command space")
    if scenario.requires_reset and not backend.supports_reset:
        issues.append("scenario requires reset but backend does not support reset")

    mode = ExecutionMode(execution_mode)
    if mode is ExecutionMode.SIMULATION and backend.is_hardware:
        issues.append("simulation execution mode cannot use a hardware backend")
    if mode is ExecutionMode.HARDWARE_SHADOW and not (backend.is_hardware and backend.supports_shadow):
        issues.append("hardware shadow execution requires a hardware backend with shadow support")
    if mode is ExecutionMode.HARDWARE_EXECUTABLE and not (
        backend.is_hardware and backend.can_apply_hardware_commands
    ):
        issues.append("hardware executable mode requires an executable hardware backend")

    return RuntimeCompatibilityReport(
        valid=not issues,
        common_command_spaces=tuple(sorted(common)),
        issues=tuple(issues),
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("capability list fields must be lists")
    return tuple(str(item) for item in value)
