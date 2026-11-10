from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from robotprobe.plugins import PluginEntry, PluginIssue


CAPABILITY_FORMAT = "robotprobe.mech.adapter_capabilities.v0"
SUPPORT_FLAGS = (
    "supports_activation_capture",
    "supports_temporal_patch",
    "supports_action_head",
    "supports_uncertainty",
    "supports_policy_replay",
    "supports_sim_replay",
    "supports_hardware_shadow",
    "requires_white_box_model",
    "imports_heavy_runtime",
)
SAFE_DEFAULT_MODES = {
    "offline_only",
    "capture_only",
    "policy_replay_only",
    "sim_patch_only",
    "hardware_shadow_only",
}


@dataclass(frozen=True)
class MechAdapterCapabilities:
    format: str = CAPABILITY_FORMAT
    adapter_id: str = ""
    model_family: str = ""
    runtime: str = ""
    supports_activation_capture: bool = False
    supports_temporal_patch: bool = False
    supports_action_head: bool = False
    supports_uncertainty: bool = False
    supports_policy_replay: bool = False
    supports_sim_replay: bool = False
    supports_hardware_shadow: bool = False
    requires_white_box_model: bool = True
    imports_heavy_runtime: bool = True
    safe_default_mode: str = "offline_only"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MechAdapterCapabilities":
        values = {field.name: data.get(field.name, field.default) for field in dataclasses.fields(cls)}
        return cls(**values)

    @classmethod
    def from_plugin(cls, plugin: PluginEntry) -> "MechAdapterCapabilities":
        if not plugin.capabilities:
            raise ValueError(f"plugin {plugin.id} does not declare MechProbe capabilities")
        return cls.from_dict(plugin.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def summary_items(self) -> list[str]:
        items = [
            f"adapter_id={self.adapter_id or 'unspecified'}",
            f"model_family={self.model_family or 'unspecified'}",
            f"runtime={self.runtime or 'unspecified'}",
        ]
        items.extend(f"{flag}={str(getattr(self, flag)).lower()}" for flag in SUPPORT_FLAGS)
        items.append(f"safe_default_mode={self.safe_default_mode}")
        return items


def is_mech_capability_map(value: dict[str, Any]) -> bool:
    return value.get("format") == CAPABILITY_FORMAT or any(flag in value for flag in SUPPORT_FLAGS)


def validate_mech_adapter_capabilities(data: dict[str, Any], *, plugin_id: str | None = None) -> list[PluginIssue]:
    issues: list[PluginIssue] = []
    if data.get("format") != CAPABILITY_FORMAT:
        issues.append(
            PluginIssue(
                "capabilities.format",
                f"expected {CAPABILITY_FORMAT}",
                plugin_id=plugin_id,
            )
        )
    for field_name in ("adapter_id", "model_family", "runtime"):
        if not isinstance(data.get(field_name), str) or not str(data.get(field_name)).strip():
            issues.append(
                PluginIssue(
                    f"capabilities.{field_name}",
                    f"{field_name} must be a non-empty string",
                    plugin_id=plugin_id,
                )
            )
    for flag in SUPPORT_FLAGS:
        if not isinstance(data.get(flag), bool):
            issues.append(
                PluginIssue(
                    f"capabilities.{flag}",
                    f"{flag} must be boolean",
                    plugin_id=plugin_id,
                )
            )
    safe_default_mode = data.get("safe_default_mode")
    if safe_default_mode not in SAFE_DEFAULT_MODES:
        issues.append(
            PluginIssue(
                "capabilities.safe_default_mode",
                f"safe_default_mode must be one of {sorted(SAFE_DEFAULT_MODES)}",
                plugin_id=plugin_id,
            )
        )
    if data.get("supports_temporal_patch") and not (
        data.get("supports_policy_replay") or data.get("supports_sim_replay")
    ):
        issues.append(
            PluginIssue(
                "capabilities.supports_temporal_patch",
                "temporal patch support requires policy or simulator replay support",
                plugin_id=plugin_id,
            )
        )
    if data.get("supports_hardware_shadow") and safe_default_mode == "capture_only":
        issues.append(
            PluginIssue(
                "capabilities.safe_default_mode",
                "hardware shadow support must not default to capture_only",
                plugin_id=plugin_id,
                severity="warning",
            )
        )
    return issues
