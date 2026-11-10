from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from robotprobe.plugins import PluginEntry, PluginManifest, load_plugin_entrypoint, load_plugin_manifest
from robotprobe.runtime.capabilities import (
    RuntimeComponentCapabilities,
    RuntimeComponentKind,
    is_runtime_capability_map,
)


@dataclass(frozen=True)
class RuntimePluginRegistry:
    pack_dir: Path
    manifest: PluginManifest
    entries: Mapping[str, PluginEntry]

    @classmethod
    def from_pack(cls, pack_dir: str | Path) -> "RuntimePluginRegistry":
        root = Path(pack_dir)
        manifest = load_plugin_manifest(root / "manifest.yaml")
        entries: dict[str, PluginEntry] = {}
        for plugin in manifest.plugins:
            if not plugin.capabilities or not is_runtime_capability_map(plugin.capabilities):
                continue
            if plugin.id in entries:
                raise ValueError(f"duplicate runtime plugin id: {plugin.id}")
            entries[plugin.id] = plugin
        return cls(pack_dir=root, manifest=manifest, entries=entries)

    def plugin(self, plugin_id: str, expected_kind: RuntimeComponentKind) -> PluginEntry:
        try:
            plugin = self.entries[plugin_id]
        except KeyError as exc:
            raise KeyError(f"runtime plugin not found: {plugin_id}") from exc
        capabilities = RuntimeComponentCapabilities.from_dict(plugin.capabilities)
        expected = RuntimeComponentKind(expected_kind)
        if capabilities.kind is not expected:
            raise ValueError(
                f"runtime plugin {plugin_id} has kind {capabilities.kind.value}, expected {expected.value}"
            )
        return plugin

    def capabilities(
        self,
        plugin_id: str,
        expected_kind: RuntimeComponentKind,
    ) -> RuntimeComponentCapabilities:
        return RuntimeComponentCapabilities.from_dict(
            self.plugin(plugin_id, expected_kind).capabilities
        )

    def resolve(self, plugin_id: str, expected_kind: RuntimeComponentKind) -> Any:
        plugin = self.plugin(plugin_id, expected_kind)
        return load_plugin_entrypoint(plugin, search_paths=[self.pack_dir])

    def instantiate(
        self,
        plugin_id: str,
        expected_kind: RuntimeComponentKind,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        factory = self.resolve(plugin_id, expected_kind)
        if not callable(factory):
            raise TypeError(f"runtime plugin entrypoint is not callable: {plugin_id}")
        return factory(**dict(options or {}))
