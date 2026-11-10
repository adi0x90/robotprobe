from __future__ import annotations

import dataclasses
import hashlib
import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


PLUGIN_TYPES = {
    "adapter",
    "intervention",
    "contract",
    "metric",
    "defense",
    "diagnoser",
    "exporter",
    "challenge_pack",
}
RELEASE_CLASSIFICATIONS = {"trace_only", "sim_only", "hardware_shadow", "hardware_executable"}


@dataclass(frozen=True)
class PluginEntry:
    id: str
    type: str
    version: str
    entrypoint: str | None = None
    files: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    claims: dict[str, list[str]] = field(default_factory=dict)

    @property
    def release_classification(self) -> str | None:
        value = self.safety.get("release_classification")
        return str(value) if value is not None else None

    @property
    def hardware_allowed(self) -> bool:
        return bool(self.safety.get("hardware_allowed"))


@dataclass(frozen=True)
class PluginManifest:
    id: str
    version: str
    plugins: list[PluginEntry]
    path: str = ""
    challenges: list[str] = field(default_factory=list)
    trust_level: str = "untrusted_local_pack"
    publisher: dict[str, Any] = field(default_factory=dict)
    signing: dict[str, Any] = field(default_factory=dict)
    compatibility: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    manifest_sha256: str = ""


@dataclass(frozen=True)
class PluginIssue:
    field: str
    message: str
    severity: str = "error"
    plugin_id: str | None = None


@dataclass(frozen=True)
class PluginValidationReport:
    path: str
    valid: bool
    plugin_count: int
    issues: list[PluginIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_plugin_manifest(path: str | Path) -> PluginManifest:
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    data = _parse_plugin_manifest_yaml(text)
    plugins = [
        PluginEntry(
            id=str(item["id"]),
            type=str(item["type"]),
            version=str(item.get("version", "")),
            entrypoint=str(item["entrypoint"]) if "entrypoint" in item else None,
            files=[str(value) for value in item.get("files", [])],
            inputs=dict(item.get("inputs", {})),
            outputs=dict(item.get("outputs", {})),
            safety=dict(item.get("safety", {})),
            capabilities=dict(item.get("capabilities", {})),
            claims=_normalize_claims(item.get("claims", {})),
        )
        for item in data.get("plugins", [])
    ]
    return PluginManifest(
        id=str(data["id"]),
        version=str(data["version"]),
        plugins=plugins,
        path=str(manifest_path),
        challenges=[str(value) for value in data.get("challenges", [])],
        trust_level=str(data.get("trust_level", "untrusted_local_pack")),
        publisher=dict(data.get("publisher", {})),
        signing=dict(data.get("signing", {})),
        compatibility=dict(data.get("compatibility", {})),
        execution=dict(data.get("execution", {})),
        manifest_sha256=f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}",
    )


def validate_plugin_manifest(
    manifest: PluginManifest,
    *,
    allow_hardware_executable: bool = False,
) -> PluginValidationReport:
    issues: list[PluginIssue] = []
    seen: set[str] = set()
    for plugin in manifest.plugins:
        if plugin.id in seen:
            issues.append(PluginIssue("id", f"duplicate plugin id: {plugin.id}", plugin_id=plugin.id))
        seen.add(plugin.id)
        if plugin.type not in PLUGIN_TYPES:
            issues.append(PluginIssue("type", f"unknown plugin type: {plugin.type}", plugin_id=plugin.id))
        if not plugin.version:
            issues.append(PluginIssue("version", "missing required field: version", plugin_id=plugin.id))
        if plugin.release_classification not in RELEASE_CLASSIFICATIONS:
            issues.append(
                PluginIssue(
                    "safety.release_classification",
                    f"unknown release classification: {plugin.release_classification}",
                    plugin_id=plugin.id,
                )
            )
        if plugin.release_classification == "hardware_executable" and not allow_hardware_executable:
            issues.append(
                PluginIssue(
                    "safety.release_classification",
                    "hardware_executable plugins are rejected unless explicitly allowed",
                    plugin_id=plugin.id,
                )
            )
        if not isinstance(plugin.safety.get("hardware_allowed"), bool):
            issues.append(PluginIssue("safety.hardware_allowed", "hardware_allowed must be boolean", plugin_id=plugin.id))
        if not plugin.claims.get("proves"):
            issues.append(PluginIssue("claims.proves", "claims.proves must contain at least one entry", plugin_id=plugin.id))
        if not plugin.claims.get("does_not_prove"):
            issues.append(
                PluginIssue(
                    "claims.does_not_prove",
                    "claims.does_not_prove must contain at least one entry",
                    plugin_id=plugin.id,
                )
            )

    return PluginValidationReport(
        path=manifest.path,
        valid=not any(issue.severity == "error" for issue in issues),
        plugin_count=len(manifest.plugins),
        issues=issues,
    )


def load_plugin_entrypoint(plugin: PluginEntry, *, search_paths: list[str | Path] | None = None) -> Any:
    if not plugin.entrypoint:
        raise ValueError(f"plugin {plugin.id} has no entrypoint")
    module_name, separator, object_path = plugin.entrypoint.partition(":")
    if not separator or not module_name or not object_path:
        raise ValueError(f"entrypoint must be module:object, got {plugin.entrypoint!r}")
    with _temporary_sys_path(search_paths or []):
        module = importlib.import_module(module_name)
    current: Any = module
    for part in object_path.split("."):
        current = getattr(current, part)
    return current


@contextmanager
def _temporary_sys_path(paths: list[str | Path]) -> Iterator[None]:
    inserted = [str(Path(path)) for path in paths]
    old_path = list(sys.path)
    try:
        sys.path[:0] = inserted
        yield
    finally:
        sys.path[:] = old_path


def _normalize_claims(value: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "proves": [str(item) for item in value.get("proves", [])],
        "does_not_prove": [str(item) for item in value.get("does_not_prove", [])],
    }


def _parse_plugin_manifest_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    lines = _meaningful_lines(text)
    index = 0
    while index < len(lines):
        indent, content = lines[index]
        if indent != 0:
            raise ValueError(f"unexpected indentation: {content!r}")
        key, value = _split_key_value(content)
        if value:
            root[key] = _parse_scalar(value)
            index += 1
            continue
        if key == "plugins":
            items, index = _parse_list_of_maps(lines, index + 1, list_indent=2)
            root[key] = items
            continue
        if key in {"publisher", "signing", "compatibility", "execution"}:
            mapping, index = _parse_map_block(lines, index + 1, map_indent=2)
            root[key] = mapping
            continue
        list_values, next_index = _parse_scalar_list(lines, index + 1, list_indent=2)
        root[key] = list_values
        index = next_index
    return root


def _parse_map_block(
    lines: list[tuple[int, str]],
    index: int,
    *,
    map_indent: int,
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    while index < len(lines):
        indent, content = lines[index]
        if indent < map_indent:
            break
        if indent == map_indent:
            key, value = _split_key_value(content)
            if value:
                result[key] = _parse_scalar(value)
                current_key = None
            else:
                result[key] = []
                current_key = key
            index += 1
            continue
        if indent == map_indent + 2 and current_key is not None and content.startswith("- "):
            container = result[current_key]
            if not isinstance(container, list):
                raise ValueError(f"cannot append list item to map field: {current_key}")
            container.append(_parse_scalar(content[2:].strip()))
            index += 1
            continue
        raise ValueError(f"unsupported map line: {content!r}")
    return result, index


def _parse_list_of_maps(
    lines: list[tuple[int, str]],
    index: int,
    *,
    list_indent: int,
) -> tuple[list[dict[str, Any]], int]:
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_nested_key: str | None = None
    while index < len(lines):
        indent, content = lines[index]
        if indent < list_indent:
            break
        if indent == list_indent and content.startswith("- "):
            current = {}
            items.append(current)
            current_nested_key = None
            first = content[2:].strip()
            if first:
                key, value = _split_key_value(first)
                current[key] = _parse_scalar(value)
            index += 1
            continue
        if current is None:
            raise ValueError(f"list entry data without list item: {content!r}")
        if indent == list_indent + 2:
            key, value = _split_key_value(content)
            if value:
                current[key] = _parse_scalar(value)
                current_nested_key = None
            else:
                current[key] = []
                current_nested_key = key
            index += 1
            continue
        if indent == list_indent + 4 and current_nested_key is not None:
            container = current[current_nested_key]
            if content.startswith("- "):
                if not isinstance(container, list):
                    raise ValueError(f"cannot append list item to map field: {current_nested_key}")
                container.append(_parse_scalar(content[2:].strip()))
            else:
                if not isinstance(container, dict):
                    current[current_nested_key] = {}
                    container = current[current_nested_key]
                child_key, child_value = _split_key_value(content)
                container[child_key] = (
                    _parse_scalar(child_value) if child_value else []
                )
            index += 1
            continue
        if indent == list_indent + 6 and current_nested_key is not None:
            parent = current[current_nested_key]
            if not isinstance(parent, dict):
                raise ValueError(f"nested list without map parent: {content!r}")
            last_key = next(reversed(parent))
            if not isinstance(parent[last_key], list):
                parent[last_key] = []
            if not content.startswith("- "):
                raise ValueError(f"expected list item: {content!r}")
            parent[last_key].append(_parse_scalar(content[2:].strip()))
            index += 1
            continue
        raise ValueError(f"unsupported plugin manifest line: {content!r}")
    return items, index


def _parse_scalar_list(
    lines: list[tuple[int, str]],
    index: int,
    *,
    list_indent: int,
) -> tuple[list[Any], int]:
    values: list[Any] = []
    while index < len(lines):
        indent, content = lines[index]
        if indent < list_indent:
            break
        if indent == list_indent and content.startswith("- "):
            values.append(_parse_scalar(content[2:].strip()))
            index += 1
            continue
        raise ValueError(f"expected scalar list item: {content!r}")
    return values, index


def _meaningful_lines(text: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        result.append((len(line) - len(line.lstrip(" ")), line.strip()))
    return result


def _split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"expected key: value line, got {line!r}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
