from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ReplayLevel(Enum):
    TRACE_PLAYBACK = "L0_TRACE_PLAYBACK"
    POLICY_REPLAY = "L1_POLICY_REPLAY"
    SIMULATOR_REPLAY = "L2_SIM_REPLAY"
    SHADOW_HARDWARE_REPLAY = "L3_SHADOW_HARDWARE_REPLAY"


@dataclass(frozen=True)
class InterventionManifest:
    id: str
    name: str
    target_layers: list[str]
    contract: str
    window: dict[str, Any] = field(default_factory=dict)
    mutation: dict[str, Any] = field(default_factory=dict)
    expected_effects: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    path: str = ""
    digest: str = ""


def load_intervention_manifest(path: str | Path) -> InterventionManifest:
    manifest_path = Path(path)
    data = manifest_path.read_bytes()
    parsed = _parse_manifest_yaml(data.decode("utf-8"))
    return InterventionManifest(
        id=str(parsed["id"]),
        name=str(parsed["name"]),
        target_layers=list(parsed.get("target_layers", [])),
        contract=str(parsed["contract"]),
        window=dict(parsed.get("window", {})),
        mutation=dict(parsed.get("mutation", {})),
        expected_effects=dict(parsed.get("expected_effects", {})),
        safety=dict(parsed.get("safety", {})),
        path=str(manifest_path),
        digest=f"sha256:{hashlib.sha256(data).hexdigest()}",
    )


def _parse_manifest_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by intervention manifests.

    This avoids adding a PyYAML dependency to the lightweight local harness.
    Supported shape: top-level scalars, top-level lists, and one-level nested maps.
    """
    root: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue

        if not line.startswith(" "):
            key, value = _split_key_value(line)
            current_key = key
            if value == "":
                root[key] = {}
            else:
                root[key] = _parse_scalar(value)
            continue

        if current_key is None:
            raise ValueError(f"nested manifest line without parent: {raw_line!r}")

        stripped = line.strip()
        if stripped.startswith("- "):
            if not isinstance(root.get(current_key), list):
                root[current_key] = []
            root[current_key].append(_parse_scalar(stripped[2:].strip()))
            continue

        child_key, child_value = _split_key_value(stripped)
        if not isinstance(root.get(current_key), dict):
            raise ValueError(f"manifest key {current_key!r} cannot contain nested fields")
        root[current_key][child_key] = _parse_scalar(child_value)

    return root


def _split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"expected key: value line, got {line!r}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
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
