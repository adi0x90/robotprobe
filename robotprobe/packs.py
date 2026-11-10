from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robotprobe.mech.capabilities import (
    MechAdapterCapabilities,
    is_mech_capability_map,
    validate_mech_adapter_capabilities,
)
from robotprobe.plugins import PluginIssue, load_plugin_entrypoint, load_plugin_manifest, validate_plugin_manifest
from robotprobe.runtime.capabilities import (
    is_runtime_capability_map,
    validate_runtime_component_capabilities,
)


PACK_TEMPLATES: dict[str, dict[str, Any]] = {
    "ros2-command-authority": {
        "plugin_id": "robotprobe.adapters.ros2_command_authority",
        "plugin_type": "adapter",
        "entrypoint": "robotprobe.integrations.ros2.mcap_adapter:ingest_ros2_mcap",
        "release_classification": "trace_only",
        "hardware_allowed": False,
        "files": {
            "contracts/command_authority.yaml": """id: ros2_command_authority_boundary
type: temporal_contract
window:
  start_step: 0
must_never:
  field: command_authority.source
  eq: bypass_proxy
violation:
  contract: ROS2CommandAuthorityContract
  check: unauthorized_command_source
  severity: 0.95
""",
            "adapters/ros2_command_authority.md": """# ROS 2 Command Authority Adapter

This template is for trace-only ROS 2 command-authority audits. It is meant to record and inspect
who produced command messages, where supervision gates sit, and whether a bypass proxy appears in
the command path.
""",
        },
        "proves": "defines a trace-only ROS 2 command-authority audit pack",
        "does_not_prove": "safe hardware execution or ROS 2 exploitability",
    },
    "vla-eval-adapter": {
        "plugin_id": "robotprobe.adapters.vla_eval",
        "plugin_type": "adapter",
        "entrypoint": "robotprobe.integrations.vla_eval.episode_ingest:ingest_vla_eval_episode",
        "release_classification": "sim_only",
        "hardware_allowed": False,
        "files": {
            "adapters/vla_eval.md": """# vla-eval Adapter

This template is for wrapping AllenAI-style VLA evaluation loops with RobotProbe trace hooks,
paired run IDs, intervention manifests, and RobotSec checks.
""",
        },
        "proves": "defines a simulator-only vla-eval adapter pack scaffold",
        "does_not_prove": "benchmark compatibility without running the upstream harness",
    },
    "isaac-lab-arena-adapter": {
        "plugin_id": "robotprobe.adapters.isaac_lab_arena",
        "plugin_type": "adapter",
        "entrypoint": "robotprobe.integrations.isaac_lab_arena.trace_ingest:ingest_isaac_lab_arena_trace",
        "release_classification": "sim_only",
        "hardware_allowed": False,
        "files": {
            "adapters/isaac_lab_arena.md": """# Isaac Lab-Arena Adapter

This template is for mapping Isaac Lab-Arena episodes into RobotProbe traces and incident bundles.
It should stay simulator-only until a project adds explicit hardware-shadow safety envelopes.
""",
        },
        "proves": "defines a simulator-only Isaac Lab-Arena adapter pack scaffold",
        "does_not_prove": "Isaac Sim runtime availability or physical robot safety",
    },
}


@dataclass(frozen=True)
class PackValidationReport:
    path: str
    valid: bool
    id: str | None = None
    version: str | None = None
    plugin_count: int = 0
    manifest_sha256: str | None = None
    issues: list[PluginIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def validate_pack(path: str | Path, *, allow_hardware_executable: bool = False) -> PackValidationReport:
    pack_dir = Path(path)
    manifest_path = pack_dir / "manifest.yaml"
    if not manifest_path.exists():
        return PackValidationReport(
            path=str(pack_dir),
            valid=False,
            issues=[PluginIssue("manifest", "missing manifest.yaml")],
        )

    manifest = load_plugin_manifest(manifest_path)
    plugin_report = validate_plugin_manifest(
        manifest,
        allow_hardware_executable=allow_hardware_executable,
    )
    issues = list(plugin_report.issues)
    for plugin in manifest.plugins:
        for file_ref in plugin.files:
            if not (pack_dir / file_ref).exists():
                issues.append(
                    PluginIssue(
                        "files",
                        f"missing declared file: {file_ref}",
                        plugin_id=plugin.id,
                    )
                )
        if plugin.capabilities and is_mech_capability_map(plugin.capabilities):
            issues.extend(
                validate_mech_adapter_capabilities(
                    plugin.capabilities,
                    plugin_id=plugin.id,
                )
            )
        if plugin.capabilities and is_runtime_capability_map(plugin.capabilities):
            issues.extend(
                validate_runtime_component_capabilities(
                    plugin.capabilities,
                    plugin_id=plugin.id,
                    release_classification=plugin.release_classification,
                    hardware_allowed=plugin.hardware_allowed,
                )
            )
    for challenge_ref in manifest.challenges:
        if not (pack_dir / challenge_ref).exists():
            issues.append(PluginIssue("challenges", f"missing challenge file: {challenge_ref}"))
    issues.extend(_trust_metadata_issues(manifest))

    return PackValidationReport(
        path=str(pack_dir),
        valid=not any(issue.severity == "error" for issue in issues),
        id=manifest.id,
        version=manifest.version,
        plugin_count=len(manifest.plugins),
        manifest_sha256=manifest.manifest_sha256,
        issues=issues,
    )


def list_pack(path: str | Path) -> dict[str, Any]:
    pack_dir = Path(path)
    manifest = load_plugin_manifest(pack_dir / "manifest.yaml")
    return {
        "id": manifest.id,
        "version": manifest.version,
        "path": str(pack_dir),
        "trust_level": manifest.trust_level,
        "publisher": manifest.publisher,
        "compatibility": manifest.compatibility,
        "execution": manifest.execution,
        "manifest_sha256": manifest.manifest_sha256,
        "plugins": [
            {
                "id": plugin.id,
                "type": plugin.type,
                "version": plugin.version,
                "entrypoint": plugin.entrypoint,
                "release_classification": plugin.release_classification,
                "hardware_allowed": plugin.hardware_allowed,
                "capabilities": plugin.capabilities,
                "files": plugin.files,
            }
            for plugin in manifest.plugins
        ],
        "challenges": manifest.challenges,
    }


def build_pack_registry(
    paths: list[str | Path] | tuple[str | Path, ...],
    *,
    out_path: str | Path | None = None,
    allow_hardware_executable: bool = False,
) -> dict[str, Any]:
    pack_dirs = _discover_pack_dirs(paths)
    packs = []
    for pack_dir in pack_dirs:
        validation = validate_pack(
            pack_dir,
            allow_hardware_executable=allow_hardware_executable,
        )
        listing = list_pack(pack_dir) if validation.id else {"plugins": [], "challenges": []}
        packs.append(
            {
                "id": validation.id,
                "version": validation.version,
                "path": str(pack_dir),
                "valid": validation.valid,
                "plugin_count": validation.plugin_count,
                "trust_level": listing.get("trust_level"),
                "publisher": listing.get("publisher", {}),
                "compatibility": listing.get("compatibility", {}),
                "execution": listing.get("execution", {}),
                "manifest_sha256": validation.manifest_sha256,
                "plugins": listing.get("plugins", []),
                "challenges": listing.get("challenges", []),
                "issues": [dataclasses.asdict(issue) for issue in validation.issues],
            }
        )

    registry = {
        "format": "robotprobe.pack_registry.v0",
        "pack_count": len(packs),
        "packs": packs,
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return registry


def load_pack_registry(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _discover_pack_dirs(paths: list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    discovered: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name == "manifest.yaml":
            discovered.add(path.parent)
            continue
        if (path / "manifest.yaml").exists():
            discovered.add(path)
            continue
        if path.is_dir():
            for manifest in path.rglob("manifest.yaml"):
                discovered.add(manifest.parent)
    return sorted(discovered)


def init_pack(
    path: str | Path,
    *,
    pack_id: str,
    plugin_id: str | None = None,
    plugin_type: str | None = None,
    version: str = "0.1.0",
    template: str | None = None,
) -> dict[str, Any]:
    pack_dir = Path(path)
    pack_dir.mkdir(parents=True, exist_ok=True)
    for dirname in (
        "adapters",
        "interventions",
        "contracts",
        "metrics",
        "defenses",
        "exporters",
        "challenges",
        "fixtures",
        "golden",
        "docs",
        "tests",
    ):
        (pack_dir / dirname).mkdir(exist_ok=True)

    template_data = PACK_TEMPLATES.get(template or "")
    if template and template_data is None:
        raise ValueError(f"unknown pack template: {template}")
    if template_data is not None:
        plugin_id = str(template_data["plugin_id"])
        plugin_type = str(template_data["plugin_type"])
    if plugin_id is None or plugin_type is None:
        raise ValueError("plugin_id and plugin_type are required without a template")

    files = dict(template_data.get("files", {})) if template_data else {}
    for file_ref, content in files.items():
        file_path = pack_dir / file_ref
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(str(content), encoding="utf-8")

    release_classification = str(template_data.get("release_classification", "sim_only")) if template_data else "sim_only"
    hardware_allowed = bool(template_data.get("hardware_allowed", False)) if template_data else False
    entrypoint = str(template_data["entrypoint"]) if template_data and template_data.get("entrypoint") else None
    proves = str(template_data.get("proves", "describes one simulator-only RobotProbe extension point")) if template_data else "describes one simulator-only RobotProbe extension point"
    does_not_prove = str(template_data.get("does_not_prove", "safe hardware execution")) if template_data else "safe hardware execution"
    files_section = ""
    if files:
        files_section = "    files:\n" + "".join(f"      - {file_ref}\n" for file_ref in sorted(files))
    entrypoint_section = f"    entrypoint: {entrypoint}\n" if entrypoint else ""

    manifest = f"""id: {pack_id}
version: {version}
trust_level: untrusted_local_pack
execution:
  imports_code_during_validate: false
  imports_code_during_test: false
  network_access_required: false
plugins:
  - id: {plugin_id}
    type: {plugin_type}
    version: {version}
{entrypoint_section.rstrip()}
{files_section.rstrip()}
    safety:
      release_classification: {release_classification}
      hardware_allowed: {str(hardware_allowed).lower()}
    claims:
      proves:
        - {proves}
      does_not_prove:
        - hardware certification
        - {does_not_prove}
"""
    (pack_dir / "manifest.yaml").write_text(manifest, encoding="utf-8")
    (pack_dir / "docs" / "README.md").write_text(
        f"""# {pack_id}

This RobotProbe pack is scaffolded as `{release_classification}`.

Validate it with:

```bash
robotprobe pack validate {pack_dir}
```
""",
        encoding="utf-8",
    )
    return {
        "id": pack_id,
        "version": version,
        "path": str(pack_dir),
        "plugin_id": plugin_id,
        "plugin_type": plugin_type,
        "template": template,
    }


def test_pack(
    path: str | Path,
    *,
    allow_hardware_executable: bool = False,
    allow_code: bool = False,
) -> PackValidationReport:
    report = validate_pack(path, allow_hardware_executable=allow_hardware_executable)
    pack_dir = Path(path)
    issues = list(report.issues)
    issues.extend(_golden_issues(pack_dir))
    manifest_path = pack_dir / "manifest.yaml"
    if manifest_path.exists():
        manifest = load_plugin_manifest(manifest_path)
        issues.extend(_code_import_issues(pack_dir, manifest, allow_code=allow_code))
    return PackValidationReport(
        path=report.path,
        valid=not any(issue.severity == "error" for issue in issues),
        id=report.id,
        version=report.version,
        plugin_count=report.plugin_count,
        manifest_sha256=report.manifest_sha256,
        issues=issues,
    )


def build_claim_card(path: str | Path) -> str:
    pack_dir = Path(path)
    manifest = load_plugin_manifest(pack_dir / "manifest.yaml")
    proves: list[str] = []
    non_claims: list[str] = []
    safety: list[str] = []
    mech_capabilities: list[str] = []
    for plugin in manifest.plugins:
        proves.extend(plugin.claims.get("proves", []))
        non_claims.extend(plugin.claims.get("does_not_prove", []))
        if plugin.release_classification:
            safety.append(f"{plugin.id}: {plugin.release_classification}, hardware_allowed={plugin.hardware_allowed}")
        if plugin.capabilities and is_mech_capability_map(plugin.capabilities):
            try:
                capabilities = MechAdapterCapabilities.from_plugin(plugin)
            except Exception:
                continue
            mech_capabilities.append(f"{plugin.id}: " + ", ".join(capabilities.summary_items()))

    lines = [
        f"# Claim Card: {manifest.id}",
        "",
        f"- Version: `{manifest.version}`",
        f"- Trust level: `{manifest.trust_level}`",
        f"- Manifest SHA256: `{manifest.manifest_sha256}`",
    ]
    if manifest.publisher:
        lines.append(f"- Publisher: `{manifest.publisher.get('name', 'unknown')}`")
    if manifest.compatibility:
        lines.append(f"- RobotProbe minimum: `{manifest.compatibility.get('robotprobe_min', 'unspecified')}`")
    lines.extend(["", "## This pack demonstrates", ""])
    lines.extend(f"- {claim}" for claim in _dedupe(proves) or ["No positive claims declared."])
    lines.extend(["", "## This pack does not demonstrate", ""])
    lines.extend(f"- {claim}" for claim in _dedupe(non_claims) or ["No non-claims declared."])
    if safety:
        lines.extend(["", "## Safety", ""])
        lines.extend(f"- {item}" for item in safety)
    if mech_capabilities:
        lines.extend(["", "## MechProbe Capabilities", ""])
        lines.extend(f"- {item}" for item in mech_capabilities)
    return "\n".join(lines) + "\n"


def _trust_metadata_issues(manifest) -> list[PluginIssue]:
    issues: list[PluginIssue] = []
    if manifest.trust_level not in {"trusted_core_pack", "reviewed_community_pack", "untrusted_local_pack"}:
        issues.append(
            PluginIssue(
                "trust_level",
                f"unknown trust_level: {manifest.trust_level}",
            )
        )
    if not manifest.publisher:
        issues.append(PluginIssue("publisher", "publisher metadata is recommended", severity="warning"))
    if not manifest.compatibility:
        issues.append(PluginIssue("compatibility", "compatibility metadata is recommended", severity="warning"))
    if not manifest.execution:
        issues.append(PluginIssue("execution", "execution policy metadata is recommended", severity="warning"))
    return issues


def _code_import_issues(pack_dir: Path, manifest, *, allow_code: bool) -> list[PluginIssue]:
    imports_during_test = bool(manifest.execution.get("imports_code_during_test", False))
    if not imports_during_test:
        return []
    if not allow_code:
        return [
            PluginIssue(
                "execution.imports_code_during_test",
                "pack test would import plugin code; rerun with --allow-code to permit entrypoint imports",
            )
        ]
    issues: list[PluginIssue] = []
    for plugin in manifest.plugins:
        if not plugin.entrypoint:
            continue
        try:
            load_plugin_entrypoint(plugin, search_paths=[pack_dir])
        except Exception as exc:
            issues.append(
                PluginIssue(
                    "entrypoint",
                    f"failed to import entrypoint {plugin.entrypoint}: {exc}",
                    plugin_id=plugin.id,
                )
            )
    return issues


def _golden_issues(pack_dir: Path) -> list[PluginIssue]:
    golden_dir = pack_dir / "golden"
    fixtures_dir = pack_dir / "fixtures"
    if not golden_dir.exists():
        return []
    issues: list[PluginIssue] = []
    for golden in sorted(golden_dir.rglob("*.json")):
        rel = golden.relative_to(golden_dir)
        fixture = fixtures_dir / rel
        if not fixture.exists():
            issues.append(PluginIssue("golden", f"missing fixture for golden file: {rel}"))
            continue
        try:
            expected = json.loads(golden.read_text(encoding="utf-8"))
            actual = json.loads(fixture.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(PluginIssue("golden", f"invalid golden json {rel}: {exc.msg}"))
            continue
        if actual != expected:
            issues.append(PluginIssue("golden", f"golden mismatch: {rel}"))
    return issues


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
