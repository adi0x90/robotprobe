from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from robotprobe.core import ReplayLevel
from robotprobe.packs import validate_pack
from robotprobe.record import TraceRecorder
from robotprobe.runtime.capabilities import (
    RuntimeComponentCapabilities,
    RuntimeComponentKind,
    check_runtime_compatibility,
)
from robotprobe.runtime.orchestrator import EpisodeRunner
from robotprobe.runtime.protocols import (
    ConfigurableCapturePolicyAdapter,
    EmbodimentAdapter,
    PolicyAdapter,
    RuntimeBackend,
    ScenarioProvider,
)
from robotprobe.runtime.registry import RuntimePluginRegistry
from robotprobe.runtime.safety import SafetyEnvelope, SafetyGateway
from robotprobe.runtime.scheduler import ActionScheduler
from robotprobe.runtime.types import ExecutionMode
from robotprobe.schema import validate_trace_file


RUNTIME_CONFIG_FORMAT = "robotprobe.runtime.run.v0"
_COMPONENT_ROLES = ("policy", "embodiment", "backend", "scenario")
_ROLE_KINDS = {
    "policy": RuntimeComponentKind.POLICY,
    "embodiment": RuntimeComponentKind.EMBODIMENT,
    "backend": RuntimeComponentKind.BACKEND,
    "scenario": RuntimeComponentKind.SCENARIO,
}


@dataclass(frozen=True)
class RuntimeConfigIssue:
    path: str
    message: str


class RuntimeConfigError(ValueError):
    def __init__(self, issues: list[RuntimeConfigIssue] | tuple[RuntimeConfigIssue, ...]):
        self.issues = tuple(issues)
        rendered = "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)
        super().__init__(f"invalid runtime configuration: {rendered}")


@dataclass(frozen=True)
class ComponentSelection:
    plugin: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeComponentsConfig:
    policy: ComponentSelection
    embodiment: ComponentSelection
    backend: ComponentSelection
    scenario: ComponentSelection

    def by_role(self) -> dict[str, ComponentSelection]:
        return {role: getattr(self, role) for role in _COMPONENT_ROLES}


@dataclass(frozen=True)
class RuntimeExecutionConfig:
    mode: ExecutionMode
    n_action_steps: int
    safety: SafetyEnvelope


@dataclass(frozen=True)
class RuntimeTraceConfig:
    out: Path
    run_id: str
    target: str
    scenario: str


@dataclass(frozen=True)
class RuntimeCaptureConfig:
    profile: Path
    storage: str | None = None
    output_dir: Path | None = None
    max_bytes_per_run: int | None = None


@dataclass(frozen=True)
class RuntimeRunConfig:
    format: str
    source_path: Path
    pack_dir: Path
    components: RuntimeComponentsConfig
    execution: RuntimeExecutionConfig
    trace: RuntimeTraceConfig
    capture: RuntimeCaptureConfig | None = None


def load_runtime_config(path: str | Path) -> RuntimeRunConfig:
    config_path = Path(path).resolve()
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeConfigError(
            [RuntimeConfigIssue("$", f"cannot load configuration: {type(error).__name__}")]
        ) from error
    if not isinstance(loaded, dict):
        raise RuntimeConfigError([RuntimeConfigIssue("$", "configuration must be a mapping")])
    data = dict(loaded)
    issues = _validate_config_shape(data)
    if issues:
        raise RuntimeConfigError(issues)

    components_data = data["components"]
    selections = {
        role: ComponentSelection(
            plugin=str(components_data[role]["plugin"]),
            options=dict(components_data[role].get("options", {})),
        )
        for role in _COMPONENT_ROLES
    }
    execution_data = dict(data["execution"])
    safety_data = dict(execution_data["safety"])
    try:
        envelope = SafetyEnvelope(
            command_space=str(safety_data["command_space"]),
            max_abs=tuple(float(value) for value in safety_data["max_abs"]),
            workspace_min=_optional_triplet(safety_data.get("workspace_min")),
            workspace_max=_optional_triplet(safety_data.get("workspace_max")),
            max_observation_age_ms=float(
                safety_data.get("max_observation_age_ms", 250.0)
            ),
        )
        mode = ExecutionMode(execution_data.get("mode", ExecutionMode.SIMULATION.value))
    except (TypeError, ValueError) as error:
        raise RuntimeConfigError(
            [RuntimeConfigIssue("execution", f"invalid execution settings: {error}")]
        ) from error

    pack_value = Path(str(data["pack"]))
    pack_dir = (
        pack_value.resolve()
        if pack_value.is_absolute()
        else (config_path.parent / pack_value).resolve()
    )
    trace_data = dict(data["trace"])
    trace_value = Path(str(trace_data["out"]))
    trace_out = (
        trace_value.resolve()
        if trace_value.is_absolute()
        else (config_path.parent / trace_value).resolve()
    )
    capture = _load_capture_config(data.get("capture"), config_path)
    return RuntimeRunConfig(
        format=RUNTIME_CONFIG_FORMAT,
        source_path=config_path,
        pack_dir=pack_dir,
        components=RuntimeComponentsConfig(**selections),
        execution=RuntimeExecutionConfig(
            mode=mode,
            n_action_steps=int(execution_data.get("n_action_steps", 1)),
            safety=envelope,
        ),
        trace=RuntimeTraceConfig(
            out=trace_out,
            run_id=str(trace_data.get("run_id", f"runtime:{config_path.stem}")),
            target=str(trace_data.get("target", selections["policy"].plugin)),
            scenario=str(trace_data.get("scenario", selections["scenario"].plugin)),
        ),
        capture=capture,
    )


def inspect_runtime_config(path: str | Path) -> dict[str, Any]:
    config = load_runtime_config(path)
    registry = RuntimePluginRegistry.from_pack(config.pack_dir)
    capabilities = _selected_capabilities(config, registry)
    compatibility = check_runtime_compatibility(
        policy=capabilities["policy"],
        embodiment=capabilities["embodiment"],
        backend=capabilities["backend"],
        scenario=capabilities["scenario"],
        execution_mode=config.execution.mode,
    )
    pack_report = validate_pack(config.pack_dir)
    configuration_issues = _execution_compatibility_issues(
        config,
        compatibility.common_command_spaces,
    )
    issues = [
        f"pack:{issue.plugin_id or '$'}:{issue.field}:{issue.message}"
        for issue in pack_report.issues
    ]
    issues.extend(compatibility.issues)
    issues.extend(configuration_issues)
    return {
        "format": "robotprobe.runtime.inspect.v0",
        "valid": pack_report.valid and compatibility.valid and not configuration_issues,
        "config_path": str(config.source_path),
        "pack": {
            "id": registry.manifest.id,
            "version": registry.manifest.version,
            "path": str(config.pack_dir),
            "manifest_sha256": registry.manifest.manifest_sha256,
            "valid": pack_report.valid,
        },
        "components": {
            role: {
                "plugin": selection.plugin,
                "component_id": capabilities[role].component_id,
                "kind": capabilities[role].kind.value,
                "option_keys": sorted(selection.options),
                "capabilities": capabilities[role].to_dict(),
            }
            for role, selection in config.components.by_role().items()
        },
        "execution": {
            "mode": config.execution.mode.value,
            "n_action_steps": config.execution.n_action_steps,
            "command_space": config.execution.safety.command_space,
        },
        "trace": {"out": str(config.trace.out)},
        "capture": (
            {
                "profile": str(config.capture.profile),
                "storage": config.capture.storage,
                "output_dir": (
                    str(config.capture.output_dir)
                    if config.capture.output_dir is not None
                    else None
                ),
                "max_bytes_per_run": config.capture.max_bytes_per_run,
            }
            if config.capture is not None
            else None
        ),
        "compatibility": compatibility.to_dict(),
        "issues": issues,
    }


def run_runtime_config(path: str | Path) -> dict[str, Any]:
    config = load_runtime_config(path)
    registry = RuntimePluginRegistry.from_pack(config.pack_dir)
    declared = _selected_capabilities(config, registry)
    compatibility = check_runtime_compatibility(
        policy=declared["policy"],
        embodiment=declared["embodiment"],
        backend=declared["backend"],
        scenario=declared["scenario"],
        execution_mode=config.execution.mode,
    )
    pack_report = validate_pack(config.pack_dir)
    if not pack_report.valid:
        issues = [
            RuntimeConfigIssue(
                f"pack.{issue.plugin_id or '$'}.{issue.field}",
                issue.message,
            )
            for issue in pack_report.issues
        ]
        raise RuntimeConfigError(issues)
    if not compatibility.valid:
        raise RuntimeConfigError(
            [
                RuntimeConfigIssue("components", issue)
                for issue in compatibility.issues
            ]
        )
    configuration_issues = _execution_compatibility_issues(
        config,
        compatibility.common_command_spaces,
    )
    if configuration_issues:
        raise RuntimeConfigError(
            [
                RuntimeConfigIssue("execution.safety.command_space", issue)
                for issue in configuration_issues
            ]
        )

    instances = {
        role: registry.instantiate(
            selection.plugin,
            _ROLE_KINDS[role],
            options=selection.options,
        )
        for role, selection in config.components.by_role().items()
    }
    _validate_instances(instances, declared)
    backend = instances["backend"]
    gateway = SafetyGateway(
        envelope=config.execution.safety,
        execution_mode=config.execution.mode,
        backend_capabilities=declared["backend"],
    )
    replay_level = (
        ReplayLevel.SIMULATOR_REPLAY
        if config.execution.mode is ExecutionMode.SIMULATION
        else ReplayLevel.SHADOW_HARDWARE_REPLAY
    )
    scenario_seed = config.components.scenario.options.get("seed")
    with TraceRecorder(
        config.trace.out,
        run_id=config.trace.run_id,
        target=config.trace.target,
        scenario=config.trace.scenario,
        replay_level=replay_level,
        seed=int(scenario_seed) if scenario_seed is not None else None,
    ) as recorder:
        _configure_capture(
            config.capture,
            policy=instances["policy"],
            recorder=recorder,
        )
        result = EpisodeRunner(
            policy=instances["policy"],
            embodiment=instances["embodiment"],
            backend=backend,
            scenario_provider=instances["scenario"],
            scheduler=ActionScheduler(
                n_action_steps=config.execution.n_action_steps
            ),
            safety_gateway=gateway,
            recorder=recorder,
        ).run()

    trace_report = validate_trace_file(config.trace.out)
    return {
        "format": "robotprobe.runtime.run_result.v0",
        "status": "completed" if result.success else "failed",
        "outcome": result.to_dict(),
        "trace_path": str(config.trace.out),
        "trace_valid": trace_report.valid,
        "event_count": trace_report.event_count,
        "components": {
            role: {
                "plugin": config.components.by_role()[role].plugin,
                "component_id": _instance_capabilities(instance).component_id,
            }
            for role, instance in instances.items()
        },
    }


def _validate_config_shape(data: dict[str, Any]) -> list[RuntimeConfigIssue]:
    issues: list[RuntimeConfigIssue] = []
    _unknown_fields(
        data,
        {"format", "pack", "components", "execution", "trace", "capture"},
        "",
        issues,
    )
    if data.get("format") != RUNTIME_CONFIG_FORMAT:
        issues.append(
            RuntimeConfigIssue("format", f"expected {RUNTIME_CONFIG_FORMAT}")
        )
    if not isinstance(data.get("pack"), str) or not data.get("pack", "").strip():
        issues.append(RuntimeConfigIssue("pack", "must be a non-empty path string"))

    components = data.get("components")
    if not isinstance(components, dict):
        issues.append(RuntimeConfigIssue("components", "must be a mapping"))
    else:
        _unknown_fields(components, set(_COMPONENT_ROLES), "components", issues)
        for role in _COMPONENT_ROLES:
            selection = components.get(role)
            path = f"components.{role}"
            if not isinstance(selection, dict):
                issues.append(RuntimeConfigIssue(path, "is required and must be a mapping"))
                continue
            _unknown_fields(selection, {"plugin", "options"}, path, issues)
            if not isinstance(selection.get("plugin"), str) or not selection.get(
                "plugin", ""
            ).strip():
                issues.append(
                    RuntimeConfigIssue(f"{path}.plugin", "must be a non-empty string")
                )
            if "options" in selection and not isinstance(selection["options"], dict):
                issues.append(
                    RuntimeConfigIssue(f"{path}.options", "must be a mapping")
                )

    execution = data.get("execution")
    if not isinstance(execution, dict):
        issues.append(RuntimeConfigIssue("execution", "is required and must be a mapping"))
    else:
        _unknown_fields(
            execution,
            {"mode", "n_action_steps", "safety"},
            "execution",
            issues,
        )
        if "mode" in execution and execution["mode"] not in {
            mode.value for mode in ExecutionMode
        }:
            issues.append(RuntimeConfigIssue("execution.mode", "unknown execution mode"))
        n_action_steps = execution.get("n_action_steps", 1)
        if (
            not isinstance(n_action_steps, int)
            or isinstance(n_action_steps, bool)
            or n_action_steps <= 0
        ):
            issues.append(
                RuntimeConfigIssue("execution.n_action_steps", "must be a positive integer")
            )
        safety = execution.get("safety")
        if not isinstance(safety, dict):
            issues.append(
                RuntimeConfigIssue(
                    "execution.safety",
                    "is required and must be a mapping",
                )
            )
        else:
            _validate_safety_shape(safety, issues)

    trace = data.get("trace")
    if not isinstance(trace, dict):
        issues.append(RuntimeConfigIssue("trace", "is required and must be a mapping"))
    else:
        _unknown_fields(
            trace,
            {"out", "run_id", "target", "scenario"},
            "trace",
            issues,
        )
        if not isinstance(trace.get("out"), str) or not trace.get("out", "").strip():
            issues.append(RuntimeConfigIssue("trace.out", "must be a non-empty path string"))
        for field_name in ("run_id", "target", "scenario"):
            if field_name in trace and (
                not isinstance(trace[field_name], str) or not trace[field_name].strip()
            ):
                issues.append(
                    RuntimeConfigIssue(
                        f"trace.{field_name}",
                        "must be a non-empty string",
                    )
                )
    if "capture" in data:
        capture = data["capture"]
        if not isinstance(capture, dict):
            issues.append(RuntimeConfigIssue("capture", "must be a mapping"))
        else:
            _unknown_fields(
                capture,
                {"profile", "storage", "output_dir", "max_bytes_per_run"},
                "capture",
                issues,
            )
            if not isinstance(capture.get("profile"), str) or not capture.get(
                "profile", ""
            ).strip():
                issues.append(
                    RuntimeConfigIssue(
                        "capture.profile",
                        "must be a non-empty path string",
                    )
                )
            storage = capture.get("storage")
            if storage is not None and storage not in {
                "summary",
                "npy",
                "safetensors",
            }:
                issues.append(
                    RuntimeConfigIssue(
                        "capture.storage",
                        "must be summary, npy, or safetensors",
                    )
                )
            output_dir = capture.get("output_dir")
            if output_dir is not None and (
                not isinstance(output_dir, str) or not output_dir.strip()
            ):
                issues.append(
                    RuntimeConfigIssue(
                        "capture.output_dir",
                        "must be a non-empty path string",
                    )
                )
            if storage in {"npy", "safetensors"} and output_dir is None:
                issues.append(
                    RuntimeConfigIssue(
                        "capture.output_dir",
                        "is required for full activation storage",
                    )
                )
            max_bytes = capture.get("max_bytes_per_run")
            if max_bytes is not None and (
                not isinstance(max_bytes, int)
                or isinstance(max_bytes, bool)
                or max_bytes <= 0
            ):
                issues.append(
                    RuntimeConfigIssue(
                        "capture.max_bytes_per_run",
                        "must be a positive integer",
                    )
                )
    return issues


def _load_capture_config(
    value: Any,
    config_path: Path,
) -> RuntimeCaptureConfig | None:
    if value is None:
        return None
    capture = dict(value)
    profile_value = Path(str(capture["profile"]))
    profile = (
        profile_value.resolve()
        if profile_value.is_absolute()
        else (config_path.parent / profile_value).resolve()
    )
    if not profile.is_file():
        raise RuntimeConfigError(
            [
                RuntimeConfigIssue(
                    "capture.profile",
                    f"profile does not exist: {profile}",
                )
            ]
        )
    output_value = capture.get("output_dir")
    output_dir = None
    if output_value is not None:
        candidate = Path(str(output_value))
        output_dir = (
            candidate.resolve()
            if candidate.is_absolute()
            else (config_path.parent / candidate).resolve()
        )
    return RuntimeCaptureConfig(
        profile=profile,
        storage=(
            str(capture["storage"]) if capture.get("storage") is not None else None
        ),
        output_dir=output_dir,
        max_bytes_per_run=(
            int(capture["max_bytes_per_run"])
            if capture.get("max_bytes_per_run") is not None
            else None
        ),
    )


def _configure_capture(
    capture: RuntimeCaptureConfig | None,
    *,
    policy: Any,
    recorder: TraceRecorder,
) -> None:
    if capture is None:
        return
    if not isinstance(policy, ConfigurableCapturePolicyAdapter):
        raise RuntimeConfigError(
            [
                RuntimeConfigIssue(
                    "capture",
                    "selected policy does not support configurable capture",
                )
            ]
        )
    try:
        policy.configure_capture(
            profile_path=capture.profile,
            recorder=recorder,
            storage=capture.storage,
            output_dir=capture.output_dir,
            max_bytes_per_run=capture.max_bytes_per_run,
        )
    except Exception as error:
        raise RuntimeConfigError(
            [
                RuntimeConfigIssue(
                    "capture",
                    f"capture configuration failed: {type(error).__name__}: {error}",
                )
            ]
        ) from error


def _validate_safety_shape(
    safety: dict[str, Any],
    issues: list[RuntimeConfigIssue],
) -> None:
    _unknown_fields(
        safety,
        {
            "command_space",
            "max_abs",
            "workspace_min",
            "workspace_max",
            "max_observation_age_ms",
        },
        "execution.safety",
        issues,
    )
    if not isinstance(safety.get("command_space"), str) or not safety.get(
        "command_space", ""
    ).strip():
        issues.append(
            RuntimeConfigIssue(
                "execution.safety.command_space",
                "must be a non-empty string",
            )
        )
    if not _numeric_list(safety.get("max_abs")):
        issues.append(
            RuntimeConfigIssue(
                "execution.safety.max_abs",
                "must be a non-empty list of numbers",
            )
        )
    for field_name in ("workspace_min", "workspace_max"):
        if field_name in safety and not _numeric_list(
            safety[field_name],
            expected_length=3,
        ):
            issues.append(
                RuntimeConfigIssue(
                    f"execution.safety.{field_name}",
                    "must be a three-value numeric list",
                )
            )
    if ("workspace_min" in safety) != ("workspace_max" in safety):
        issues.append(
            RuntimeConfigIssue(
                "execution.safety.workspace_min",
                "workspace_min and workspace_max must be configured together",
            )
        )
    age = safety.get("max_observation_age_ms", 250.0)
    if (
        not isinstance(age, (int, float))
        or isinstance(age, bool)
        or not math.isfinite(float(age))
        or float(age) <= 0
    ):
        issues.append(
            RuntimeConfigIssue(
                "execution.safety.max_observation_age_ms",
                "must be a positive finite number",
            )
        )


def _unknown_fields(
    mapping: Mapping[str, Any],
    allowed: set[str],
    prefix: str,
    issues: list[RuntimeConfigIssue],
) -> None:
    for field_name in sorted(set(mapping) - allowed):
        path = f"{prefix}.{field_name}" if prefix else field_name
        issues.append(RuntimeConfigIssue(path, "unknown field"))


def _numeric_list(value: Any, *, expected_length: int | None = None) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if expected_length is not None and len(value) != expected_length:
        return False
    return all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    )


def _optional_triplet(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return tuple(float(item) for item in value)


def _selected_capabilities(
    config: RuntimeRunConfig,
    registry: RuntimePluginRegistry,
) -> dict[str, RuntimeComponentCapabilities]:
    return {
        role: registry.capabilities(selection.plugin, _ROLE_KINDS[role])
        for role, selection in config.components.by_role().items()
    }


def _execution_compatibility_issues(
    config: RuntimeRunConfig,
    common_command_spaces: tuple[str, ...],
) -> list[str]:
    command_space = config.execution.safety.command_space
    if command_space not in common_command_spaces:
        return [
            f"safety command space {command_space!r} is not shared by the selected components"
        ]
    return []


def _instance_capabilities(instance: Any) -> RuntimeComponentCapabilities:
    capabilities = getattr(instance, "capabilities", None)
    if isinstance(capabilities, RuntimeComponentCapabilities):
        return capabilities
    if isinstance(capabilities, dict):
        return RuntimeComponentCapabilities.from_dict(capabilities)
    raise TypeError("runtime instance must expose declared capabilities")


def _validate_instances(
    instances: Mapping[str, Any],
    declared: Mapping[str, RuntimeComponentCapabilities],
) -> None:
    protocols = {
        "policy": PolicyAdapter,
        "embodiment": EmbodimentAdapter,
        "backend": RuntimeBackend,
        "scenario": ScenarioProvider,
    }
    issues: list[RuntimeConfigIssue] = []
    for role, instance in instances.items():
        if not isinstance(instance, protocols[role]):
            issues.append(
                RuntimeConfigIssue(
                    f"components.{role}",
                    f"instance does not implement {protocols[role].__name__}",
                )
            )
            continue
        actual = _instance_capabilities(instance)
        if actual != declared[role]:
            actual_fields = actual.to_dict()
            declared_fields = declared[role].to_dict()
            changed = sorted(
                field_name
                for field_name in declared_fields
                if actual_fields[field_name] != declared_fields[field_name]
            )
            issues.append(
                RuntimeConfigIssue(
                    f"components.{role}.capabilities",
                    f"runtime capabilities differ from pack declaration: {changed}",
                )
            )
    if issues:
        raise RuntimeConfigError(issues)


def runtime_result_json(path: str | Path) -> str:
    return json.dumps(run_runtime_config(path), indent=2, sort_keys=True) + "\n"
