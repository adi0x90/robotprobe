#!/usr/bin/env python3
"""Bounded simulation experiment: matched PI0.5 runs, one action fault, and a guard.

This is an experiment driver over the public runtime APIs, not a new runtime plugin.
The guard is a demonstration envelope in normalized LIBERO action units, not a
calibrated physical safety limit. Task completion and integration checks are separate.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robotprobe.core import ReplayLevel  # noqa: E402
from robotprobe.mech.collect import persist_capture_pair  # noqa: E402
from robotprobe.mech.diff import diff_activations  # noqa: E402
from robotprobe.mech.report import build_mech_report, render_mech_report_markdown  # noqa: E402
from robotprobe.record import TraceRecorder  # noqa: E402
from robotprobe.report import build_incident_report  # noqa: E402
from robotprobe.runtime.capabilities import RuntimeComponentKind  # noqa: E402
from robotprobe.runtime.config import load_runtime_config  # noqa: E402
from robotprobe.runtime.evidence import analyze_runtime_trace, create_runtime_evidence_bundle  # noqa: E402
from robotprobe.runtime.orchestrator import EpisodeRunner  # noqa: E402
from robotprobe.runtime.registry import RuntimePluginRegistry  # noqa: E402
from robotprobe.runtime.safety import SafetyEnvelope, SafetyGateway  # noqa: E402
from robotprobe.runtime.scheduler import ActionScheduler  # noqa: E402
from robotprobe.runtime.types import ExecutionMode, ScheduledAction  # noqa: E402
from robotprobe.summarize import read_trace, summarize_trace_pair  # noqa: E402


@dataclasses.dataclass(frozen=True)
class RunSettings:
    out: Path
    runtime_config: Path = REPO_ROOT / "examples/packs/franka-pi05-runtime/configs/pi05-libero.yaml"
    max_steps: int = 10
    seed: int = 7
    policy_seed: int = 7
    guard_limit: float = 0.5
    fault_magnitude: float = 0.9
    capture_max_bytes: int = 64 * 1024 * 1024
    max_observation_age_ms: float = 120_000.0
    tolerance: float = 1e-6
    save_frames: bool = False
    normalize_libero_actions: bool = False
    checkpoint: str | None = None
    revision: str | None = None
    tokenizer_checkpoint: str | None = None
    tokenizer_revision: str | None = None
    local_files_only: bool = False

    def __post_init__(self):
        for field in ("max_steps", "capture_max_bytes"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.max_steps > 1000:
            raise ValueError("max_steps must be bounded at 1000 or fewer")
        if not 0 < self.guard_limit < self.fault_magnitude <= 1:
            raise ValueError("require 0 < guard_limit < fault_magnitude <= 1")
        if not np.isfinite(self.max_observation_age_ms) or self.max_observation_age_ms <= 0:
            raise ValueError("max_observation_age_ms must be positive and finite")
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be nonnegative and finite")


@dataclasses.dataclass(frozen=True)
class FaultSpec:
    dimension: int
    scale: float
    step: int
    guard_limit: float

    def apply(self, value: np.ndarray) -> np.ndarray:
        result = np.array(value, dtype=float, copy=True)
        result[self.dimension] *= self.scale
        if not np.isfinite(result).all() or np.any(np.abs(result) > 1.0):
            raise ValueError("fault would exceed LIBERO normalized action bounds; no command sent")
        return result


def select_fault(value, *, guard_limit: float, fault_magnitude: float) -> FaultSpec:
    action = np.asarray(value, dtype=float)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("fault selection needs a finite seven-dimensional action")
    if not 0 < guard_limit < fault_magnitude <= 1:
        raise ValueError("invalid declared fault and guard bounds")
    candidates = [i for i in range(3) if 1e-8 < abs(action[i]) < guard_limit]
    if not candidates:
        raise ValueError("no nonzero translation dimension fits the declared guard; choose another task/seed")
    dimension = max(candidates, key=lambda i: abs(action[i]))
    return FaultSpec(dimension, fault_magnitude / abs(action[dimension]), 0, guard_limit)


class BoundaryScheduler(ActionScheduler):
    """Observe the real queued model action and mutate one command before safety."""

    def __init__(self, *, recorder, n_action_steps, fault=None, normalize_libero_actions=False):
        super().__init__(n_action_steps=n_action_steps)
        self.recorder, self.fault = recorder, fault
        self.normalize_libero_actions = normalize_libero_actions
        self.step = 0

    def reset(self):
        super().reset()
        self.step = 0

    def next_action(self) -> ScheduledAction:
        raw = super().next_action()
        self.recorder.event(self.step, "raw_action", "policy_forward", **raw.to_trace_fields())
        normalized = raw
        if self.normalize_libero_actions:
            normalized = dataclasses.replace(raw, value=np.clip(raw.value, -1.0, 1.0))
            self.recorder.event(
                self.step, "sim_state", "action_postprocess", boundary="libero_action_normalization",
                command_id=raw.command_id, before=raw.value.tolist(), value=normalized.value.tolist(),
                clipped_dimensions=np.flatnonzero(raw.value != normalized.value).tolist(),
                declared_bounds=[-1.0, 1.0], normalization="explicit_simulation_np_clip",
            )
        changed = self.fault is not None and self.step == self.fault.step
        result = dataclasses.replace(normalized, value=self.fault.apply(normalized.value)) if changed else normalized
        self.recorder.event(
            self.step, "postprocessed_action", "action_postprocess",
            **result.to_trace_fields(), intervention_applied=changed,
            mutation=dataclasses.asdict(self.fault) if changed else None,
        )
        self.step += 1
        return result


class ObservedBackend:
    def __init__(self, backend, recorder):
        self.backend, self.recorder, self.step = backend, recorder, 0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def execute(self, action):
        self.recorder.event(
            self.step, "sim_state", "backend_execution", boundary="backend_received_action",
            command_id=action.command_id, value=action.value.tolist(),
        )
        receipt = self.backend.execute(action)
        self.step += 1
        return receipt


class FrameEmbodiment:
    def __init__(self, embodiment, frame_dir):
        self.embodiment, self.frame_dir, self.step = embodiment, frame_dir, 0

    def __getattr__(self, name):
        return getattr(self.embodiment, name)

    def canonicalize_observation(self, raw):
        observation = self.embodiment.canonicalize_observation(raw)
        if self.frame_dir is not None:
            from PIL import Image
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            image = np.asarray(observation.images["agent"])
            Image.fromarray(image).save(self.frame_dir / f"{self.step:05d}.png")
        self.step += 1
        return observation


def _seed_runtime(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("real PI0.5 experiment requires CUDA; unavailable is not a pass")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _instances(config):
    registry = RuntimePluginRegistry.from_pack(config.pack_dir)
    return {
        role: registry.instantiate(selection.plugin, RuntimeComponentKind(role), options=selection.options)
        for role, selection in config.components.by_role().items()
    }


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def execute_variant(
    settings: RunSettings, name: str, *, fault: FaultSpec | None,
    instance_factory: Callable | None = None, seed_runtime: Callable = _seed_runtime,
    capture: bool = True,
) -> dict[str, Any]:
    directory = settings.out / "runs" / name
    directory.mkdir(parents=True, exist_ok=False)
    source = settings.runtime_config.resolve()
    data = yaml.safe_load(source.read_text())
    if data["execution"]["mode"] != "simulation":
        raise ValueError("this experiment permits simulation only")
    data["pack"] = str((source.parent / data["pack"]).resolve())
    policy = data["components"]["policy"]["options"]
    for key in ("checkpoint", "revision", "tokenizer_checkpoint", "tokenizer_revision"):
        override = getattr(settings, key)
        if override is not None:
            policy[key] = override
    policy["local_files_only"] = settings.local_files_only
    data["components"]["scenario"]["options"].update(seed=settings.seed, max_steps=settings.max_steps)
    data["components"]["backend"]["options"]["episode_length"] = max(280, settings.max_steps)
    data["execution"]["safety"]["max_abs"] = [1.0] * 7
    data["execution"]["safety"]["max_observation_age_ms"] = settings.max_observation_age_ms
    if name == "guarded":
        if fault is None:
            raise ValueError("guarded variant requires a declared fault")
        data["execution"]["safety"]["max_abs"][fault.dimension] = fault.guard_limit
    trace = directory / f"{name}.jsonl"
    data["trace"].update(out=str(trace.resolve()), run_id=f"pi05-e2e:{name}", target=policy["checkpoint"])
    profile = directory / "capture-profile.yaml"
    profile.write_text(yaml.safe_dump({
        "format": "robotprobe.runtime.capture_profile.v0", "id": "pi05-e2e-first-denoise-call",
        "storage": "npy", "max_bytes_per_run": settings.capture_max_bytes,
        "every_n_policy_calls": 1, "max_module_calls": 1,
        "semantic_roles": ["action_input_projection", "action_output_projection"],
        "user_selectors": [], "claim_boundary": {
            "summaries_are_observational": True, "no_causal_claims": True,
        },
    }, sort_keys=False))
    data["capture"] = {
        "profile": str(profile.resolve()), "storage": "npy",
        "output_dir": str((directory / "activations").resolve()), "max_bytes_per_run": settings.capture_max_bytes,
    }
    config_path = directory / "runtime.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))
    config = load_runtime_config(config_path)
    if instance_factory is None and (
        config.components.policy.plugin != "robotprobe.policies.pi05_lerobot"
        or config.components.backend.plugin != "robotprobe.backends.libero_franka"
    ):
        raise ValueError("real validation requires the PI0.5 policy and LIBERO backend plugins")
    seed_runtime(settings.policy_seed)
    instances = (instance_factory or _instances)(config)
    if instances["backend"].capabilities.is_hardware:
        raise ValueError("hardware backends are forbidden in this experiment")
    with TraceRecorder(
        trace, run_id=config.trace.run_id, target=config.trace.target, scenario=config.trace.scenario,
        replay_level=ReplayLevel.SIMULATOR_REPLAY, seed=settings.seed,
        baseline_run_id="pi05-e2e:clean" if name != "clean" else None,
        perturbation_group_id="pi05-e2e:action-scale",
    ) as recorder:
        if capture:
            instances["policy"].configure_capture(
                profile_path=profile, recorder=recorder, storage="npy",
                output_dir=directory / "activations", max_bytes_per_run=settings.capture_max_bytes,
            )
        runner = EpisodeRunner(
            policy=instances["policy"],
            embodiment=FrameEmbodiment(instances["embodiment"], directory / "frames" if settings.save_frames else None),
            backend=ObservedBackend(instances["backend"], recorder),
            scenario_provider=instances["scenario"],
            scheduler=BoundaryScheduler(
                recorder=recorder, n_action_steps=config.execution.n_action_steps,
                fault=fault if name in {"fault", "guarded"} else None,
                normalize_libero_actions=settings.normalize_libero_actions,
            ),
            safety_gateway=SafetyGateway(
                envelope=SafetyEnvelope(**dataclasses.asdict(config.execution.safety)),
                execution_mode=ExecutionMode.SIMULATION,
                backend_capabilities=instances["backend"].capabilities,
            ), recorder=recorder,
        )
        result = runner.run().to_dict()
    events = read_trace(trace)
    evidence = analyze_runtime_trace(trace, config_path=config_path)
    _json(directory / "run-result.json", result)
    create_runtime_evidence_bundle(
        trace_path=trace, config_path=config_path, out_dir=directory / "evidence",
        baseline_path=settings.out / "runs/clean/clean.jsonl" if name != "clean" else None,
    )
    return {"result": result, "events": events, "evidence": evidence, "trace": trace, "config": data}


def _events(run, kind):
    return [event for event in run["events"] if event["event_type"] == kind]


def _same_values(left, right, tolerance):
    if len(left) != len(right) or not left:
        return False
    return all(
        a.get("step") == b.get("step") and np.allclose(a["value"], b["value"], atol=tolerance, rtol=0)
        for a, b in zip(left, right)
    )


def _initial_input(run):
    observations = _events(run, "observation")
    if not observations:
        return None
    return {key: observations[0].get(key) for key in (
        "images", "eef_position", "eef_quaternion", "gripper_qpos", "instruction",
    )}


def _normalization_valid(run, enabled):
    raw = _events(run, "raw_action")
    events = [event for event in run["events"] if event.get("boundary") == "libero_action_normalization"]
    if not enabled:
        return not events
    if not raw or len(raw) != len(events):
        return False
    for before, event in zip(raw, events):
        value = np.asarray(before["value"])
        expected = np.clip(value, -1.0, 1.0)
        if (
            before["step"] != event["step"] or event.get("before") != before["value"]
            or event.get("declared_bounds") != [-1.0, 1.0]
            or not np.array_equal(expected, event["value"])
            or event.get("clipped_dimensions") != np.flatnonzero(value != expected).tolist()
        ):
            return False
    return True


def validate_runs(runs, *, tolerance, fault, normalize_libero_actions=False):
    clean = runs["clean"]
    raw_first = _events(clean, "raw_action")[:1]
    post_fault = _events(runs["fault"], "postprocessed_action")[:1]
    received = [e for e in runs["fault"]["events"] if e.get("boundary") == "backend_received_action"][:1]
    guarded = runs["guarded"]
    decisions = _events(guarded, "safety_decision")
    return {
        "all_runtime_integrity_passed": all(r["evidence"]["runtime_integrity"]["status"] == "passed" for r in runs.values()),
        "initial_observations_match": _initial_input(clean) is not None and all(_initial_input(r) == _initial_input(clean) for r in runs.values()),
        "clean_repeat_raw_actions_match": _same_values(_events(clean, "raw_action"), _events(runs["clean-repeat"], "raw_action"), tolerance),
        "clean_repeat_model_chunks_match": _same_values(_events(clean, "action_chunk"), _events(runs["clean-repeat"], "action_chunk"), tolerance),
        "pre_feedback_raw_actions_match": all(_same_values(raw_first, _events(r, "raw_action")[:1], tolerance) for r in runs.values()),
        "pre_feedback_model_chunks_match": all(_same_values(_events(clean, "action_chunk")[:1], _events(r, "action_chunk")[:1], tolerance) for r in runs.values()),
        "fault_changes_command": bool(raw_first and post_fault) and not _same_values(raw_first, post_fault, tolerance),
        "fault_matches_declared_scale": bool(raw_first and post_fault) and bool(np.allclose(fault.apply(np.clip(raw_first[0]["value"], -1.0, 1.0) if normalize_libero_actions else np.asarray(raw_first[0]["value"])), post_fault[0]["value"], atol=tolerance, rtol=0)),
        "normalization_exactly_reported": all(_normalization_valid(run, normalize_libero_actions) for run in runs.values()),
        "fault_reaches_backend_unchanged": _same_values(post_fault, received, tolerance),
        "fault_and_guarded_commands_match": _same_values(post_fault, _events(guarded, "postprocessed_action")[:1], tolerance),
        "fault_has_applied_receipt": any(e.get("command_applied") is True for e in _events(runs["fault"], "execution_receipt")),
        "guard_rejects_action_limit": bool(decisions) and decisions[0].get("accepted") is False and decisions[0].get("reason") == "action_limit_exceeded",
        "guard_has_no_execution": not _events(guarded, "execution_receipt") and not any(e.get("boundary") == "backend_received_action" for e in guarded["events"]),
        "clean_first_command_inside_guard": bool(raw_first) and abs(raw_first[0]["value"][fault.dimension]) < fault.guard_limit,
        "no_capture_budget_exhaustion": not any(e.get("event_type") == "activation_capture_dropped" and e.get("reason") == "capture_budget_exhausted" for r in runs.values() for e in r["events"]),
    }


def select_initial_captures(events, run_dir):
    """Select an explicit pre-feedback coordinate; never silently flatten calls."""
    selected, receipt, seen = [], [], set()
    for event in events:
        if event.get("event_type") != "model_activation" or event.get("step") != 0:
            continue
        if event.get("policy_call_id") != "policy:000000" or event.get("module_call_index") != 0:
            continue
        module = event["module"]
        if module in seen:
            raise ValueError(f"ambiguous first-call activation for {module}")
        seen.add(module)
        ref = event.get("activation_ref", "")
        if not ref or ref.startswith("summary:") or event.get("capture", {}).get("storage_mode") != "npy":
            raise ValueError("paired MechProbe analysis requires full stored npy captures")
        path = Path(ref)
        if not path.is_absolute():
            path = Path(run_dir) / path
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != event["activation_shape"] or str(array.dtype) != event["activation_dtype"] or not np.isfinite(array).all():
            raise ValueError(f"invalid stored activation for {module}")
        expected_hash = event.get("capture", {}).get("tensor_summary", {}).get("sha256")
        actual_hash = "sha256:" + hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError(f"stored activation checksum disagrees with runtime trace for {module}")
        selected.append({"module": module, "step": 0, "activation": array, "source_dtype": event["activation_dtype"]})
        receipt.append({
            **{key: event.get(key) for key in ("module", "step", "policy_call_id", "module_call_index", "denoise_step", "phase", "observation_id", "action_chunk_id", "model_ref")},
            "source_activation_ref": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    if not selected:
        raise ValueError("no full first-policy-call, first-module-call captures were found")
    return selected, receipt


def _write_pairs(settings, runs, manifest):
    clean_captures, clean_selection = select_initial_captures(runs["clean"]["events"], settings.out / "runs/clean")
    modules = sorted(item["module"] for item in clean_captures)
    if len(modules) != 2 or not all(any(module.endswith(suffix) for module in modules) for suffix in ("action_in_proj", "action_out_proj")):
        raise ValueError("first-call capture must cover both configured action input and output projections")
    config = {
        "format": "robotprobe.mech.capture_config.v0", "id": "pi05-e2e-first-call-negative-control",
        "model_access": "white_box", "layers": [{"name": module} for module in modules],
        "storage": {"format": "npy", "dtype": "float32", "max_bytes_per_run": settings.capture_max_bytes},
        "claim_boundary": {"no_feature_labels": True, "no_causal_claims": True},
    }
    results = {}
    for name in ("clean-repeat", "fault", "guarded"):
        captures, selection = select_initial_captures(runs[name]["events"], settings.out / "runs" / name)
        if sorted(item["module"] for item in captures) != modules:
            raise ValueError(f"selected module coverage differs for {name}")
        clean_identity = {item["module"]: item["model_ref"] for item in clean_selection}
        if any(item["model_ref"] != clean_identity[item["module"]] for item in selection):
            raise ValueError(f"model/capture provenance differs for {name}")
        pair_dir = settings.out / "comparisons" / name
        mech = pair_dir / "mech"
        summary = summarize_trace_pair(
            runs["clean"]["trace"], runs[name]["trace"],
            intervention_manifest_ref=manifest if name != "clean-repeat" else None,
        )
        summary["experiment_boundary"] = "Observed simulation comparison; first-call activations only. No causal feature claim."
        _json(pair_dir / "summary.json", summary)
        (pair_dir / "report.md").write_text(build_incident_report(summary, title=f"PI0.5 LIBERO: clean vs {name}"))
        policy_options = runs["clean"]["config"]["components"]["policy"]["options"]
        persist_capture_pair(
            capture_pair={"target": "pi05-runtime-first-call", "clean": clean_captures, "perturbed": captures},
            capture_config=config, out_dir=mech, incident_id=f"pi05-e2e:{name}",
            clean_run_id="pi05-e2e:clean", perturbed_run_id=f"pi05-e2e:{name}",
            model={
                "name": policy_options["checkpoint"], "revision": policy_options.get("revision"),
                "tokenizer_name": policy_options.get("tokenizer_checkpoint"), "tokenizer_revision": policy_options.get("tokenizer_revision"),
                "adapter": "robotprobe.pi05.runtime", "pair_selection": "env_step=0;policy_call=0;module_call=0",
            }, target=policy_options["checkpoint"], scenario=runs["clean"]["config"]["trace"]["scenario"],
            replay_level=ReplayLevel.SIMULATOR_REPLAY.value,
            external_first_divergence_step=None if name == "clean-repeat" else 0,
            negative_control=True, synthetic=False,
        )
        _json(mech / "selection.json", {"clean": clean_selection, "perturbed": selection})
        _json(mech / "capture-config.json", config)
        diff = diff_activations(mech_dir=mech, out_path=mech / "activation_diff.json", threshold=settings.tolerance)
        report = build_mech_report(summary=summary, activation_diff=diff)
        _json(mech / "mech_report.json", report)
        (mech / "mech_report.md").write_text(render_mech_report_markdown(report))
        results[name] = {
            "evidence_eligible": report["evidence_eligible"], "evidence_level": report["evidence_level"],
            "negative_control_stable": report["negative_control_stable"],
            "model_root_cause_claim_allowed": report["model_root_cause_claim_allowed"],
            "modules": modules, "report": str(mech / "mech_report.json"),
        }
    return results


def write_intervention_manifest(path, fault):
    # RobotProbe's lightweight intervention reader requires indented list items.
    path.write_text(
        "id: PI05.E2E.ACTION_SCALE.001\n"
        "name: pi05_action_scale\n"
        "release_classification: sim_only\n"
        "target_layers:\n  - action_postprocess\n"
        "contract: DemonstrationActionEnvelope\n"
        "window:\n  start_step: 0\n  end_step: 0\n"
        f"mutation:\n  dimension: {fault.dimension}\n  scale: {fault.scale}\n"
        f"  guard_limit: {fault.guard_limit}\n"
        "safety:\n  sim_only: true\n  hardware_allowed: false\n"
    )


def run_experiment(settings):
    if settings.out.exists() and any(settings.out.iterdir()):
        raise ValueError("output directory must be absent or empty")
    settings.out.mkdir(parents=True, exist_ok=True)
    _json(settings.out / "settings.json", {key: str(value) if isinstance(value, Path) else value for key, value in dataclasses.asdict(settings).items()})
    runs = {}
    fault = None
    for name in ("clean", "clean-repeat", "fault", "guarded"):
        print(f"Running {name} (max_steps={settings.max_steps})", flush=True)
        runs[name] = execute_variant(settings, name, fault=fault)
        if name == "clean":
            if not any(event.get("command_applied") is True for event in _events(runs[name], "execution_receipt")):
                raise RuntimeError(f"clean run had no applied backend receipt: {runs[name]['result']['reason']}")
            first = _events(runs[name], "postprocessed_action")
            if not first:
                raise RuntimeError("clean run produced no raw action")
            fault = select_fault(first[0]["value"], guard_limit=settings.guard_limit, fault_magnitude=settings.fault_magnitude)
            _json(settings.out / "fault.json", dataclasses.asdict(fault))
        print(f"{name}: integrity={runs[name]['evidence']['runtime_integrity']['status']} task_success={runs[name]['result']['success']} reason={runs[name]['result']['reason']}", flush=True)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    assert fault is not None
    manifest = settings.out / "action-scale.yaml"
    write_intervention_manifest(manifest, fault)
    checks = validate_runs(runs, tolerance=settings.tolerance, fault=fault, normalize_libero_actions=settings.normalize_libero_actions)
    pairs = _write_pairs(settings, runs, manifest)
    checks["paired_mech_negative_controls_pass"] = all(
        pair["negative_control_stable"] and pair["evidence_eligible"] and pair["evidence_level"] <= 1
        and not pair["model_root_cause_claim_allowed"] for pair in pairs.values()
    )
    result = {
        "format": "robotprobe.pi05_libero_e2e.v0", "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "fault": dataclasses.asdict(fault), "mech_pairs": pairs,
        "action_normalization": {
            "enabled": settings.normalize_libero_actions,
            "method": "np.clip(-1,1) before fault injection" if settings.normalize_libero_actions else "disabled",
            "scope": "explicit simulation command adapter; original policy actions retained",
            "runs": {
                name: {
                    "normalization_events": sum(event.get("boundary") == "libero_action_normalization" for event in run["events"]),
                    "commands_clipped": sum(event.get("boundary") == "libero_action_normalization" and bool(event.get("clipped_dimensions")) for event in run["events"]),
                } for name, run in runs.items()
            },
        },
        "runs": {name: {"outcome": run["result"], "runtime_integrity": run["evidence"]["runtime_integrity"]["status"]} for name, run in runs.items()},
        "claim_boundary": "One seeded simulator experiment; optional normalized-action clipping is an explicit simulation adapter and the action guard is a demonstration envelope, neither a physical safety certification. Bounded integration completion does not require task success. MechProbe compares only matched first-call activations before environment feedback.",
    }
    _json(settings.out / "validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, help="JSON RunSettings mapping for a short reproducible launch command")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    for name in ("max-steps", "seed", "policy-seed", "capture-max-bytes"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("guard-limit", "fault-magnitude", "max-observation-age-ms", "tolerance"):
        parser.add_argument(f"--{name}", type=float)
    for name in ("checkpoint", "revision", "tokenizer-checkpoint", "tokenizer-revision"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--save-frames", action="store_true", default=None)
    parser.add_argument("--normalize-libero-actions", action="store_true", default=None)
    parser.add_argument("--local-files-only", action="store_true", default=None)
    args = vars(parser.parse_args(argv))
    config_file = args.pop("config_file")
    values = json.loads(config_file.read_text()) if config_file else {}
    values.update({key: value for key, value in args.items() if value is not None})
    if "out" not in values:
        parser.error("--out or config-file out is required")
    for name in ("out", "runtime_config"):
        if name in values:
            values[name] = Path(values[name]).expanduser().resolve()
    settings = RunSettings(**values)
    if settings.out.exists() and any(settings.out.iterdir()):
        raise ValueError("output directory must be absent or empty; existing evidence was not changed")
    try:
        result = run_experiment(settings)
    except Exception as error:
        result = {"format": "robotprobe.pi05_libero_e2e.v0", "status": "failed", "error_type": type(error).__name__, "error": str(error)}
        if settings.out.exists():
            _json(settings.out / "failure.json", result)
        print(json.dumps(result, indent=2), file=sys.stderr)
        raise
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
