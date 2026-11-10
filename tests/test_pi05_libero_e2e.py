import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from robotprobe.integrations.franka.deterministic import DeterministicFrankaBackend, DeterministicReachPolicy
from robotprobe.integrations.franka.embodiment import FrankaEmbodimentAdapter
from robotprobe.integrations.franka.scenarios import DeterministicReachScenario
from robotprobe.integrations.lerobot.pi05_probes import load_pi05_capture_settings
from robotprobe.core import load_intervention_manifest
from robotprobe.runtime.capture import TensorSummary
from robotprobe.runtime.scheduler import ActionScheduler
from robotprobe.runtime.types import ProposedActionChunk
from scripts.run_pi05_libero_e2e import (
    FaultSpec,
    RunSettings,
    execute_variant,
    select_fault,
    select_initial_captures,
    validate_runs,
    main,
    write_intervention_manifest,
    _write_pairs,
    BoundaryScheduler,
    _normalization_valid,
    run_experiment,
)


class Pi05EndToEndRunnerTests(unittest.TestCase):
    def test_opt_in_libero_normalization_preserves_raw_and_records_clipping_before_fault(self):
        class Recorder:
            def __init__(self):
                self.events = []

            def event(self, step, event_type, layer, **fields):
                self.events.append({"step": step, "event_type": event_type, "layer": layer, **fields})

        recorder = Recorder()
        scheduler = BoundaryScheduler(
            recorder=recorder, n_action_steps=1, fault=FaultSpec(0, 9, 0, 0.5),
            normalize_libero_actions=True,
        )
        chunk = ProposedActionChunk(
            chunk_id="chunk:0", policy_call_id="policy:0", source_observation_id="observation:0",
            source_observation_captured_at_ns=0,
            command_space="franka.cartesian_delta_gripper.v0",
            values=[[0.1, 0, 0, 0, 0, 0, -1.0034]],
        )
        scheduler.load(chunk)
        result = scheduler.next_action()
        self.assertAlmostEqual(result.value[0], 0.9)
        self.assertEqual(result.value[6], -1.0)
        raw, normalized, post = recorder.events
        self.assertEqual(raw["value"][6], -1.0034)
        self.assertEqual(normalized["boundary"], "libero_action_normalization")
        self.assertEqual(normalized["before"], raw["value"])
        self.assertEqual(normalized["value"][6], -1.0)
        self.assertEqual(normalized["clipped_dimensions"], [6])
        self.assertEqual(normalized["declared_bounds"], [-1.0, 1.0])
        self.assertEqual(post["value"], result.value.tolist())
        self.assertTrue(_normalization_valid({"events": recorder.events}, True))
        normalized["clipped_dimensions"] = []
        self.assertFalse(_normalization_valid({"events": recorder.events}, True))
        self.assertEqual(chunk.values[0, 6], -1.0034)
        default_scheduler = BoundaryScheduler(recorder=Recorder(), n_action_steps=1)
        default_scheduler.load(chunk)
        self.assertEqual(default_scheduler.next_action().value[6], -1.0034)
        self.assertTrue(issubclass(BoundaryScheduler, ActionScheduler))

    def test_clean_rejection_stops_before_remaining_model_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = RunSettings(out=Path(directory))
            failed_clean = {"events": [], "result": {"reason": "safety_rejected:action_limit_exceeded"}}
            with patch("scripts.run_pi05_libero_e2e.execute_variant", return_value=failed_clean) as execute:
                with self.assertRaisesRegex(RuntimeError, "no applied backend receipt"):
                    run_experiment(settings)
            self.assertEqual(execute.call_count, 1)

    def test_generated_manifest_is_accepted_by_the_actual_lightweight_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fault.yaml"
            write_intervention_manifest(path, FaultSpec(0, 9, 0, 0.5))
            manifest = load_intervention_manifest(path)
            self.assertEqual(manifest.target_layers, ["action_postprocess"])
            self.assertEqual(manifest.mutation["scale"], 9)
            self.assertIs(manifest.safety["hardware_allowed"], False)

    def test_fault_selection_is_declared_bounded_and_changes_a_nonzero_translation(self):
        spec = select_fault([0.0, -0.1, 0.7, 0, 0, 0, 1], guard_limit=0.5, fault_magnitude=0.9)
        self.assertEqual(spec.dimension, 1)
        self.assertAlmostEqual(spec.scale, 9.0)
        self.assertAlmostEqual(spec.apply(np.array([0, -0.1, 0.7, 0, 0, 0, 1]))[1], -0.9)
        with self.assertRaisesRegex(ValueError, "nonzero translation"):
            select_fault([0, 0, 0, 0, 0, 0, 1], guard_limit=0.5, fault_magnitude=0.9)

    def test_bounded_mutation_keeps_raw_action_and_gateway_prevents_backend_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = RunSettings(out=Path(directory), max_steps=2, save_frames=False)
            profiles = []

            class CaptureConfiguredPolicy(DeterministicReachPolicy):
                def configure_capture(self, *, profile_path, recorder, storage, output_dir, max_bytes_per_run):
                    profiles.append(load_pi05_capture_settings(
                        profile_path, storage_override=storage, output_dir=output_dir,
                        max_bytes_per_run_override=max_bytes_per_run,
                    ))

            def factory(_config):
                return {
                    "policy": CaptureConfiguredPolicy(step_size=0.1, action_horizon=2),
                    "embodiment": FrankaEmbodimentAdapter(),
                    "backend": DeterministicFrankaBackend(),
                    "scenario": DeterministicReachScenario(
                        max_steps=2, target_position=(0.65, 0, 0.3),
                        workspace_max=(2.0, 2.0, 2.0),
                    ),
                }

            runs = {}
            for name in ("clean", "clean-repeat", "fault", "guarded"):
                runs[name] = execute_variant(
                    settings, name, fault=FaultSpec(0, 9, 0, 0.5),
                    instance_factory=factory, seed_runtime=lambda _: None,
                    capture=True,
                )
            checks = validate_runs(runs, tolerance=1e-6, fault=FaultSpec(0, 9, 0, 0.5))
            self.assertTrue(all(checks.values()), checks)
            fault_events = runs["fault"]["events"]
            raw = next(e for e in fault_events if e["event_type"] == "raw_action")
            modified = next(e for e in fault_events if e["event_type"] == "postprocessed_action")
            received = next(e for e in fault_events if e.get("boundary") == "backend_received_action")
            self.assertAlmostEqual(raw["value"][0], 0.1)
            self.assertAlmostEqual(modified["value"][0], 0.9)
            self.assertEqual(modified["value"], received["value"])
            self.assertTrue(all(run["evidence"]["runtime_integrity"]["status"] == "passed" for run in runs.values()))
            self.assertEqual(runs["guarded"]["result"]["reason"], "safety_rejected:action_limit_exceeded")
            self.assertFalse(any(e["event_type"] == "execution_receipt" for e in runs["guarded"]["events"]))
            self.assertEqual(len(profiles), 4)
            self.assertTrue(all(profile.storage.value == "npy" and profile.max_module_calls == 1 for profile in profiles))

    def test_input_mismatch_fails_the_comparison(self):
        base = {"events": [
            {"event_type": "observation", "step": 0, "images": {"agent": {"sha256": "one"}}},
            {"event_type": "raw_action", "step": 0, "value": [0.1] * 7},
            {"event_type": "postprocessed_action", "step": 0, "value": [0.1] * 7},
        ], "evidence": {"runtime_integrity": {"status": "passed"}}}
        runs = {name: json.loads(json.dumps(base)) for name in ("clean", "clean-repeat", "fault", "guarded")}
        runs["clean-repeat"]["events"][0]["images"]["agent"]["sha256"] = "two"
        self.assertFalse(validate_runs(runs, tolerance=1e-6, fault=FaultSpec(0, 9, 0, 0.5))["initial_observations_match"])

    def test_capture_pairing_selects_explicit_coordinate_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "a.npy", np.array([1, 2], dtype=np.float32))
            event = {
                "event_type": "model_activation", "step": 0, "module": "action_out_proj",
                "module_call_index": 0, "denoise_step": 0, "policy_call_id": "policy:000000",
                "activation_ref": str(root / "a.npy"), "activation_shape": [2],
                "activation_dtype": "float32", "capture": {
                    "storage_mode": "npy", "tensor_summary": {
                        "sha256": "sha256:" + hashlib.sha256(np.array([1, 2], dtype=np.float32).tobytes()).hexdigest(),
                    },
                },
            }
            captures, receipt = select_initial_captures([event, {**event, "module_call_index": 1}], root)
            self.assertEqual(len(captures), 1)
            self.assertEqual(receipt[0]["module_call_index"], 0)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                select_initial_captures([event, event], root)
            with self.assertRaisesRegex(ValueError, "full stored"):
                select_initial_captures([{**event, "activation_ref": "summary:abc"}], root)
            np.save(root / "a.npy", np.array([3, 4], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "checksum"):
                select_initial_captures([event], root)

    def test_existing_evidence_is_not_overwritten_on_cli_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "failure.json").write_text("existing evidence")
            with self.assertRaisesRegex(ValueError, "existing evidence"):
                main(["--out", str(root)])
            self.assertEqual((root / "failure.json").read_text(), "existing evidence")

    def test_persisted_first_call_pairs_reach_real_diff_and_report_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = RunSettings(out=root)
            runs = {}
            model = {"model_name": "unit-test-model", "revision": "unit-test-revision"}
            for name in ("clean", "clean-repeat", "fault", "guarded"):
                run_dir = root / "runs" / name
                run_dir.mkdir(parents=True)
                events = [
                    {"event_type": "raw_action", "step": 0, "value": [0.1] * 7},
                    {"event_type": "postprocessed_action", "step": 0, "value": [0.9 if name in {"fault", "guarded"} else 0.1] + [0.1] * 6},
                    {"event_type": "outcome", "step": 1, "success": False, "steps": 1},
                ]
                for module in ("model.action_in_proj", "model.action_out_proj"):
                    tensor = np.ones((1, 2, 3), dtype=np.float32)
                    path = run_dir / f"{module}.npy"
                    np.save(path, tensor)
                    events.append({
                        "event_type": "model_activation", "step": 0, "module": module,
                        "module_call_index": 0, "denoise_step": 0,
                        "policy_call_id": "policy:000000", "activation_ref": str(path),
                        "activation_shape": list(tensor.shape), "activation_dtype": "float32",
                        "capture": {"storage_mode": "npy", "tensor_summary": TensorSummary.from_value(tensor).to_dict()},
                        "model_ref": model,
                    })
                trace = run_dir / f"{name}.jsonl"
                trace.write_text("".join(json.dumps(event) + "\n" for event in events))
                runs[name] = {
                    "events": events, "trace": trace,
                    "config": {"trace": {"scenario": "unit-test"}, "components": {"policy": {"options": {"checkpoint": "unit-test-model", "revision": "unit-test-revision"}}}},
                }
            manifest = root / "fault.yaml"
            write_intervention_manifest(manifest, FaultSpec(0, 9, 0, 0.5))
            pairs = _write_pairs(settings, runs, manifest)
            self.assertEqual(set(pairs), {"clean-repeat", "fault", "guarded"})
            self.assertTrue(all(pair["negative_control_stable"] for pair in pairs.values()))
            report = json.loads((root / "comparisons/fault/mech/mech_report.json").read_text())
            self.assertEqual(report["evidence_level"], 1)
            self.assertFalse(report["model_root_cause_claim_allowed"])
            self.assertEqual(report["external_root_cause_layer"], "action_postprocess")
            self.assertTrue((root / "comparisons/fault/mech/selection.json").exists())


if __name__ == "__main__":
    unittest.main()
