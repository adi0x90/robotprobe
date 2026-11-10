import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from robotprobe.runtime.evidence import (
    analyze_runtime_trace,
    create_runtime_evidence_bundle,
    render_runtime_report,
    render_runtime_timeline,
)

ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC_CONFIG = (
    ROOT
    / "examples"
    / "packs"
    / "franka-pi05-runtime"
    / "configs"
    / "deterministic.yaml"
)


class RuntimeEvidenceTests(unittest.TestCase):
    def test_safety_rejection_is_integrity_pass_without_backend_motion(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "rejected.jsonl", rejected_events())

            evidence = analyze_runtime_trace(trace_path)

        self.assertEqual(evidence["format"], "robotprobe.runtime.evidence.v0")
        self.assertEqual(evidence["runtime_integrity"]["status"], "passed")
        self.assertEqual(evidence["task_outcome"]["status"], "failed")
        self.assertEqual(
            evidence["task_outcome"]["reason"],
            "safety_rejected:action_limit_exceeded",
        )
        self.assertEqual(evidence["motion"]["disposition"], "rejected_pre_backend")
        self.assertEqual(evidence["motion"]["execution_receipts"], 0)
        self.assertEqual(evidence["motion"]["applied_commands"], 0)

    def test_applied_simulation_command_is_classified_as_applied(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "applied.jsonl", applied_events())

            evidence = analyze_runtime_trace(trace_path)

        self.assertEqual(evidence["runtime_integrity"]["status"], "passed")
        self.assertEqual(evidence["task_outcome"]["status"], "succeeded")
        self.assertEqual(evidence["motion"]["disposition"], "applied")
        self.assertEqual(evidence["motion"]["applied_commands"], 1)

    def test_hardware_shadow_receipt_never_becomes_task_success(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "shadow.jsonl", shadow_events())

            evidence = analyze_runtime_trace(trace_path)

        self.assertEqual(evidence["runtime_integrity"]["status"], "passed")
        self.assertEqual(evidence["task_outcome"]["status"], "not_assessed")
        self.assertEqual(evidence["motion"]["disposition"], "shadow_evaluated")
        self.assertEqual(evidence["motion"]["applied_commands"], 0)

    def test_trace_without_actions_has_no_motion_disposition(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "no-action.jsonl", no_action_events())

            evidence = analyze_runtime_trace(trace_path)

        self.assertEqual(evidence["runtime_integrity"]["status"], "passed")
        self.assertEqual(evidence["motion"]["disposition"], "none")

    def test_generic_runtime_claims_do_not_hardcode_franka(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(
                Path(td) / "generic.jsonl",
                no_action_events(),
            )

            evidence = analyze_runtime_trace(trace_path)

        boundaries = " ".join(evidence["claims"]["does_not_prove"])
        self.assertIn("physical robot safety", boundaries)
        self.assertNotIn("Franka", boundaries)

    def test_broken_runtime_lineage_fails_integrity_with_specific_issues(self):
        events = copy.deepcopy(applied_events())
        action_chunk = next(
            item for item in events if item["event_type"] == "action_chunk"
        )
        action_chunk["policy_call_id"] = "policy:missing"
        action_chunk["source_observation_id"] = "obs:missing"
        scheduled = next(
            item for item in events if item["event_type"] == "scheduled_action"
        )
        scheduled["chunk_id"] = "chunk:missing"
        decision = next(
            item for item in events if item["event_type"] == "safety_decision"
        )
        decision["command_id"] = "command:missing-scheduled"
        receipt = next(
            item for item in events if item["event_type"] == "execution_receipt"
        )
        receipt["command_id"] = "command:missing-authorized"
        events.insert(
            -1,
            event(
                0,
                "model_activation",
                "model_activation",
                module="model.layers.0",
                module_kind="vision",
                probe_id="vision.early",
                activation_ref="sha256:activation",
                activation_shape=[1, 4],
                activation_dtype="float32",
                model_ref={"checkpoint": "example/model"},
                capture={"storage": "summary"},
                policy_call_id="policy:000000",
                action_chunk_id="chunk:missing-activation",
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "broken-lineage.jsonl", events)

            evidence = analyze_runtime_trace(trace_path)

        self.assertTrue(evidence["runtime_integrity"]["trace_valid"])
        self.assertEqual(evidence["runtime_integrity"]["status"], "failed")
        codes = {
            issue["code"] for issue in evidence["runtime_integrity"]["issues"]
        }
        self.assertTrue(
            {
                "unknown_policy_call",
                "unknown_observation",
                "unknown_action_chunk",
                "unknown_scheduled_command",
                "unknown_authorized_command",
                "unknown_activation_action_chunk",
            }.issubset(codes),
            codes,
        )

    def test_activation_summary_is_observational_and_reports_drops(self):
        events = applied_events()
        events.insert(
            -1,
            event(
                0,
                "model_activation",
                "model_activation",
                module="model.layers.0",
                module_kind="vision",
                probe_id="vision.early",
                activation_ref="sha256:activation",
                activation_shape=[1, 4],
                activation_dtype="float32",
                model_ref={"checkpoint": "example/model"},
                capture={"storage": "summary"},
                policy_call_id="policy:000000",
                action_chunk_id="chunk:000000",
            ),
        )
        events.insert(
            -1,
            event(
                0,
                "activation_capture_dropped",
                "model_activation",
                reason="per_module_call_limit",
                module="model.layers.0",
                probe_id="vision.early",
                policy_call_id="policy:000000",
                action_chunk_id="chunk:000000",
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "capture.jsonl", events)

            evidence = analyze_runtime_trace(trace_path)

        instrumentation = evidence["instrumentation"]
        self.assertEqual(evidence["runtime_integrity"]["status"], "passed")
        self.assertEqual(instrumentation["activation_events"], 1)
        self.assertEqual(instrumentation["dropped_events"], 1)
        self.assertEqual(instrumentation["modules"], ["model.layers.0"])
        self.assertEqual(instrumentation["probes"], ["vision.early"])
        self.assertEqual(instrumentation["storage_modes"], ["summary"])
        self.assertFalse(instrumentation["causal_effect_established"])

    def test_hardware_shadow_cannot_claim_an_applied_command(self):
        events = copy.deepcopy(shadow_events())
        receipt = next(
            item
            for item in events
            if item["event_type"] == "execution_receipt"
        )
        receipt["command_applied"] = True
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "shadow-applied.jsonl", events)

            evidence = analyze_runtime_trace(trace_path)

        self.assertEqual(evidence["runtime_integrity"]["status"], "failed")
        self.assertIn(
            "hardware_shadow_command_applied",
            {
                issue["code"]
                for issue in evidence["runtime_integrity"]["issues"]
            },
        )

    def test_duplicate_runtime_lineage_identifier_fails_integrity(self):
        events = applied_events()
        action_chunk = next(
            item for item in events if item["event_type"] == "action_chunk"
        )
        events.insert(-1, copy.deepcopy(action_chunk))
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "duplicate-chunk.jsonl", events)

            evidence = analyze_runtime_trace(trace_path)

        self.assertEqual(evidence["runtime_integrity"]["status"], "failed")
        self.assertIn(
            "duplicate_action_chunk_id",
            {
                issue["code"]
                for issue in evidence["runtime_integrity"]["issues"]
            },
        )

    def test_non_string_lineage_identifier_is_a_diagnostic_failure(self):
        events = applied_events()
        action_chunk = next(
            item for item in events if item["event_type"] == "action_chunk"
        )
        action_chunk["chunk_id"] = ["not", "a", "string"]
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "invalid-id.jsonl", events)

            evidence = analyze_runtime_trace(trace_path)

        self.assertEqual(evidence["runtime_integrity"]["status"], "failed")
        self.assertIn(
            "invalid_lineage_identifier",
            {
                issue["code"]
                for issue in evidence["runtime_integrity"]["issues"]
            },
        )

    def test_comparison_finds_first_action_and_safety_divergence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = write_trace(root / "baseline.jsonl", applied_events())
            candidate = write_trace(root / "candidate.jsonl", rejected_events())

            evidence = analyze_runtime_trace(
                candidate,
                baseline_path=baseline,
            )

        comparison = evidence["comparison"]
        self.assertEqual(comparison["first_action_divergence_step"], 0)
        self.assertTrue(comparison["safety_decision_changed"])
        self.assertEqual(comparison["applied_command_delta"], -1)
        self.assertTrue(comparison["outcome_changed"])

    def test_report_and_timeline_are_deterministic_and_path_free(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = write_trace(root / "baseline.jsonl", applied_events())
            candidate_events = rejected_events()
            candidate = write_trace(root / "candidate.jsonl", candidate_events)
            evidence = analyze_runtime_trace(
                candidate,
                baseline_path=baseline,
            )

            report = render_runtime_report(evidence)
            timeline = render_runtime_timeline(candidate_events, evidence)

        self.assertIn("Runtime integrity: `PASS`", report)
        self.assertIn("Motion disposition: `rejected_pre_backend`", report)
        self.assertIn("does not prove", report.lower())
        self.assertIn("policy:000000", timeline)
        self.assertIn("command:000000:000", timeline)
        self.assertNotIn(str(root), report)
        self.assertNotIn(str(root), timeline)
        self.assertEqual(report, render_runtime_report(evidence))
        self.assertEqual(
            timeline,
            render_runtime_timeline(candidate_events, evidence),
        )

    def test_activation_aggregate_stays_at_its_trace_position(self):
        events = applied_events()
        action_chunk_index = next(
            index
            for index, item in enumerate(events)
            if item["event_type"] == "action_chunk"
        )
        events.insert(
            action_chunk_index,
            event(
                0,
                "model_activation",
                "model_activation",
                module="model.layers.0",
                module_kind="vision",
                probe_id="vision.early",
                activation_ref="sha256:activation",
                activation_shape=[1, 4],
                activation_dtype="float32",
                model_ref={"checkpoint": "example/model"},
                capture={"storage": "summary"},
                policy_call_id="policy:000000",
                action_chunk_id="chunk:000000",
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            trace_path = write_trace(Path(td) / "activation-order.jsonl", events)
            evidence = analyze_runtime_trace(trace_path)

        timeline = render_runtime_timeline(events, evidence)

        self.assertLess(
            timeline.index("model_activation (aggregated)"),
            timeline.index("| action_chunk |"),
        )

    def test_bundle_is_self_contained_relative_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = write_trace(root / "safe.jsonl", applied_events())
            candidate = write_trace(root / "rejected.jsonl", rejected_events())
            out = root / "evidence"

            result = create_runtime_evidence_bundle(
                trace_path=candidate,
                config_path=DETERMINISTIC_CONFIG,
                baseline_path=baseline,
                out_dir=out,
            )

            relative_files = sorted(
                str(path.relative_to(out))
                for path in out.rglob("*")
                if path.is_file()
            )
            evidence_text = (out / "evidence.json").read_text(encoding="utf-8")
            evidence_data = json.loads(evidence_text)
            observed_hashes = {
                item["path"]: sha256(out / item["path"])
                for item in result["files"]
            }

        self.assertEqual(
            result["format"],
            "robotprobe.runtime.evidence_bundle.v0",
        )
        self.assertEqual(
            relative_files,
            [
                "evidence.json",
                "inputs/baseline.jsonl",
                "inputs/runtime-config.yaml",
                "inputs/trace.jsonl",
                "manifest.json",
                "report.md",
                "timeline.md",
            ],
        )
        self.assertNotIn(str(ROOT), evidence_text)
        self.assertNotIn(str(root), evidence_text)
        self.assertEqual(
            evidence_data["configuration"]["pack"]["id"],
            "robotprobe-pack-franka-pi05-runtime",
        )
        self.assertEqual(
            evidence_data["configuration"]["components"]["backend"][
                "component_id"
            ],
            "franka.deterministic_kinematic.v0",
        )
        self.assertRegex(
            evidence_data["configuration"]["sha256"],
            r"^[0-9a-f]{64}$",
        )
        for item in result["files"]:
            self.assertEqual(item["sha256"], observed_hashes[item["path"]])

    def test_bundle_refuses_non_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace = write_trace(root / "trace.jsonl", rejected_events())
            out = root / "evidence"
            out.mkdir()
            (out / "unrelated.txt").write_text("preserve me", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be empty"):
                create_runtime_evidence_bundle(
                    trace_path=trace,
                    out_dir=out,
                )

            self.assertEqual(
                (out / "unrelated.txt").read_text(encoding="utf-8"),
                "preserve me",
            )


def write_trace(path: Path, events: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejected_events() -> list[dict]:
    return [
        event(0, "instruction", "instruction", value="Move to the target."),
        event(
            0,
            "observation",
            "observation",
            observation_ref="obs:000000",
            observation_id="obs:000000",
        ),
        event(
            0,
            "policy_inference",
            "policy_forward",
            status="started",
            policy_call_id="policy:000000",
            action_chunk_id="chunk:000000",
            observation_id="obs:000000",
        ),
        event(
            0,
            "policy_inference",
            "policy_forward",
            status="completed",
            policy_call_id="policy:000000",
            action_chunk_id="chunk:000000",
            observation_id="obs:000000",
        ),
        event(
            0,
            "action_chunk",
            "policy_output",
            chunk_id="chunk:000000",
            policy_call_id="policy:000000",
            source_observation_id="obs:000000",
            command_space="franka.cartesian_delta_gripper.v0",
            shape=[1, 7],
            value=[[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        ),
        event(
            0,
            "scheduled_action",
            "action_schedule",
            command_id="command:000000:000",
            chunk_id="chunk:000000",
            policy_call_id="policy:000000",
            source_observation_id="obs:000000",
            action_index=0,
            command_space="franka.cartesian_delta_gripper.v0",
            value=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
        event(
            0,
            "safety_decision",
            "safety",
            command_id="command:000000:000",
            chunk_id="chunk:000000",
            policy_call_id="policy:000000",
            source_observation_id="obs:000000",
            accepted=False,
            disposition="rejected",
            reason="action_limit_exceeded",
        ),
        event(
            0,
            "outcome",
            "outcome",
            success=False,
            steps=0,
            reason="safety_rejected:action_limit_exceeded",
            execution_mode="simulation",
        ),
    ]


def applied_events() -> list[dict]:
    return [
        *action_prefix([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        event(
            0,
            "safety_decision",
            "safety",
            command_id="command:000000:000",
            chunk_id="chunk:000000",
            policy_call_id="policy:000000",
            source_observation_id="obs:000000",
            accepted=True,
            disposition="authorized",
            reason="within_envelope",
        ),
        event(
            0,
            "execution_receipt",
            "backend_execution",
            command_id="command:000000:000",
            backend_id="franka.deterministic_kinematic.v0",
            execution_mode="simulation",
            command_applied=True,
            accepted=True,
        ),
        event(
            1,
            "outcome",
            "outcome",
            success=True,
            steps=1,
            reason="scenario_success",
            execution_mode="simulation",
        ),
    ]


def shadow_events() -> list[dict]:
    return [
        *action_prefix([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        event(
            0,
            "safety_decision",
            "safety",
            command_id="command:000000:000",
            chunk_id="chunk:000000",
            policy_call_id="policy:000000",
            source_observation_id="obs:000000",
            accepted=True,
            disposition="shadow_only",
            reason="within_envelope",
        ),
        event(
            0,
            "execution_receipt",
            "backend_execution",
            command_id="command:000000:000",
            backend_id="franka.ros2_read_only_shadow.v0",
            execution_mode="hardware_shadow",
            command_applied=False,
            accepted=True,
        ),
        event(
            1,
            "outcome",
            "outcome",
            success=False,
            steps=1,
            reason="max_steps_exhausted",
            execution_mode="hardware_shadow",
        ),
    ]


def no_action_events() -> list[dict]:
    return [
        event(0, "instruction", "instruction", value="Observe only."),
        event(
            0,
            "observation",
            "observation",
            observation_ref="obs:000000",
            observation_id="obs:000000",
        ),
        event(
            0,
            "outcome",
            "outcome",
            success=False,
            steps=0,
            reason="no_action_requested",
            execution_mode="simulation",
        ),
    ]


def action_prefix(value: list[float]) -> list[dict]:
    return [
        event(0, "instruction", "instruction", value="Move to the target."),
        event(
            0,
            "observation",
            "observation",
            observation_ref="obs:000000",
            observation_id="obs:000000",
        ),
        event(
            0,
            "policy_inference",
            "policy_forward",
            status="started",
            policy_call_id="policy:000000",
            action_chunk_id="chunk:000000",
            observation_id="obs:000000",
        ),
        event(
            0,
            "policy_inference",
            "policy_forward",
            status="completed",
            policy_call_id="policy:000000",
            action_chunk_id="chunk:000000",
            observation_id="obs:000000",
        ),
        event(
            0,
            "action_chunk",
            "policy_output",
            chunk_id="chunk:000000",
            policy_call_id="policy:000000",
            source_observation_id="obs:000000",
            command_space="franka.cartesian_delta_gripper.v0",
            shape=[1, 7],
            value=[value],
        ),
        event(
            0,
            "scheduled_action",
            "action_schedule",
            command_id="command:000000:000",
            chunk_id="chunk:000000",
            policy_call_id="policy:000000",
            source_observation_id="obs:000000",
            action_index=0,
            command_space="franka.cartesian_delta_gripper.v0",
            value=value,
        ),
    ]


def event(step: int, event_type: str, layer: str, **fields) -> dict:
    return {
        "run_id": "runtime:test",
        "target": "deterministic-franka-policy",
        "scenario": "deterministic-franka-reach",
        "replay_level": "L2_SIM_REPLAY",
        "step": step,
        "event_type": event_type,
        "layer": layer,
        **fields,
    }


if __name__ == "__main__":
    unittest.main()
