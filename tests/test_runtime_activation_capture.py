import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.runtime.capture import (
    CaptureProfile,
    CaptureStorageMode,
    ContextualActivationCollector,
    ModuleProbeSpec,
    TensorSummary,
    extract_activation_array,
)
from robotprobe.runtime.types import InferenceContext
from robotprobe.schema import validate_trace_events


class RuntimeActivationCaptureTests(unittest.TestCase):
    def test_extracts_tensor_like_tuple_and_mapping_outputs(self):
        expected = np.array([[1.0, 2.0]], dtype=np.float32)

        self.assertIsNone(extract_activation_array({"metadata": "not-a-tensor"}))
        np.testing.assert_array_equal(extract_activation_array(FakeTensor(expected)), expected)
        np.testing.assert_array_equal(
            extract_activation_array(("ignored", FakeTensor(expected))),
            expected,
        )
        np.testing.assert_array_equal(
            extract_activation_array({"metadata": "ignored", "hidden": FakeTensor(expected)}),
            expected,
        )

    def test_tensor_summary_is_complete_deterministic_and_contains_no_values(self):
        activation = np.array([1.0, -2.0, np.nan, np.inf], dtype=np.float32)

        summary = TensorSummary.from_value(activation)
        rendered = summary.to_dict()

        self.assertEqual(rendered["shape"], [4])
        self.assertEqual(rendered["dtype"], "float32")
        self.assertEqual(rendered["finite_count"], 2)
        self.assertEqual(rendered["non_finite_count"], 2)
        self.assertEqual(rendered["minimum"], -2.0)
        self.assertEqual(rendered["maximum"], 1.0)
        self.assertEqual(rendered["mean"], -0.5)
        self.assertEqual(rendered["standard_deviation"], 1.5)
        self.assertAlmostEqual(rendered["l2_norm"], np.sqrt(5.0))
        expected_digest = hashlib.sha256(np.ascontiguousarray(activation).tobytes()).hexdigest()
        self.assertEqual(rendered["sha256"], f"sha256:{expected_digest}")
        self.assertNotIn("value", rendered)
        self.assertNotIn("values", rendered)

    def test_summary_capture_records_context_and_exact_or_glob_matches(self):
        model = FakeModel("backbone.layer17", "backbone.layer18", "action_head")
        recorder = FakeRecorder()
        collector = ContextualActivationCollector(
            CaptureProfile(
                probes=(
                    ModuleProbeSpec(
                        probe_id="backbone",
                        selector="backbone.layer*",
                        module_kind="residual_stream",
                        phase="backbone",
                    ),
                    ModuleProbeSpec(
                        probe_id="head",
                        selector="action_head",
                        module_kind="action_projection",
                        phase="denoise",
                        denoise_step_from_module_call=True,
                    ),
                )
            ),
            recorder=recorder,
            model_ref={"model_name": "fake", "checkpoint_hash": "sha256:test"},
        )
        context = inference_context(policy_call=7, env_step=12)

        with collector.session(model, context):
            model.modules["backbone.layer17"].run(np.array([1.0], dtype=np.float32))
            model.modules["backbone.layer18"].run((None, np.array([2.0], dtype=np.float32)))
            model.modules["action_head"].run({"sample": np.array([3.0], dtype=np.float32)})

        events = recorder.events
        self.assertEqual(
            [event["module"] for event in events],
            ["backbone.layer17", "backbone.layer18", "action_head"],
        )
        for event in events:
            self.assertEqual(event["event_type"], "model_activation")
            self.assertEqual(event["policy_call_id"], "policy:000007")
            self.assertEqual(event["action_chunk_id"], "chunk:000007")
            self.assertEqual(event["observation_id"], "observation:000012")
            self.assertEqual(event["module_call_index"], 0)
            self.assertTrue(event["activation_ref"].startswith("summary:sha256:"))
            self.assertNotIn("value", event)
            self.assertNotIn("values", event["capture"])
        self.assertEqual(events[0]["phase"], "backbone")
        self.assertNotIn("denoise_step", events[0])
        self.assertEqual(events[-1]["phase"], "denoise")
        self.assertEqual(events[-1]["denoise_step"], 0)
        self.assertTrue(validate_trace_events(events).valid)

    def test_sampling_and_module_call_limit_are_scoped_to_policy_calls(self):
        model = FakeModel("action_head")
        recorder = FakeRecorder()
        collector = ContextualActivationCollector(
            CaptureProfile(
                probes=(
                    ModuleProbeSpec(
                        probe_id="head",
                        selector="action_head",
                        module_kind="action_projection",
                        phase="denoise",
                        every_n_policy_calls=2,
                        max_module_calls=2,
                        denoise_step_from_module_call=True,
                    ),
                )
            ),
            recorder=recorder,
            model_ref={"model_name": "fake"},
        )

        for policy_call in range(1, 4):
            with collector.session(model, inference_context(policy_call=policy_call)):
                for value in range(3):
                    model.modules["action_head"].run(np.array([value], dtype=np.float32))

        captures = [event for event in recorder.events if event["event_type"] == "model_activation"]
        self.assertEqual(
            [(event["policy_call_id"], event["module_call_index"]) for event in captures],
            [
                ("policy:000001", 0),
                ("policy:000001", 1),
                ("policy:000003", 0),
                ("policy:000003", 1),
            ],
        )
        self.assertEqual([event["denoise_step"] for event in captures], [0, 1, 0, 1])

    def test_numpy_storage_is_collision_free_and_stops_before_budget_overflow(self):
        model = FakeModel("action_head")
        recorder = FakeRecorder()
        activation = np.arange(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            reference_payload_size = numpy_payload_size(activation)
            collector = ContextualActivationCollector(
                CaptureProfile(
                    probes=(
                        ModuleProbeSpec(
                            probe_id="head",
                            selector="action_head",
                            module_kind="action_projection",
                            phase="denoise",
                            storage=CaptureStorageMode.NUMPY,
                        ),
                    ),
                    output_dir=Path(td),
                    max_bytes_per_run=reference_payload_size,
                ),
                recorder=recorder,
                model_ref={"model_name": "fake"},
            )

            with collector.session(model, inference_context(policy_call=1)):
                model.modules["action_head"].run(activation)
                model.modules["action_head"].run(activation + 1)

            stored = [event for event in recorder.events if event["event_type"] == "model_activation"]
            dropped = [
                event
                for event in recorder.events
                if event["event_type"] == "activation_capture_dropped"
            ]
            files = sorted(Path(td).rglob("*.npy"))

            self.assertEqual(len(stored), 1)
            self.assertEqual(len(files), 1)
            self.assertEqual(np.load(files[0], allow_pickle=False).tolist(), activation.tolist())
            self.assertIn("policy_000001", files[0].as_posix())
            self.assertIn("call_0000", files[0].name)
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0]["reason"], "capture_budget_exhausted")
            self.assertEqual(dropped[0]["module_call_index"], 1)
            self.assertEqual(collector.budget.used_bytes, reference_payload_size)
            self.assertTrue(validate_trace_events(recorder.events).valid)

    def test_safetensors_storage_uses_distinct_policy_and_module_coordinates(self):
        model = FakeModel("action_head")
        recorder = FakeRecorder()
        with tempfile.TemporaryDirectory() as td:
            collector = ContextualActivationCollector(
                CaptureProfile(
                    probes=(
                        ModuleProbeSpec(
                            probe_id="head",
                            selector="action_head",
                            module_kind="action_projection",
                            phase="denoise",
                            storage=CaptureStorageMode.SAFETENSORS,
                        ),
                    ),
                    output_dir=Path(td),
                    max_bytes_per_run=4096,
                ),
                recorder=recorder,
                model_ref={"model_name": "fake"},
            )

            for policy_call in (1, 2):
                with collector.session(model, inference_context(policy_call=policy_call)):
                    model.modules["action_head"].run(np.array([policy_call], dtype=np.float32))

            refs = [
                event["activation_ref"]
                for event in recorder.events
                if event["event_type"] == "model_activation"
            ]
            self.assertEqual(len(refs), 2)
            self.assertNotEqual(refs[0], refs[1])
            self.assertTrue(all(Path(ref).is_file() for ref in refs))
            self.assertTrue(all(ref.endswith(".safetensors") for ref in refs))

    def test_hooks_are_removed_even_when_inference_raises(self):
        model = FakeModel("action_head")
        collector = ContextualActivationCollector(
            CaptureProfile(
                probes=(
                    ModuleProbeSpec(
                        probe_id="head",
                        selector="action_head",
                        module_kind="action_projection",
                        phase="denoise",
                    ),
                )
            ),
            recorder=FakeRecorder(),
            model_ref={"model_name": "fake"},
        )

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with collector.session(model, inference_context(policy_call=1)):
                self.assertEqual(len(model.modules["action_head"].hooks), 1)
                raise RuntimeError("inference failed")

        self.assertEqual(model.modules["action_head"].hooks, [])


class FakeTensor:
    def __init__(self, value):
        self.value = np.array(value, copy=True)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value, copy=True)


class FakeHandle:
    def __init__(self, module, hook):
        self.module = module
        self.hook = hook

    def remove(self):
        if self.hook in self.module.hooks:
            self.module.hooks.remove(self.hook)


class FakeModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle(self, hook)

    def run(self, output):
        for hook in list(self.hooks):
            hook(self, (), output)
        return output


class FakeModel:
    def __init__(self, *module_names):
        self.modules = {name: FakeModule() for name in module_names}

    def named_modules(self):
        return list(self.modules.items())


class FakeRecorder:
    def __init__(self):
        self.events = []

    def event(self, step, event_type, layer, **fields):
        self.events.append(
            {
                "run_id": "run:test",
                "target": "fake-policy",
                "scenario": "capture",
                "replay_level": "L0_TRACE_PLAYBACK",
                "step": step,
                "event_type": event_type,
                "layer": layer,
                **fields,
            }
        )


def inference_context(*, policy_call, env_step=0):
    return InferenceContext(
        env_step=env_step,
        policy_call_id=f"policy:{policy_call:06d}",
        action_chunk_id=f"chunk:{policy_call:06d}",
        observation_id=f"observation:{env_step:06d}",
    )


def numpy_payload_size(value):
    import io

    payload = io.BytesIO()
    np.save(payload, value, allow_pickle=False)
    return len(payload.getvalue())


if __name__ == "__main__":
    unittest.main()
