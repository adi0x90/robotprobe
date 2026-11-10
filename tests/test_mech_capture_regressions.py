import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe.mech import collect
from robotprobe.mech.activation_hooks import CapturePolicy, ForwardHookActivationCollector, ModuleSelector
from robotprobe.mech.activation_store import ActivationStore
from robotprobe.mech.schemas import load_capture_config, validate_capture_config
from robotprobe.schema import validate_trace_file


class Handle:
    def __init__(self, hooks, hook):
        self.hooks, self.hook = hooks, hook

    def remove(self):
        self.hooks.remove(self.hook)


class Module:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return Handle(self.hooks, hook)

    def forward(self, value):
        for hook in self.hooks:
            hook(self, (), value)


class Model:
    def __init__(self):
        self.pre_hooks = []
        self.layers = {"first": Module(), "second": Module()}

    def register_forward_pre_hook(self, hook):
        self.pre_hooks.append(hook)
        return Handle(self.pre_hooks, hook)

    def named_modules(self):
        return self.layers.items()

    def forward(self, value):
        for hook in self.pre_hooks:
            hook(self, (value,))
        for layer in self.layers.values():
            layer.forward(value)


def config():
    return {
        "format": "robotprobe.mech.capture_config.v0", "id": "regression", "model_access": "white_box",
        "layers": [{"name": "layer", "kind": "activation", "capture": "output"}],
        "storage": {"format": "npy", "dtype": "float16", "max_bytes_per_run": 4096},
        "claim_boundary": {"no_feature_labels": True, "no_causal_claims": True},
    }


class CaptureRegressionTests(unittest.TestCase):
    def test_forward_steps_are_shared_across_modules_and_ignore_activation_values(self):
        model = Model()
        collector = ForwardHookActivationCollector(
            module_selector=ModuleSelector(["first", "second"]), capture_policy=CapturePolicy()
        )
        with collector.attach(model):
            for value in [0.25, 0.75, -1.75]:
                model.forward(np.array([value]))
        self.assertEqual([c["step"] for c in collector.captures], [0, 0, 1, 1, 2, 2])
        self.assertFalse(model.pre_hooks)

    def test_forward_sampling_uses_forward_index(self):
        model = Model()
        collector = ForwardHookActivationCollector(
            module_selector=ModuleSelector(["first"]), capture_policy=CapturePolicy(every_n_steps=2, max_steps=2)
        )
        with collector.attach(model):
            for _ in range(4):
                model.forward(np.array([100.0]))
        self.assertEqual([c["step"] for c in collector.captures], [0, 2])

    def test_hook_capture_is_a_snapshot(self):
        model = Model()
        collector = ForwardHookActivationCollector(
            module_selector=ModuleSelector(["first"]), capture_policy=CapturePolicy()
        )
        values = np.array([0.5, -3.0])
        with collector.attach(model):
            model.forward(values)
        values[:] = 99
        np.testing.assert_array_equal(collector.captures[0]["activation"], [0.5, -3.0])

    def test_explicit_step_for_non_torch_runtime(self):
        collector = ForwardHookActivationCollector(
            module_selector=ModuleSelector([]), capture_policy=CapturePolicy()
        )
        self.assertTrue(hasattr(collector, "set_step"), "non-Torch runtimes need explicit episode steps")
        collector.set_step(12)
        collector._make_hook("layer")(None, (), np.array([-3.5]))
        self.assertEqual(collector.captures[0]["step"], 12)

    def test_invalid_sampling_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            CapturePolicy(every_n_steps=0)

    def test_non_mapping_json_is_rejected_at_load(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "capture.json"
            p.write_text("[]")
            with self.assertRaises(ValueError):
                load_capture_config(p)

    def test_validator_reports_bad_types_without_crashing(self):
        self.assertFalse(validate_capture_config([])["valid"])
        for value in ["invalid", True, 1.2, [], {}]:
            candidate = config()
            candidate["storage"]["max_bytes_per_run"] = value
            result = validate_capture_config(candidate)
            self.assertFalse(result["valid"], value)

    def test_validator_rejects_unsupported_capture_settings(self):
        for field, value in [("dtype", "bfloat16"), ("dtype", "object")]:
            candidate = config()
            candidate["storage"][field] = value
            self.assertFalse(validate_capture_config(candidate)["valid"])
        candidate = config()
        candidate["steps"] = {"mode": "every_n", "n": 0}
        self.assertFalse(validate_capture_config(candidate)["valid"])

    def test_first_action_divergence_uses_chronology(self):
        clean = [{"event_type": "postprocessed_action", "step": 2, "value": [0]},
                 {"event_type": "raw_action", "step": 14, "value": [0]}]
        perturbed = [{"event_type": "postprocessed_action", "step": 2, "value": [1]},
                     {"event_type": "raw_action", "step": 14, "value": [1]}]
        self.assertEqual(collect._first_action_divergence(clean, perturbed), 2)

    def test_synthetic_step_zero_and_per_run_trace_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            c = root / "capture.json"
            c.write_text(json.dumps(config()))
            traces = []
            for label, value in [("clean", 0), ("perturbed", 1)]:
                p = root / f"{label}.jsonl"
                p.write_text(json.dumps({"run_id": label, "event_type": "raw_action", "step": 0, "value": [value]}) + "\n")
                traces.append(p)
            result = collect.collect_activations(target="synthetic", capture_config=c, out_dir=root / "mech",
                                                clean_trace=traces[0], perturbed_trace=traces[1])
            index = json.loads(Path(result["activation_index"]).read_text())
            self.assertEqual({cap["step"] for cap in index["captures"]}, {0})
            self.assertEqual(len(index["captures"]), 2)
            self.assertIsInstance(result["activation_events"], dict)
            for path in result["activation_events"].values():
                self.assertTrue(validate_trace_file(path).valid)

    def test_persist_pair_honors_sampling_dtype_and_identity(self):
        self.assertTrue(hasattr(collect, "persist_capture_pair"), "missing supported adapter persistence bridge")
        with tempfile.TemporaryDirectory() as td:
            candidate = config()
            candidate["steps"] = {"mode": "every_n", "n": 2}
            pair = {label: [{"step": i, "module": "layer", "activation": np.array([i + delta], dtype=np.float32),
                            "source_dtype": "torch.bfloat16"} for i in range(4)]
                    for label, delta in [("clean", 0), ("perturbed", 1)]}
            result = collect.persist_capture_pair(
                capture_pair=pair, capture_config=candidate, out_dir=td, incident_id="i", clean_run_id="a",
                perturbed_run_id="b", model={"name": "cpu-toy"}, target="cpu-toy", scenario="swap",
                replay_level="L0_TRACE_PLAYBACK",
            )
            index = json.loads(Path(result["activation_index"]).read_text())
            self.assertEqual({cap["step"] for cap in index["captures"]}, {0, 2})
            self.assertEqual({cap["dtype"] for cap in index["captures"]}, {"float16"})
            self.assertEqual({cap["source_dtype"] for cap in index["captures"]}, {"torch.bfloat16"})
            self.assertFalse(index["synthetic"])
            self.assertEqual(index["trace_pair"], {"clean_run_id": "a", "perturbed_run_id": "b"})
            self.assertEqual(result["activation_count"], 4)
            for path in result["activation_events"].values():
                self.assertTrue(validate_trace_file(path).valid)
                event = json.loads(Path(path).read_text().splitlines()[0])
                self.assertEqual(event["scenario"], "swap")
                self.assertEqual(event["capture"]["source_dtype"], "torch.bfloat16")

    def test_serialized_storage_budget_is_enforced_before_writing(self):
        with tempfile.TemporaryDirectory() as td:
            try:
                store = ActivationStore(td, store_format="npy", max_bytes_per_run=1)
            except TypeError:
                self.fail("ActivationStore has no budget enforcement")
            with self.assertRaises(ValueError):
                store.write_activation(run_label="clean", run_id="a", step=0, module="layer", module_kind="activation",
                                       activation=np.ones(3))
            self.assertFalse(list(Path(td).rglob("*.npy")))

    def test_persist_rejects_existing_evidence_before_rewriting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "activation_index.json"
            marker.write_text('{"original": true}')
            with self.assertRaisesRegex(ValueError, "empty"):
                collect.persist_capture_pair(
                    capture_pair={"clean": [], "perturbed": []}, capture_config=config(), out_dir=root,
                    incident_id="i", clean_run_id="a", perturbed_run_id="b", model={}, target="toy", scenario="s",
                    replay_level="L0_TRACE_PLAYBACK",
                )
            self.assertEqual(marker.read_text(), '{"original": true}')
