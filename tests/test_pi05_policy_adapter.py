import tempfile
import subprocess
import sys
import unittest
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from robotprobe.integrations.lerobot.pi05_policy import (
    DEFAULT_PI05_CHECKPOINT,
    PI05PolicyAdapter,
    Pi05OptionalDependencyError,
    Pi05RuntimeBindings,
    _load_default_runtime,
    _make_revision_aware_processor_factory,
)
from robotprobe.runtime.types import (
    CanonicalObservation,
    InferenceContext,
    ScenarioSpec,
)


ROOT = Path(__file__).resolve().parents[1]


class Pi05PolicyAdapterTests(unittest.TestCase):
    def test_default_runtime_uses_libero_environment_processor_without_reexport(self):
        modules = {
            name: ModuleType(name) for name in (
                "torch", "huggingface_hub", "lerobot", "lerobot.configs",
                "lerobot.configs.policies", "lerobot.policies", "lerobot.policies.factory",
                "lerobot.policies.pi05", "lerobot.processor", "lerobot.processor.env_processor",
                "transformers",
            )
        }

        class LiberoProcessor:
            def observation(self, batch):
                return {**batch, "observation.state": "processed-state"}

        modules["huggingface_hub"].snapshot_download = lambda **_: "/unused"
        modules["lerobot.configs.policies"].PreTrainedConfig = SimpleNamespace(from_pretrained=lambda *_: None)
        modules["lerobot.policies.factory"].make_pre_post_processors = lambda *_: None
        modules["lerobot.policies.pi05"].PI05Policy = object()
        modules["lerobot.processor.env_processor"].LiberoProcessorStep = LiberoProcessor
        modules["transformers"].AutoTokenizer = SimpleNamespace(from_pretrained=lambda *_: None)
        with patch.dict(sys.modules, modules):
            bindings = _load_default_runtime()
        processor = bindings.environment_processor_factory()
        self.assertIsInstance(processor, LiberoProcessor)
        self.assertEqual(processor.observation({"task": "pick bowl"})["observation.state"], "processed-state")

    def test_module_import_is_lazy_without_torch_or_lerobot_imports(self):
        code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("lerobot"):
        raise AssertionError(f"heavy import attempted: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import robotprobe.integrations.lerobot.pi05_policy
print("lazy-import-ok")
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "lazy-import-ok")

    def test_missing_optional_runtime_is_reported_only_at_load(self):
        def missing_runtime():
            raise ImportError("No module named 'lerobot'")

        adapter = PI05PolicyAdapter(runtime_loader=missing_runtime)

        with self.assertRaises(Pi05OptionalDependencyError) as raised:
            adapter.load()

        self.assertIn("LeRobot PI0.5", str(raised.exception))
        self.assertIn("load()", str(raised.exception))

    def test_load_disables_compile_before_model_creation_and_applies_configuration(self):
        runtime, state = fake_runtime()
        adapter = PI05PolicyAdapter(
            checkpoint="org/pi05-test",
            device="cuda:1",
            num_inference_steps=6,
            n_action_steps=3,
            runtime=runtime,
        )

        adapter.load()

        self.assertEqual(state["config_checkpoint"], "org/pi05-test")
        snapshot = state["policy_config_snapshot"]
        self.assertFalse(snapshot["compile_model"])
        self.assertEqual(snapshot["compile_mode"], "default")
        self.assertEqual(snapshot["device"], "cuda:1")
        self.assertEqual(snapshot["num_inference_steps"], 6)
        self.assertEqual(snapshot["n_action_steps"], 3)
        self.assertEqual(state["policy_checkpoint"], "org/pi05-test")
        self.assertEqual(state["policy"].to_device, "cuda:1")
        self.assertTrue(state["policy"].eval_called)
        self.assertEqual(
            state["processor_args"]["preprocessor_overrides"],
            {"device_processor": {"device": "cuda:1"}},
        )

    def test_load_resolves_config_and_weights_from_identical_artifact_options(self):
        runtime, state = fake_runtime()
        adapter = PI05PolicyAdapter(
            checkpoint="org/pi05-test",
            device="cpu",
            n_action_steps=2,
            revision="0123456789abcdef0123456789abcdef01234567",
            cache_dir="/tmp/robotprobe-pi05-cache",
            local_files_only=True,
            runtime=runtime,
        )

        adapter.load()

        expected = {
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "cache_dir": "/tmp/robotprobe-pi05-cache",
            "local_files_only": True,
        }
        self.assertEqual(state["config_load_options"], expected)
        self.assertEqual(state["policy_load_options"], expected)
        self.assertEqual(
            {
                key: state["processor_args"][key]
                for key in ("revision", "cache_dir", "local_files_only")
            },
            expected,
        )

    def test_processor_factory_resolves_pinned_remote_artifacts_to_one_local_snapshot(self):
        calls = {}

        def snapshot_downloader(**kwargs):
            calls["download"] = kwargs
            return "/cache/models--org--pi05/snapshots/012345"

        def processor_factory(config, checkpoint, **kwargs):
            calls["processor"] = {
                "config": config,
                "checkpoint": checkpoint,
                **kwargs,
            }
            return "preprocessor", "postprocessor"

        wrapped = _make_revision_aware_processor_factory(
            processor_factory,
            snapshot_downloader,
        )
        config = object()

        result = wrapped(
            config,
            "org/pi05",
            revision="0123456789abcdef0123456789abcdef01234567",
            cache_dir="/cache",
            local_files_only=True,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        self.assertEqual(result, ("preprocessor", "postprocessor"))
        self.assertEqual(
            calls["download"],
            {
                "repo_id": "org/pi05",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "cache_dir": "/cache",
                "local_files_only": True,
                "allow_patterns": (
                    "policy_preprocessor*",
                    "policy_postprocessor*",
                ),
            },
        )
        self.assertEqual(
            calls["processor"],
            {
                "config": config,
                "checkpoint": "/cache/models--org--pi05/snapshots/012345",
                "preprocessor_overrides": {
                    "device_processor": {"device": "cpu"}
                },
            },
        )

    def test_load_pins_the_transitive_tokenizer_and_records_its_provenance(self):
        runtime, state = fake_runtime(with_tokenizer_loader=True)
        adapter = PI05PolicyAdapter(
            checkpoint="org/pi05-test",
            device="cpu",
            n_action_steps=2,
            revision="0123456789abcdef0123456789abcdef01234567",
            tokenizer_checkpoint="org/pi05-tokenizer",
            tokenizer_revision="abcdef0123456789abcdef0123456789abcdef01",
            cache_dir="/tmp/robotprobe-pi05-cache",
            local_files_only=True,
            runtime=runtime,
        )

        adapter.load()
        adapter.reset(
            ScenarioSpec(
                scenario_id="libero-task-0",
                instruction="Pick up the black bowl.",
                seed=7,
                max_steps=20,
            )
        )
        chunk = adapter.predict(
            canonical_observation(),
            InferenceContext(
                env_step=0,
                policy_call_id="policy:000000",
                action_chunk_id="chunk:000000",
                observation_id="observation:000003",
            ),
        )

        self.assertEqual(state["tokenizer_checkpoint"], "org/pi05-tokenizer")
        self.assertEqual(
            state["tokenizer_load_options"],
            {
                "revision": "abcdef0123456789abcdef0123456789abcdef01",
                "cache_dir": "/tmp/robotprobe-pi05-cache",
                "local_files_only": True,
            },
        )
        tokenizer_override = state["processor_args"]["preprocessor_overrides"][
            "tokenizer_processor"
        ]
        self.assertIs(tokenizer_override["tokenizer"], state["tokenizer"])
        self.assertEqual(
            chunk.metadata["revision"],
            "0123456789abcdef0123456789abcdef01234567",
        )
        self.assertEqual(
            chunk.metadata["tokenizer_revision"],
            "abcdef0123456789abcdef0123456789abcdef01",
        )

    def test_predict_preserves_environment_and_policy_processor_boundaries_and_full_chunk(self):
        log = []
        runtime, state = fake_runtime(log=log, chunk_length=4)
        collector = FakeCollector(log)
        adapter = PI05PolicyAdapter(
            checkpoint="org/pi05-test",
            device="cpu",
            num_inference_steps=5,
            n_action_steps=2,
            runtime=runtime,
            activation_collector=collector,
        )
        adapter.load()
        scenario = ScenarioSpec(
            scenario_id="libero-task-0",
            instruction="Pick up the black bowl.",
            seed=7,
            max_steps=20,
        )
        adapter.reset(scenario)
        context = InferenceContext(
            env_step=3,
            policy_call_id="policy:000004",
            action_chunk_id="chunk:000004",
            observation_id="observation:000003",
        )
        log.clear()

        chunk = adapter.predict(canonical_observation(), context)

        self.assertEqual(
            log,
            [
                "environment",
                "preprocessor",
                "capture.enter",
                "inference.enter",
                "policy.predict_action_chunk",
                "inference.exit",
                "capture.exit",
                "postprocessor",
            ],
        )
        raw_batch = state["environment_input"]
        self.assertEqual(
            sorted(key for key in raw_batch if key.startswith("observation.images.")),
            ["observation.images.image", "observation.images.image2"],
        )
        self.assertEqual(raw_batch["observation.images.image"].shape, (1, 3, 2, 3))
        self.assertEqual(raw_batch["task"], ["Pick up the black bowl."])
        robot_state = raw_batch["observation.robot_state"]
        self.assertEqual(robot_state["eef"]["pos"].shape, (1, 3))
        self.assertEqual(robot_state["eef"]["quat"].shape, (1, 4))
        self.assertEqual(robot_state["gripper"]["qpos"].shape, (1, 2))
        preprocessor_input = state["preprocessor_input"]
        self.assertEqual(preprocessor_input["observation.state"].shape, (1, 8))
        self.assertTrue(preprocessor_input["environment_processed"])
        self.assertIs(state["policy_batch"], state["preprocessor_output"])
        self.assertIs(collector.model, state["policy"])
        self.assertEqual(collector.context, context)
        self.assertEqual(chunk.values.shape, (4, 7))
        self.assertEqual(chunk.chunk_id, context.action_chunk_id)
        self.assertEqual(chunk.policy_call_id, context.policy_call_id)
        self.assertEqual(chunk.source_observation_id, context.observation_id)
        self.assertEqual(chunk.source_observation_captured_at_ns, 10_000)
        self.assertEqual(chunk.metadata["checkpoint"], "org/pi05-test")
        self.assertEqual(chunk.metadata["configured_n_action_steps"], 2)
        self.assertEqual(state["policy"].reset_calls, 1)

    def test_close_releases_loaded_runtime_and_adapter_can_be_loaded_again(self):
        runtime, state = fake_runtime()
        adapter = PI05PolicyAdapter(runtime=runtime, n_action_steps=2)
        adapter.load()

        adapter.close()
        adapter.load()

        self.assertEqual(state["policy_instances"], 2)
        self.assertEqual(state["closed_policies"], 1)

    def test_defaults_target_requested_v044_checkpoint(self):
        adapter = PI05PolicyAdapter(runtime=fake_runtime()[0])

        self.assertEqual(adapter.checkpoint, DEFAULT_PI05_CHECKPOINT)
        self.assertEqual(adapter.checkpoint, "lerobot/pi05_libero_finetuned_v044")
        self.assertEqual(adapter.n_action_steps, 10)
        self.assertEqual(adapter.num_inference_steps, 10)

    def test_configured_profile_captures_semantic_activation_in_policy_context(self):
        runtime, _state = fake_runtime()
        recorder = FakeEventRecorder()
        adapter = PI05PolicyAdapter(
            checkpoint="org/pi05-test",
            device="cpu",
            n_action_steps=2,
            runtime=runtime,
        )
        with tempfile.TemporaryDirectory() as td:
            profile_path = Path(td) / "capture.yaml"
            profile_path.write_text(
                """
format: robotprobe.runtime.capture_profile.v0
id: policy-adapter-test
storage: summary
max_bytes_per_run: 4096
every_n_policy_calls: 1
semantic_roles:
  - action_output_projection
user_selectors: []
claim_boundary:
  summaries_are_observational: true
  no_causal_claims: true
""".lstrip(),
                encoding="utf-8",
            )
            adapter.configure_capture(
                profile_path=profile_path,
                recorder=recorder,
                storage="summary",
                output_dir=None,
                max_bytes_per_run=None,
            )
            adapter.load()
            adapter.reset(
                ScenarioSpec(
                    scenario_id="libero-task-0",
                    instruction="Pick up the black bowl.",
                    seed=7,
                    max_steps=20,
                )
            )
            context = InferenceContext(
                env_step=3,
                policy_call_id="policy:000004",
                action_chunk_id="chunk:000004",
                observation_id="observation:000003",
            )

            adapter.predict(canonical_observation(), context)

        activation = next(
            event
            for event in recorder.events
            if event["event_type"] == "model_activation"
        )
        self.assertEqual(activation["policy_call_id"], "policy:000004")
        self.assertEqual(activation["action_chunk_id"], "chunk:000004")
        self.assertEqual(activation["observation_id"], "observation:000003")
        self.assertEqual(activation["step"], 3)
        self.assertEqual(activation["module"], "model.action_out_proj")
        self.assertEqual(activation["probe_id"], "pi05_action_output_projection")
        self.assertEqual(activation["denoise_step"], 0)
        self.assertTrue(
            activation["model_ref"]["capture_profile_sha256"].startswith("sha256:")
        )


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def permute(self, *axes):
        return FakeTensor(np.transpose(self.value, axes))

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.value, axis))

    def float(self):
        return FakeTensor(self.value.astype(np.float32))

    def __truediv__(self, value):
        return FakeTensor(self.value / value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value, copy=True)


class FakeInferenceMode(AbstractContextManager):
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self.log.append("inference.enter")
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.log.append("inference.exit")


class FakeTorch:
    float32 = np.float32

    def __init__(self, log):
        self.log = log

    def as_tensor(self, value, dtype=None):
        return FakeTensor(np.asarray(value, dtype=dtype))

    def inference_mode(self):
        return FakeInferenceMode(self.log)


class FakeConfig:
    def __init__(self):
        self.compile_model = True
        self.compile_mode = "max-autotune"
        self.device = None
        self.num_inference_steps = 10
        self.n_action_steps = 50
        self.chunk_size = 4


class FakePolicy:
    def __init__(self, config, state, log, chunk_length):
        self.config = config
        self.state = state
        self.log = log
        self.chunk_length = chunk_length
        self.to_device = None
        self.eval_called = False
        self.reset_calls = 0
        self.action_out_projection = FakeHookModule()

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, batch):
        self.log.append("policy.predict_action_chunk")
        self.state["policy_batch"] = batch
        self.action_out_projection.run(
            FakeTensor(np.ones((1, self.chunk_length, 7), dtype=np.float32))
        )
        values = np.arange(self.chunk_length * 7, dtype=np.float32).reshape(
            1,
            self.chunk_length,
            7,
        )
        return FakeTensor(values)

    def named_modules(self):
        return [
            ("", self),
            ("model.action_out_proj", self.action_out_projection),
        ]

    def close(self):
        self.state["closed_policies"] += 1


class FakeEnvironmentProcessor:
    def __init__(self, state, log):
        self.state = state
        self.log = log

    def observation(self, batch):
        self.log.append("environment")
        self.state["environment_input"] = batch
        robot_state = batch["observation.robot_state"]
        state = np.concatenate(
            (
                robot_state["eef"]["pos"].value,
                np.zeros((1, 3), dtype=np.float32),
                robot_state["gripper"]["qpos"].value,
            ),
            axis=-1,
        )
        processed = dict(batch)
        processed.pop("observation.robot_state")
        processed["observation.state"] = FakeTensor(state)
        processed["environment_processed"] = True
        return processed


class FakeCollector:
    def __init__(self, log):
        self.log = log
        self.model = None
        self.context = None

    def session(self, model, context):
        self.model = model
        self.context = context
        return CaptureContext(self.log)


class FakeHookHandle:
    def __init__(self, module, hook):
        self.module = module
        self.hook = hook

    def remove(self):
        if self.hook in self.module.hooks:
            self.module.hooks.remove(self.hook)


class FakeHookModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHookHandle(self, hook)

    def run(self, output):
        for hook in list(self.hooks):
            hook(self, (), output)
        return output


class FakeEventRecorder:
    def __init__(self):
        self.events = []

    def event(self, step, event_type, layer, **fields):
        self.events.append(
            {
                "step": step,
                "event_type": event_type,
                "layer": layer,
                **fields,
            }
        )


class CaptureContext(AbstractContextManager):
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self.log.append("capture.enter")
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.log.append("capture.exit")


def fake_runtime(*, log=None, chunk_length=4, with_tokenizer_loader=False):
    log = log if log is not None else []
    state = {"closed_policies": 0, "policy_instances": 0}
    torch = FakeTorch(log)

    def config_loader(checkpoint, **kwargs):
        state["config_checkpoint"] = checkpoint
        state["config_load_options"] = kwargs
        return FakeConfig()

    class PolicyClass:
        @classmethod
        def from_pretrained(cls, checkpoint, *, config, **kwargs):
            state["policy_checkpoint"] = checkpoint
            state["policy_load_options"] = kwargs
            state["policy_config_snapshot"] = {
                "compile_model": config.compile_model,
                "compile_mode": config.compile_mode,
                "device": config.device,
                "num_inference_steps": config.num_inference_steps,
                "n_action_steps": config.n_action_steps,
            }
            policy = FakePolicy(config, state, log, chunk_length)
            state["policy"] = policy
            state["policy_instances"] += 1
            return policy

    def processor_factory(config, checkpoint, **kwargs):
        state["processor_args"] = {
            "config": config,
            "checkpoint": checkpoint,
            **kwargs,
        }

        def preprocessor(batch):
            log.append("preprocessor")
            state["preprocessor_input"] = batch
            output = {**batch, "policy_processed": True}
            state["preprocessor_output"] = output
            return output

        def postprocessor(actions):
            log.append("postprocessor")
            state["postprocessor_input"] = actions
            return actions

        return preprocessor, postprocessor

    def tokenizer_loader(checkpoint, **kwargs):
        tokenizer = object()
        state["tokenizer_checkpoint"] = checkpoint
        state["tokenizer_load_options"] = kwargs
        state["tokenizer"] = tokenizer
        return tokenizer

    return (
        Pi05RuntimeBindings(
            torch=torch,
            config_loader=config_loader,
            policy_class=PolicyClass,
            processor_factory=processor_factory,
            environment_processor_factory=lambda: FakeEnvironmentProcessor(state, log),
            tokenizer_loader=tokenizer_loader if with_tokenizer_loader else None,
        ),
        state,
    )


def canonical_observation():
    return CanonicalObservation(
        observation_id="observation:000003",
        sequence=3,
        captured_at_ns=10_000,
        images={
            "agent": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
            "wrist": np.full((2, 3, 3), 5, dtype=np.uint8),
        },
        eef_position=[0.4, 0.0, 0.3],
        eef_quaternion=[0.0, 0.0, 0.0, 1.0],
        gripper_qpos=[0.02, 0.02],
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        joint_efforts=np.zeros(7),
        instruction="Pick up the black bowl.",
        source="libero",
        frame_id="franka_link0",
        calibration_id="libero-v1",
    )


if __name__ == "__main__":
    unittest.main()
