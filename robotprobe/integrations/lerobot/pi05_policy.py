from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from robotprobe.integrations.franka.embodiment import FRANKA_CARTESIAN_DELTA_COMMAND_SPACE
from robotprobe.runtime.capabilities import RuntimeComponentCapabilities
from robotprobe.runtime.types import (
    CanonicalObservation,
    InferenceContext,
    ProposedActionChunk,
    ScenarioSpec,
)


DEFAULT_PI05_CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"
DEFAULT_PI05_TOKENIZER_CHECKPOINT = "google/paligemma-3b-pt-224"


class Pi05OptionalDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pi05RuntimeBindings:
    torch: Any
    config_loader: Callable[..., Any]
    policy_class: Any
    processor_factory: Callable[..., tuple[Any, Any]]
    environment_processor_factory: Callable[[], Any]
    tokenizer_loader: Callable[..., Any] | None = None


class PI05PolicyAdapter:
    """Lazy LeRobot PI0.5 adapter with complete action-chunk and capture boundaries."""

    capabilities = RuntimeComponentCapabilities.from_dict(
        {
            "format": "robotprobe.runtime.component.v0",
            "kind": "policy",
            "component_id": "lerobot.pi05_libero.v044",
            "command_spaces": [FRANKA_CARTESIAN_DELTA_COMMAND_SPACE],
            "observation_streams": ["rgb", "eef_pose", "gripper", "joint_state"],
            "runtime_family": "lerobot_pi05",
            "is_hardware": False,
            "supports_reset": False,
            "supports_ground_truth": False,
            "supports_shadow": False,
            "can_apply_hardware_commands": False,
            "requires_reset": False,
            "optional_dependencies": ["torch", "lerobot", "transformers"],
        }
    )

    def __init__(
        self,
        *,
        checkpoint: str = DEFAULT_PI05_CHECKPOINT,
        device: str = "cuda",
        num_inference_steps: int = 10,
        n_action_steps: int = 10,
        local_files_only: bool = False,
        revision: str | None = None,
        cache_dir: str | None = None,
        tokenizer_checkpoint: str = DEFAULT_PI05_TOKENIZER_CHECKPOINT,
        tokenizer_revision: str | None = None,
        image_key_map: Mapping[str, str] | None = None,
        activation_collector: Any | None = None,
        runtime: Pi05RuntimeBindings | None = None,
        runtime_loader: Callable[[], Pi05RuntimeBindings] | None = None,
    ):
        if not checkpoint.strip():
            raise ValueError("checkpoint must be non-empty")
        if not tokenizer_checkpoint.strip():
            raise ValueError("tokenizer_checkpoint must be non-empty")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        if runtime is not None and runtime_loader is not None:
            raise ValueError("configure runtime or runtime_loader, not both")
        self.checkpoint = checkpoint
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.n_action_steps = n_action_steps
        self.local_files_only = local_files_only
        self.revision = revision
        self.cache_dir = cache_dir
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.tokenizer_revision = tokenizer_revision
        self.image_key_map = dict(
            image_key_map
            or {
                "agent": "observation.images.image",
                "wrist": "observation.images.image2",
            }
        )
        if set(self.image_key_map) != {"agent", "wrist"}:
            raise ValueError("PI0.5 LIBERO image_key_map must define agent and wrist roles")
        self.activation_collector = activation_collector
        self._provided_runtime = runtime
        self._runtime_loader = runtime_loader or _load_default_runtime
        self._bindings: Pi05RuntimeBindings | None = None
        self._policy: Any | None = None
        self._preprocessor: Any | None = None
        self._postprocessor: Any | None = None
        self._environment_processor: Any | None = None
        self._scenario: ScenarioSpec | None = None

    @property
    def instrumentation_model(self) -> Any:
        if self._policy is None:
            raise RuntimeError("PI0.5 policy must be loaded before model instrumentation")
        return self._policy

    def set_activation_collector(self, collector: Any | None) -> None:
        if self._policy is not None:
            raise RuntimeError("activation collector must be configured before PI0.5 load")
        self.activation_collector = collector

    def configure_capture(
        self,
        *,
        profile_path: Path,
        recorder: Any,
        storage: str | None,
        output_dir: Path | None,
        max_bytes_per_run: int | None,
    ) -> None:
        if self._policy is not None:
            raise RuntimeError("PI0.5 capture must be configured before load")
        if self.activation_collector is not None:
            raise RuntimeError("PI0.5 activation capture is already configured")
        from robotprobe.integrations.lerobot.pi05_probes import (
            PI05SemanticActivationCapture,
            load_pi05_capture_settings,
        )

        settings = load_pi05_capture_settings(
            profile_path,
            storage_override=storage,
            output_dir=output_dir,
            max_bytes_per_run_override=max_bytes_per_run,
        )
        self.activation_collector = PI05SemanticActivationCapture(
            settings,
            recorder=recorder,
            model_ref={
                "adapter": self.capabilities.component_id,
                "model_name": self.checkpoint,
                "revision": self.revision or "default",
                "tokenizer_name": self.tokenizer_checkpoint,
                "tokenizer_revision": self.tokenizer_revision or "default",
                "device": self.device,
            },
        )

    def load(self) -> None:
        if self._policy is not None:
            return
        try:
            bindings = self._provided_runtime or self._runtime_loader()
            artifact_options: dict[str, Any] = {
                "local_files_only": self.local_files_only,
            }
            if self.revision is not None:
                artifact_options["revision"] = self.revision
            if self.cache_dir is not None:
                artifact_options["cache_dir"] = self.cache_dir
            config = bindings.config_loader(
                self.checkpoint,
                **artifact_options,
            )
            if self.n_action_steps > int(getattr(config, "chunk_size", self.n_action_steps)):
                raise ValueError("n_action_steps cannot exceed the PI0.5 action chunk size")
            config.compile_model = False
            config.compile_mode = "default"
            config.device = self.device
            config.num_inference_steps = self.num_inference_steps
            config.n_action_steps = self.n_action_steps
            policy = bindings.policy_class.from_pretrained(
                self.checkpoint,
                config=config,
                **artifact_options,
            )
            policy = policy.to(self.device).eval()
            preprocessor_overrides: dict[str, Any] = {
                "device_processor": {"device": str(self.device)}
            }
            if bindings.tokenizer_loader is not None:
                tokenizer_options: dict[str, Any] = {
                    "local_files_only": self.local_files_only,
                }
                if self.tokenizer_revision is not None:
                    tokenizer_options["revision"] = self.tokenizer_revision
                if self.cache_dir is not None:
                    tokenizer_options["cache_dir"] = self.cache_dir
                tokenizer = bindings.tokenizer_loader(
                    self.tokenizer_checkpoint,
                    **tokenizer_options,
                )
                preprocessor_overrides["tokenizer_processor"] = {
                    "tokenizer": tokenizer,
                }
            elif self.tokenizer_revision is not None:
                raise RuntimeError(
                    "configured tokenizer_revision requires a tokenizer-aware runtime"
                )
            preprocessor, postprocessor = bindings.processor_factory(
                policy.config,
                self.checkpoint,
                preprocessor_overrides=preprocessor_overrides,
                **artifact_options,
            )
            environment_processor = bindings.environment_processor_factory()
        except ImportError as error:
            self._clear_loaded_state()
            raise Pi05OptionalDependencyError(
                "LeRobot PI0.5 dependencies are unavailable at load(); install "
                "LeRobot with its PI0.5 Torch and Transformers dependencies."
            ) from error
        except Exception:
            self._clear_loaded_state()
            raise

        self._bindings = bindings
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._environment_processor = environment_processor

    def reset(self, scenario: ScenarioSpec) -> None:
        self._require_loaded()
        if not isinstance(scenario, ScenarioSpec):
            raise TypeError("PI0.5 reset requires ScenarioSpec")
        reset_budget = getattr(self.activation_collector, "reset_budget", None)
        if callable(reset_budget):
            reset_budget()
        self._scenario = scenario
        self._policy.reset()

    def predict(
        self,
        observation: CanonicalObservation,
        context: InferenceContext,
    ) -> ProposedActionChunk:
        self._require_loaded()
        if self._scenario is None:
            raise RuntimeError("PI0.5 policy must be reset before prediction")
        if observation.observation_id != context.observation_id:
            raise ValueError("PI0.5 inference context observation lineage mismatch")

        raw_batch = self._canonical_to_libero_batch(observation)
        environment_batch = self._environment_processor.observation(raw_batch)
        policy_batch = self._preprocessor(environment_batch)
        capture_context = (
            self.activation_collector.session(self._policy, context)
            if self.activation_collector is not None
            else nullcontext()
        )
        with capture_context:
            with self._bindings.torch.inference_mode():
                raw_actions = self._policy.predict_action_chunk(policy_batch)
        postprocessed = self._postprocessor(raw_actions)
        action_values = _action_array(postprocessed)
        if action_values.ndim == 3 and action_values.shape[0] == 1:
            action_values = action_values[0]
        if action_values.ndim != 2:
            raise ValueError(
                f"PI0.5 action chunk must have shape [horizon, action_dim], got {action_values.shape}"
            )
        if action_values.shape[1] != 7:
            raise ValueError(
                f"PI0.5 LIBERO action dimension must be seven, got {action_values.shape[1]}"
            )

        return ProposedActionChunk(
            chunk_id=context.action_chunk_id,
            policy_call_id=context.policy_call_id,
            source_observation_id=context.observation_id,
            source_observation_captured_at_ns=observation.captured_at_ns,
            command_space=FRANKA_CARTESIAN_DELTA_COMMAND_SPACE,
            values=action_values,
            metadata={
                "adapter": self.capabilities.component_id,
                "checkpoint": self.checkpoint,
                "revision": self.revision or "default",
                "tokenizer_checkpoint": self.tokenizer_checkpoint,
                "tokenizer_revision": self.tokenizer_revision or "default",
                "device": self.device,
                "num_inference_steps": self.num_inference_steps,
                "configured_n_action_steps": self.n_action_steps,
                "returned_chunk_length": int(action_values.shape[0]),
                "compile_model": False,
                "environment_processor": "libero",
            },
        )

    def close(self) -> None:
        policy = self._policy
        try:
            if policy is not None and callable(getattr(policy, "close", None)):
                policy.close()
        finally:
            self._clear_loaded_state()

    def _canonical_to_libero_batch(
        self,
        observation: CanonicalObservation,
    ) -> dict[str, Any]:
        torch = self._bindings.torch
        batch: dict[str, Any] = {}
        for role, output_key in self.image_key_map.items():
            if role not in observation.images:
                raise ValueError(f"PI0.5 observation is missing {role!r} image role")
            batch[output_key] = _image_tensor(torch, observation.images[role])
        batch["observation.robot_state"] = {
            "eef": {
                "pos": _batched_tensor(torch, observation.eef_position),
                "quat": _batched_tensor(torch, observation.eef_quaternion),
            },
            "gripper": {
                "qpos": _batched_tensor(torch, observation.gripper_qpos),
            },
            "joint": {
                "positions": _batched_tensor(torch, observation.joint_positions),
                "velocities": _batched_tensor(torch, observation.joint_velocities),
                "efforts": _batched_tensor(torch, observation.joint_efforts),
            },
        }
        batch["task"] = [observation.instruction]
        return batch

    def _require_loaded(self) -> None:
        if (
            self._policy is None
            or self._bindings is None
            or self._preprocessor is None
            or self._postprocessor is None
            or self._environment_processor is None
        ):
            raise RuntimeError("PI0.5 policy must be loaded before use")

    def _clear_loaded_state(self) -> None:
        self._bindings = None
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._environment_processor = None
        self._scenario = None


def _load_default_runtime() -> Pi05RuntimeBindings:
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.processor.env_processor import LiberoProcessorStep
    from transformers import AutoTokenizer

    return Pi05RuntimeBindings(
        torch=torch,
        config_loader=PreTrainedConfig.from_pretrained,
        policy_class=PI05Policy,
        processor_factory=_make_revision_aware_processor_factory(
            make_pre_post_processors,
            snapshot_download,
        ),
        environment_processor_factory=LiberoProcessorStep,
        tokenizer_loader=AutoTokenizer.from_pretrained,
    )


def _make_revision_aware_processor_factory(
    processor_factory: Callable[..., tuple[Any, Any]],
    snapshot_downloader: Callable[..., str],
) -> Callable[..., tuple[Any, Any]]:
    def load_processors(
        policy_config: Any,
        checkpoint: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        **processor_options: Any,
    ) -> tuple[Any, Any]:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_dir():
            processor_source = checkpoint
        else:
            processor_source = snapshot_downloader(
                repo_id=checkpoint,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=(
                    "policy_preprocessor*",
                    "policy_postprocessor*",
                ),
            )
        return processor_factory(
            policy_config,
            processor_source,
            **processor_options,
        )

    return load_processors


def _image_tensor(torch: Any, image: np.ndarray) -> Any:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("PI0.5 images must have three dimensions")
    tensor = torch.as_tensor(np.array(array, copy=True))
    if array.shape[-1] in {1, 3, 4}:
        tensor = tensor.permute(2, 0, 1)
    elif array.shape[0] not in {1, 3, 4}:
        raise ValueError("PI0.5 images must be HWC or CHW")
    return tensor.unsqueeze(0).float() / 255.0


def _batched_tensor(torch: Any, value: Any) -> Any:
    if value is None:
        raise ValueError("PI0.5 requires complete Franka joint state")
    return torch.as_tensor(
        np.asarray(value, dtype=np.float32),
        dtype=torch.float32,
    ).unsqueeze(0)


def _action_array(value: Any) -> np.ndarray:
    candidate = value
    if isinstance(candidate, Mapping):
        if "action" not in candidate:
            raise ValueError("PI0.5 postprocessor mapping does not contain action")
        candidate = candidate["action"]
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    array = np.asarray(candidate, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("PI0.5 produced non-finite actions")
    return np.array(array, copy=True)
