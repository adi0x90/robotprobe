from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Observation:
    image: np.ndarray
    instruction: str
    state: Any = None


class Attack:
    name: str = "base"
    channel: str = "none"

    def reset(self) -> None:
        """Reset any per-episode state."""

    def perturb(self, obs: Observation, step: int = 0) -> Observation:
        raise NotImplementedError

    def perturb_action(self, action, step: int = 0):
        return action


@dataclass
class NoAttack(Attack):
    name: str = "clean"
    channel: str = "none"

    def perturb(self, obs: Observation, step: int = 0) -> Observation:
        return obs


@dataclass
class InstructionOverrideAttack(Attack):
    name: str = "instruction_override"
    channel: str = "language"
    payload: str = ""

    def perturb(self, obs: Observation, step: int = 0) -> Observation:
        new = copy.copy(obs)
        new.instruction = self.payload
        return new


@dataclass
class ActionFaultAttack(Attack):
    name: str = "action_fault"
    channel: str = "actuation"
    mode: str = "zero_dims"
    dims: tuple[int, ...] = (0, 1, 2)
    start: int = 0
    end: Optional[int] = None
    scale: float = 0.0
    value: float = 0.0
    offset: float = 0.0
    probability: float = 1.0
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._held_values = None

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._held_values = None

    def perturb(self, obs: Observation, step: int = 0) -> Observation:
        return obs

    def perturb_action(self, action, step: int = 0):
        arr = np.array(action, copy=True)
        dims = tuple(d for d in self.dims if -arr.shape[-1] <= d < arr.shape[-1])
        if not dims:
            return arr
        if not self._is_active(step):
            if self.mode == "hold_previous_dims":
                self._held_values = arr[..., dims].copy()
            return arr
        if self.probability < 1.0 and float(self._rng.random()) > self.probability:
            return arr

        if self.mode == "random_zero_dim":
            dims = (int(self._rng.choice(dims)),)
            arr[..., dims] = 0.0
        elif self.mode == "zero_dims":
            arr[..., dims] = 0.0
        elif self.mode == "scale_dims":
            arr[..., dims] = arr[..., dims] * self.scale
        elif self.mode == "set_dims":
            arr[..., dims] = self.value
        elif self.mode == "offset_dims":
            arr[..., dims] = arr[..., dims] + self.offset
        elif self.mode == "hold_previous_dims":
            if self._held_values is None:
                self._held_values = arr[..., dims].copy()
            arr[..., dims] = self._held_values
        else:
            raise ValueError(f"unknown action fault mode: {self.mode}")
        return arr

    def _is_active(self, step: int) -> bool:
        if step < self.start:
            return False
        if self.end is not None and step >= self.end:
            return False
        return True


@dataclass
class StaleObservationReplayAttack(Attack):
    name: str = "stale_observation_replay"
    channel: str = "visual"
    start: int = 20
    end: Optional[int] = 80

    def __post_init__(self):
        self._held_image = None

    def reset(self) -> None:
        self._held_image = None

    def perturb(self, obs: Observation, step: int = 0) -> Observation:
        image = np.asarray(obs.image)
        if not self._is_active(step):
            self._held_image = np.array(image, copy=True)
            return obs

        if self._held_image is None:
            self._held_image = np.array(image, copy=True)

        new = copy.copy(obs)
        new.image = np.array(self._held_image, copy=True)
        return new

    def _is_active(self, step: int) -> bool:
        if step < self.start:
            return False
        if self.end is not None and step >= self.end:
            return False
        return True
