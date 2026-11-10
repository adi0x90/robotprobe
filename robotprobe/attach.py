"""Live in-process instrumentation for a Python policy callable.

``PolicyProbe`` wraps a ``policy(observation) -> action`` callable and, on each step:

- **hooks** the inference boundary, streaming typed RobotProbe events (instruction, observation,
  raw_action, postprocessed_action) to a live ``TraceRecorder``;
- **intervenes**, optionally perturbing the input observation and/or the output action in-flight via
  RobotProbe ``Attack`` objects (modify-args / modify-return);
- fires **watchpoints** when a streamed event matches a predicate, invoking a callback so callers
  can log, count, or raise.

Scope: this hooks an *in-process Python* policy. Cross-process attach and live activation patching
of a model runtime are out of scope (both need a running model server and a GPU). The
instrumentation itself is CPU-light and model-agnostic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from robotprobe.attacks import Attack, Observation
from robotprobe.record import TraceRecorder


@dataclass
class Watchpoint:
    """A predicate over streamed trace events; ``on_hit`` is called when it matches."""

    name: str
    event_type: str
    predicate: Callable[[dict[str, Any]], bool]
    on_hit: Callable[[dict[str, Any]], None] | None = None


@dataclass
class WatchpointHit:
    watchpoint: str
    step: int
    event: dict[str, Any]


class PolicyProbe:
    def __init__(
        self,
        policy: Callable[[Observation], Any],
        recorder: TraceRecorder,
        *,
        input_attack: Attack | None = None,
        output_attack: Attack | None = None,
        watchpoints: Iterable[Watchpoint] = (),
    ) -> None:
        self.policy = policy
        self.recorder = recorder
        self.input_attack = input_attack
        self.output_attack = output_attack
        self.watchpoints = list(watchpoints)
        self.hits: list[WatchpointHit] = []

    def step(self, step: int, observation: Observation, instruction: str | None = None) -> np.ndarray:
        obs = observation
        if self.input_attack is not None:
            obs = self.input_attack.perturb(obs, step)

        instr = instruction if instruction is not None else getattr(obs, "instruction", None)
        if instr is not None:
            self._emit(step, "instruction", "instruction", value=instr)
        image = getattr(obs, "image", None)
        if image is not None:
            self._emit(step, "observation", "observation", image_sha256=_image_hash(image))

        raw = np.asarray(self.policy(obs), dtype=float)
        self._emit(step, "raw_action", "policy_forward", value=raw.tolist())

        post = raw
        if self.output_attack is not None:
            post = np.asarray(self.output_attack.perturb_action(raw, step), dtype=float)
        self._emit(step, "postprocessed_action", "action_postprocess", value=post.tolist())
        return post

    def run(self, episode: Iterable[Any]) -> list[np.ndarray]:
        actions: list[np.ndarray] = []
        for step, item in enumerate(episode):
            if isinstance(item, tuple) and len(item) == 2:
                obs, instr = item
            else:
                obs, instr = item, None
            actions.append(self.step(step, obs, instruction=instr))
        return actions

    def _emit(self, step: int, event_type: str, layer: str, **fields: Any) -> None:
        self.recorder.event(step, event_type, layer, **fields)
        event = {"step": step, "event_type": event_type, "layer": layer, **fields}
        for watchpoint in self.watchpoints:
            if event_type == watchpoint.event_type and watchpoint.predicate(event):
                hit = WatchpointHit(watchpoint=watchpoint.name, step=step, event=event)
                self.hits.append(hit)
                if watchpoint.on_hit is not None:
                    watchpoint.on_hit(event)


def _image_hash(image: Any) -> str:
    return "sha256:" + hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
