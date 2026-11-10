from __future__ import annotations

from collections import deque

from robotprobe.runtime.types import ProposedActionChunk, ScheduledAction


class ActionScheduler:
    def __init__(self, *, n_action_steps: int):
        if n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        self.n_action_steps = n_action_steps
        self._queue: deque[ScheduledAction] = deque()

    @property
    def has_pending(self) -> bool:
        return bool(self._queue)

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def load(self, chunk: ProposedActionChunk) -> int:
        if self._queue:
            raise RuntimeError("cannot load an action chunk while actions are pending")
        count = min(self.n_action_steps, chunk.values.shape[0])
        for action_index in range(count):
            self._queue.append(
                ScheduledAction(
                    command_id=f"{chunk.chunk_id}:action:{action_index:03d}",
                    chunk_id=chunk.chunk_id,
                    policy_call_id=chunk.policy_call_id,
                    source_observation_id=chunk.source_observation_id,
                    source_observation_captured_at_ns=chunk.source_observation_captured_at_ns,
                    action_index=action_index,
                    command_space=chunk.command_space,
                    value=chunk.values[action_index],
                    metadata={"chunk_metadata": dict(chunk.metadata)},
                )
            )
        return count

    def next_action(self) -> ScheduledAction:
        if not self._queue:
            raise RuntimeError("action scheduler queue is empty")
        return self._queue.popleft()

    def reset(self) -> None:
        self._queue.clear()
