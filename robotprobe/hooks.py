from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class HookContext:
    step: int
    run_id: str | None = None
    scenario: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


HookFn = Callable[[HookContext, Any], Any]


class HookRegistry:
    def __init__(self):
        self._hooks: dict[str, list[tuple[int, int, HookFn]]] = {}
        self._insert_counter = 0

    def hook(self, name: str, *, order: int = 100) -> Callable[[HookFn], HookFn]:
        def decorator(fn: HookFn) -> HookFn:
            self.register(name, fn, order=order)
            return fn

        return decorator

    def register(self, name: str, fn: HookFn, *, order: int = 100) -> None:
        self._insert_counter += 1
        self._hooks.setdefault(name, []).append((order, self._insert_counter, fn))

    def apply(self, name: str, value: Any, ctx: HookContext) -> Any:
        current = value
        for _order, _inserted_at, fn in sorted(self._hooks.get(name, [])):
            current = fn(ctx, current)
        return current
