from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WhiteBoxActivationAdapter:
    target: str

    def capture_pair(
        self,
        model: Any,
        clean_episode: Any,
        perturbed_episode: Any,
        capture_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError("white-box activation adapters must implement capture_pair")
