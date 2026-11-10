from __future__ import annotations

from typing import Any

from robotprobe.mech.adapters.base import WhiteBoxActivationAdapter


class OpenVLAActivationAdapter(WhiteBoxActivationAdapter):
    """Pack boundary for future OpenVLA/LIBERO white-box activation capture.

    The class is intentionally lightweight: importing it must not import OpenVLA, Torch,
    TensorFlow, vla-eval, robosuite, or LIBERO. Runtime-specific imports belong inside a future
    capture implementation after the caller provides an already-working white-box environment.
    """

    def __init__(self) -> None:
        super().__init__(target="openvla-libero")

    def capture_pair(
        self,
        model: Any,
        clean_episode: Any,
        perturbed_episode: Any,
        capture_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "format": "robotprobe.mech.adapter_capture.v0",
            "target": self.target,
            "status": "not_implemented",
            "reason": "OpenVLA/LIBERO live activation capture requires a running white-box model runtime.",
        }
