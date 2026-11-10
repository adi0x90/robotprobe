from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChallengeSpec:
    id: str
    diagnosis: dict[str, Any]
    required_evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def list_builtin_challenges() -> list[ChallengeSpec]:
    return [
        _challenge_from_json(path)
        for path in sorted(_challenge_dir().glob("*.json"))
    ]


def load_builtin_challenge(challenge_id: str) -> ChallengeSpec:
    for challenge in list_builtin_challenges():
        if challenge.id == challenge_id:
            return challenge
    raise KeyError(f"unknown built-in challenge: {challenge_id}")


def _challenge_from_json(path: Path) -> ChallengeSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ChallengeSpec(
        id=data["id"],
        diagnosis=data["diagnosis"],
        required_evidence=data.get("required_evidence", {}),
        metadata=data.get("metadata", {}),
    )


def _challenge_dir() -> Path:
    return Path(__file__).resolve().parent / "challenges"
