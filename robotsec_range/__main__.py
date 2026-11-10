from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from robotsec_range.challenge_spec import ChallengeSpec, list_builtin_challenges, load_builtin_challenge
from robotsec_range.checker import check_summary
from robotsec_range.mech_checker import check_mech_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotsec-range")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Validate a causal summary against a challenge.")
    check.add_argument("--challenge", required=True)
    check.add_argument("--summary", required=True)

    check_mech = sub.add_parser("check-mech", help="Validate a MechProbe report against a mech-forensics challenge.")
    check_mech.add_argument("--challenge", required=True)
    check_mech.add_argument("--report", required=True)

    sub.add_parser("list-challenges", help="List built-in RobotSec challenges.")

    args = parser.parse_args(argv)
    if args.command == "check":
        challenge_data = _load_challenge(args.challenge)
        summary = _load_json(args.summary)
        challenge = (challenge_data if isinstance(challenge_data, ChallengeSpec)
                     else ChallengeSpec(
                         id=challenge_data["id"],
                         diagnosis=challenge_data["diagnosis"],
                         required_evidence=challenge_data.get("required_evidence", {}),
                     ))
        result = check_summary(challenge, summary)
        json.dump(dataclasses.asdict(result), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if result.passed else 1
    if args.command == "check-mech":
        challenge_data = _load_challenge(args.challenge)
        report = _load_json(args.report)
        challenge = (challenge_data if isinstance(challenge_data, ChallengeSpec)
                     else ChallengeSpec(
                         id=challenge_data["id"],
                         diagnosis=challenge_data.get("diagnosis", {}),
                         required_evidence=challenge_data.get("required_evidence", {}),
                     ))
        result = check_mech_report(challenge, report)
        json.dump(dataclasses.asdict(result), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if result.passed else 1
    if args.command == "list-challenges":
        json.dump(
            [dataclasses.asdict(challenge) for challenge in list_builtin_challenges()],
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_challenge(ref: str) -> dict | ChallengeSpec:
    path = Path(ref)
    if path.exists():
        return _load_json(ref)
    return load_builtin_challenge(ref)


if __name__ == "__main__":
    raise SystemExit(main())
