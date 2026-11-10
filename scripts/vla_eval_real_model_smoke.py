from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from robotprobe.integrations.vla_eval.model_server_smoke import run_real_model_smoke


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a real AllenAI vla-eval model server config and run a RobotProbe smoke episode pair.",
    )
    parser.add_argument("--model-name", required=True, help="Human-readable model server name for evidence output.")
    parser.add_argument("--server-config", required=True, type=Path, help="Official vla-eval model-server YAML.")
    parser.add_argument(
        "--vla-eval-cwd",
        required=True,
        type=Path,
        help="vla-evaluation-harness checkout root; relative YAML script paths resolve from here.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("interventions/gripper_freeze_mid.yaml"), type=Path)
    parser.add_argument("--vla-eval-executable", default="vla-eval")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--episode-timeout", type=float, default=120.0)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--arg", action="append", default=[], help="Extra vla-eval serve --arg KEY=VALUE override.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment override for the server subprocess, formatted KEY=VALUE.",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Return exit code 0 after writing structured failure evidence.",
    )
    args = parser.parse_args()

    env_overrides = dict(_split_key_value(item, "--env") for item in args.env)
    result = asyncio.run(
        run_real_model_smoke(
            model_name=args.model_name,
            server_config=args.server_config,
            vla_eval_cwd=args.vla_eval_cwd,
            out_dir=args.out,
            manifest=args.manifest,
            vla_eval_executable=args.vla_eval_executable,
            host=args.host,
            port=args.port,
            startup_timeout=args.startup_timeout,
            episode_timeout=args.episode_timeout,
            max_steps=args.max_steps,
            arg_overrides=tuple(args.arg),
            env_overrides=env_overrides,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "passed" or args.allow_failure:
        return 0
    return 2


def _split_key_value(value: str, flag: str) -> tuple[str, str]:
    key, sep, parsed = value.partition("=")
    if not key or not sep:
        raise SystemExit(f"{flag} must be KEY=VALUE, got {value!r}")
    return key, parsed


if __name__ == "__main__":
    raise SystemExit(main())
