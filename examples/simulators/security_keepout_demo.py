from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from robotprobe.integrations.simulators.security_keepout import run_keepout_episode
from robotprobe.bundle import create_incident_bundle


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local security demo where binary success hides a keepout violation."
    )
    parser.add_argument("--out", default="examples/simulators/out/security-keepout")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--bias-start-step", type=int, default=10)
    parser.add_argument("--bias-end-step", type=int, default=28)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    manifest = ROOT / "interventions" / "stealth_velocity_bias_keepout.yaml"
    clean_trace = work / "clean.jsonl"
    perturbed_trace = work / "stealth_velocity_bias_keepout.jsonl"

    clean = run_keepout_episode(
        trace_path=clean_trace,
        run_id="security_keepout_clean",
        scenario="clean",
        seed=args.seed,
        max_steps=args.max_steps,
    )
    perturbed = run_keepout_episode(
        trace_path=perturbed_trace,
        run_id="security_keepout_bias",
        scenario="stealth_velocity_bias_keepout",
        seed=args.seed,
        max_steps=args.max_steps,
        baseline_run_id="security_keepout_clean",
        perturbation_group_id="security_keepout_pair_bias",
        intervention_manifest_ref=manifest,
        bias_start_step=args.bias_start_step,
        bias_end_step=args.bias_end_step,
    )

    bundle = create_incident_bundle(
        clean_trace=clean_trace,
        perturbed_trace=perturbed_trace,
        manifest=manifest,
        challenge="robotprobe-v1-security-keepout-bypass",
        metrics=["action_energy", "jerk", "trajectory_deviation", "recovery_latency", "near_miss"],
        title="Security Demo: Hidden Keepout Violation",
        out_dir=out,
    )

    print(f"RobotProbe security demo wrote {out}")
    print(
        "binary_success="
        f"clean:{clean['success']} perturbed:{perturbed['success']} "
        f"keepout_violated=clean:{clean['keepout_violated']} perturbed:{perturbed['keepout_violated']}"
    )
    print(f"report={out / bundle['report_ref']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
