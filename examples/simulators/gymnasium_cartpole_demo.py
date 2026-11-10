from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from robotprobe.integrations.simulators.gymnasium_cartpole import cartpole_angle_policy, run_episode_trace
from robotprobe.bundle import create_incident_bundle


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local Gymnasium CartPole RobotProbe demo.")
    parser.add_argument("--out", default="examples/simulators/out/gymnasium-cartpole")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fault-start-step", type=int, default=20)
    parser.add_argument("--fault-end-step", type=int, default=80)
    args = parser.parse_args(argv)

    try:
        import gymnasium as gym
    except ImportError:
        sys.stderr.write(
            "Gymnasium is not installed. Install it with:\n"
            "  python -m pip install '.[simulators]'\n"
            "or:\n"
            "  python -m pip install 'gymnasium[classic-control]>=0.29'\n"
        )
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "interventions" / "action_flip_discrete.yaml"

    clean_trace = work / "clean.jsonl"
    perturbed_trace = work / "action_flip_discrete.jsonl"
    clean_result = run_episode_trace(
        env=gym.make("CartPole-v1"),
        policy=cartpole_angle_policy,
        trace_path=clean_trace,
        run_id="gym_cartpole_clean",
        scenario="clean",
        target="gymnasium-cartpole-v1",
        seed=args.seed,
        max_steps=args.max_steps,
    )
    perturbed_result = run_episode_trace(
        env=gym.make("CartPole-v1"),
        policy=cartpole_angle_policy,
        trace_path=perturbed_trace,
        run_id="gym_cartpole_action_flip",
        scenario="action_flip_discrete",
        target="gymnasium-cartpole-v1",
        seed=args.seed,
        max_steps=args.max_steps,
        baseline_run_id="gym_cartpole_clean",
        perturbation_group_id="gym_cartpole_pair_action_flip",
        intervention_manifest_ref=manifest,
        fault_start_step=args.fault_start_step,
        fault_end_step=args.fault_end_step,
    )

    bundle = create_incident_bundle(
        clean_trace=clean_trace,
        perturbed_trace=perturbed_trace,
        manifest=manifest,
        challenge="robotprobe-v1-gymnasium-cartpole-action-flip",
        metrics=["action_energy", "jerk", "trajectory_deviation", "recovery_latency", "near_miss"],
        title="Gymnasium CartPole: Discrete Action Flip",
        out_dir=out,
    )

    print(f"RobotProbe Gymnasium demo wrote {out}")
    print(f"clean_steps={clean_result['steps']} perturbed_steps={perturbed_result['steps']}")
    print(f"report={out / bundle['report_ref']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
