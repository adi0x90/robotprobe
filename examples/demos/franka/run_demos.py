#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from robotprobe.runtime.config import inspect_runtime_config, run_runtime_config
from robotprobe.runtime.evidence import create_runtime_evidence_bundle


ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "examples" / "packs" / "franka-pi05-runtime"
DETERMINISTIC_CONFIG = PACK / "configs" / "deterministic.yaml"
REPLAY_CONFIG = PACK / "configs" / "franka-recorded-replay.yaml"
PI05_LIBERO_CONFIG = PACK / "configs" / "pi05-libero.yaml"
FRANKA_SHADOW_CONFIG = PACK / "configs" / "franka-ros2-shadow.yaml"


class DemoSuite:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self._runs: dict[str, dict[str, Any]] = {}

    def safety_gate(self) -> dict[str, Any]:
        safe = self._deterministic_run("safe", step_size=0.02)
        rejected = self._deterministic_run(
            "rejected",
            step_size=0.2,
            baseline=safe,
        )
        safe_evidence = _load_json(safe["evidence_path"])
        evidence = _load_json(rejected["evidence_path"])
        motion = evidence["motion"]
        task = evidence["task_outcome"]
        if (
            safe_evidence["task_outcome"]["status"] != "succeeded"
            or safe_evidence["motion"]["disposition"] != "applied"
            or evidence["runtime_integrity"]["status"] != "passed"
            or motion["disposition"] != "rejected_pre_backend"
            or motion["execution_receipts"] != 0
            or motion["applied_commands"] != 0
            or task["reason"] != "safety_rejected:action_limit_exceeded"
        ):
            raise RuntimeError(
                "safety-gate evidence did not prove pre-backend rejection"
            )
        demo_dir = self.out_dir / "demos"
        demo_dir.mkdir(parents=True, exist_ok=True)
        report_path = demo_dir / "safety-gate.md"
        report_path.write_text(
            _safety_gate_report(safe, rejected, evidence),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "report_ref": report_path.relative_to(self.out_dir).as_posix(),
            "safe_run_ref": "runs/safe",
            "rejected_run_ref": "runs/rejected",
            "proves": "unsafe action rejected before backend execution",
        }

    def backend_swap(self) -> dict[str, Any]:
        deterministic = self._deterministic_run("safe", step_size=0.02)
        replay = self._replay_run()
        deterministic_evidence = _load_json(
            deterministic["evidence_path"]
        )
        replay_evidence = _load_json(replay["evidence_path"])
        inspections = {
            "deterministic": inspect_runtime_config(
                DETERMINISTIC_CONFIG
            ),
            "recorded_replay": inspect_runtime_config(REPLAY_CONFIG),
            "pi05_libero": inspect_runtime_config(PI05_LIBERO_CONFIG),
            "franka_ros2_shadow": inspect_runtime_config(
                FRANKA_SHADOW_CONFIG
            ),
        }
        matrix = {
            "format": "robotprobe.franka.backend_matrix.v0",
            "deterministic": {
                "status": "completed",
                "execution_mode": "simulation",
                "task_success": deterministic["result"]["outcome"]["success"],
                "applied_commands": deterministic_evidence["motion"][
                    "applied_commands"
                ],
                "evidence_boundary": "deterministic kinematic contract",
            },
            "recorded_replay": {
                "status": "completed",
                "execution_mode": "simulation",
                "task_success": replay["result"]["outcome"]["success"],
                "applied_commands": replay_evidence["motion"][
                    "applied_commands"
                ],
                "evidence_boundary": (
                    "synthetic recorded-format read-only replay"
                ),
            },
            "pi05_libero": {
                "status": "inspect_only",
                "execution_mode": "simulation",
                "static_valid": inspections["pi05_libero"]["valid"],
                "evidence_boundary": (
                    "optional LIBERO execution not run by CPU demo"
                ),
            },
            "franka_ros2_shadow": {
                "status": "inspect_only",
                "execution_mode": "hardware_shadow",
                "static_valid": inspections["franka_ros2_shadow"]["valid"],
                "evidence_boundary": (
                    "site transport and physical observation not run"
                ),
            },
            "all_static_configs_valid": all(
                inspection["valid"] for inspection in inspections.values()
            ),
        }
        if (
            deterministic_evidence["task_outcome"]["status"] != "succeeded"
            or deterministic_evidence["motion"]["disposition"] != "applied"
            or replay_evidence["runtime_integrity"]["status"] != "passed"
            or replay_evidence["motion"]["applied_commands"] != 0
            or not matrix["all_static_configs_valid"]
        ):
            raise RuntimeError("backend-swap evidence invariants failed")
        demo_dir = self.out_dir / "demos"
        demo_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = demo_dir / "backend-matrix.json"
        matrix_path.write_text(_canonical_json(matrix), encoding="utf-8")
        report_path = demo_dir / "backend-swap.md"
        report_path.write_text(
            _backend_swap_report(matrix),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "report_ref": report_path.relative_to(self.out_dir).as_posix(),
            "matrix_ref": matrix_path.relative_to(self.out_dir).as_posix(),
            "deterministic_run_ref": "runs/safe",
            "recorded_replay_run_ref": "runs/replay",
        }

    def action_journey(self) -> dict[str, Any]:
        safe = self._deterministic_run("safe", step_size=0.02)
        evidence = _load_json(safe["evidence_path"])
        if (
            evidence["task_outcome"]["status"] != "succeeded"
            or evidence["motion"]["disposition"] != "applied"
        ):
            raise RuntimeError(
                "action-journey source did not complete the safe contract run"
            )
        timeline_path = (
            safe["evidence_path"].parent / "timeline.md"
        )
        timeline = timeline_path.read_text(encoding="utf-8")
        demo_dir = self.out_dir / "demos"
        demo_dir.mkdir(parents=True, exist_ok=True)
        report_path = demo_dir / "action-journey.md"
        report_path.write_text(
            "\n".join(
                [
                    "# Follow One Franka Action",
                    "",
                    "This view follows stable observation, policy-call, "
                    "action-chunk, and command identifiers through the "
                    "RobotProbe runtime.",
                    "",
                    timeline.rstrip(),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "report_ref": report_path.relative_to(self.out_dir).as_posix(),
            "source_run_ref": "runs/safe",
        }

    def _deterministic_run(
        self,
        name: str,
        *,
        step_size: float,
        baseline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if name in self._runs:
            return self._runs[name]
        run_dir = self.out_dir / "runs" / name
        run_dir.mkdir(parents=True)
        trace_path = run_dir / "runtime.jsonl"
        config_path = run_dir / "runtime.yaml"
        data = yaml.safe_load(
            DETERMINISTIC_CONFIG.read_text(encoding="utf-8")
        )
        data["pack"] = str(PACK)
        data["components"]["policy"]["options"]["step_size"] = step_size
        data["components"]["scenario"]["options"]["target_position"] = [
            0.5,
            0.0,
            0.3,
        ]
        data["trace"]["out"] = str(trace_path)
        data["trace"]["run_id"] = f"runtime:franka-demo:{name}"
        config_path.write_text(
            yaml.safe_dump(data, sort_keys=False),
            encoding="utf-8",
        )
        result = run_runtime_config(config_path)
        (run_dir / "run-result.json").write_text(
            _canonical_json(result),
            encoding="utf-8",
        )
        evidence_dir = run_dir / "evidence"
        create_runtime_evidence_bundle(
            trace_path=trace_path,
            config_path=config_path,
            baseline_path=(
                baseline["trace_path"] if baseline is not None else None
            ),
            out_dir=evidence_dir,
        )
        evidence_path = evidence_dir / "evidence.json"
        _require_run_integrity(
            name,
            result,
            _load_json(evidence_path),
        )
        run = {
            "name": name,
            "config_path": config_path,
            "trace_path": trace_path,
            "evidence_path": evidence_path,
            "result": result,
        }
        self._runs[name] = run
        return run

    def _replay_run(self) -> dict[str, Any]:
        name = "replay"
        if name in self._runs:
            return self._runs[name]
        run_dir = self.out_dir / "runs" / name
        run_dir.mkdir(parents=True)
        trace_path = run_dir / "runtime.jsonl"
        config_path = run_dir / "runtime.yaml"
        data = yaml.safe_load(REPLAY_CONFIG.read_text(encoding="utf-8"))
        data["pack"] = str(PACK)
        data["trace"]["out"] = str(trace_path)
        data["trace"]["run_id"] = "runtime:franka-demo:replay"
        config_path.write_text(
            yaml.safe_dump(data, sort_keys=False),
            encoding="utf-8",
        )
        result = run_runtime_config(config_path)
        (run_dir / "run-result.json").write_text(
            _canonical_json(result),
            encoding="utf-8",
        )
        evidence_dir = run_dir / "evidence"
        create_runtime_evidence_bundle(
            trace_path=trace_path,
            config_path=config_path,
            out_dir=evidence_dir,
        )
        evidence_path = evidence_dir / "evidence.json"
        _require_run_integrity(
            name,
            result,
            _load_json(evidence_path),
        )
        run = {
            "name": name,
            "config_path": config_path,
            "trace_path": trace_path,
            "evidence_path": evidence_path,
            "result": result,
        }
        self._runs[name] = run
        return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run RobotProbe Franka evidence demos.",
    )
    parser.add_argument(
        "--demo",
        required=True,
        choices=("safety-gate", "backend-swap", "action-journey", "all"),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error(f"--out must be empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_runner = DemoSuite(out_dir)
    demos: dict[str, Any] = {}
    if args.demo in {"safety-gate", "all"}:
        demos["safety-gate"] = suite_runner.safety_gate()
    if args.demo in {"backend-swap", "all"}:
        demos["backend-swap"] = suite_runner.backend_swap()
    if args.demo in {"action-journey", "all"}:
        demos["action-journey"] = suite_runner.action_journey()
    if args.demo not in {
        "safety-gate",
        "backend-swap",
        "action-journey",
        "all",
    }:
        raise NotImplementedError(f"demo is not implemented yet: {args.demo}")

    suite = {
        "format": "robotprobe.franka.demo_suite.v0",
        "requested_demo": args.demo,
        "demos": demos,
    }
    (out_dir / "suite.json").write_text(
        _canonical_json(suite),
        encoding="utf-8",
    )
    print(_canonical_json(suite), end="")
    return 0


def _safety_gate_report(
    safe: dict[str, Any],
    rejected: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    comparison = evidence["comparison"]
    return "\n".join(
        [
            "# Franka Safety Gate Demo",
            "",
            "A safe deterministic action reaches the synthetic kinematic backend. "
            "An oversized action is rejected by the shared safety gateway before "
            "backend execution.",
            "",
            f"- Safe task success: `{str(safe['result']['outcome']['success']).lower()}`",
            f"- Rejected task success: `{str(rejected['result']['outcome']['success']).lower()}`",
            f"- Rejection reason: `{evidence['task_outcome']['reason']}`",
            f"- First action divergence step: `{comparison['first_action_divergence_step']}`",
            f"- Candidate execution receipts: `{evidence['motion']['execution_receipts']}`",
            f"- Candidate applied commands: `{evidence['motion']['applied_commands']}`",
            "",
            "## Evidence Boundary",
            "",
            "This is a deterministic kinematic contract demonstration. It does not "
            "prove learned-model behavior, contact physics, or physical Franka safety.",
            "",
        ]
    )


def _backend_swap_report(matrix: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# One Runtime, Multiple Franka Worlds",
            "",
            "The same RobotProbe runner, Franka embodiment, command space, "
            "safety gateway, and evidence schema are retained while the "
            "backend changes through configuration.",
            "",
            "| Backend | Status | Applied commands | Evidence boundary |",
            "| --- | --- | ---: | --- |",
            (
                "| Deterministic kinematic | "
                f"{matrix['deterministic']['status']} | "
                f"{matrix['deterministic']['applied_commands']} | "
                f"{matrix['deterministic']['evidence_boundary']} |"
            ),
            (
                "| Recorded replay | "
                f"{matrix['recorded_replay']['status']} | "
                f"{matrix['recorded_replay']['applied_commands']} | "
                f"{matrix['recorded_replay']['evidence_boundary']} |"
            ),
            (
                "| PI0.5/LIBERO | "
                f"{matrix['pi05_libero']['status']} | n/a | "
                f"{matrix['pi05_libero']['evidence_boundary']} |"
            ),
            (
                "| Franka ROS 2 shadow | "
                f"{matrix['franka_ros2_shadow']['status']} | n/a | "
                f"{matrix['franka_ros2_shadow']['evidence_boundary']} |"
            ),
            "",
            "Static inspection proves declared compatibility only. It is not "
            "runtime, simulator, or physical evidence.",
            "",
        ]
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_run_integrity(
    name: str,
    result: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    if result.get("trace_valid") is not True:
        raise RuntimeError(f"{name} trace validation failed")
    if evidence.get("runtime_integrity", {}).get("status") != "passed":
        raise RuntimeError(f"{name} runtime integrity failed")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
