from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

from robotprobe.packs import validate_pack


def run_doctor(
    *,
    pack_path: str | Path | None = None,
    integration: str | None = None,
) -> dict[str, Any]:
    checks = [
        _check(
            "python_version",
            sys.version_info >= (3, 10),
            details={"version": sys.version.split()[0], "required": ">=3.10"},
        ),
        _import_check("robotprobe_import", "robotprobe"),
        _import_check("robotsec_range_import", "robotsec_range"),
        _import_check("mcap_import", "mcap"),
        _optional_check("huggingface_hub", "huggingface_hub"),
        _optional_check("h5py", "h5py"),
        _optional_check("gymnasium", "gymnasium"),
        _optional_check("mujoco", "mujoco"),
        _optional_check("dm_control", "dm_control"),
        _optional_check("pybullet", "pybullet"),
        {
            "name": "hardware_executable_blocked_by_default",
            "status": "pass",
            "details": {
                "allow_flag": "--allow-hardware-executable",
                "default": "blocked",
            },
        },
    ]
    if pack_path is not None:
        checks.append(_pack_check(pack_path))
    if integration is not None:
        checks.append(_integration_check(integration))

    return {
        "format": "robotprobe.doctor.v0",
        "ok": not any(check["status"] == "fail" for check in checks),
        "checks": checks,
    }


def _check(name: str, passed: bool, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "details": details or {},
    }


def _import_check(name: str, module: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    return _check(name, spec is not None, details={"module": module})


def _optional_check(name: str, module: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    return {
        "name": f"optional_{name}_import",
        "status": "pass" if spec is not None else "warn",
        "details": {"module": module, "installed": spec is not None},
    }


def _pack_check(path: str | Path) -> dict[str, Any]:
    report = validate_pack(path)
    return {
        "name": "pack_valid",
        "status": "pass" if report.valid else "fail",
        "details": {
            "id": report.id,
            "version": report.version,
            "path": report.path,
            "manifest_sha256": report.manifest_sha256,
            "issues": [issue.message for issue in report.issues],
        },
    }


def _integration_check(name: str) -> dict[str, Any]:
    normalized = name.lower()
    if normalized == "vla-eval":
        executable = shutil.which("vla-eval")
        return {
            "name": "integration_vla_eval",
            "status": "pass" if executable else "warn",
            "details": {"executable": executable, "hint": "Install AllenAI vla-evaluation-harness for live runner smoke."},
        }
    if normalized == "ros2":
        executable = shutil.which("ros2")
        return {
            "name": "integration_ros2",
            "status": "pass" if executable else "warn",
            "details": {"executable": executable, "hint": "ROS 2 is optional; MCAP ingest can work on exported logs."},
        }
    if normalized in {"isaac", "isaac-lab-arena"}:
        return {
            "name": "integration_isaac_lab_arena",
            "status": "warn",
            "details": {"hint": "Isaac runtime is intentionally not a RobotProbe core dependency."},
        }
    return {
        "name": f"integration_{normalized.replace('-', '_')}",
        "status": "warn",
        "details": {"hint": "Unknown integration name; no specialized doctor check exists."},
    }
