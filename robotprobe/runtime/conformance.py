from __future__ import annotations

from dataclasses import dataclass

from robotprobe.runtime.capabilities import (
    RuntimeComponentCapabilities,
    RuntimeComponentKind,
)
from robotprobe.runtime.types import (
    AuthorizedAction,
    ExecutionReceipt,
    RawObservation,
    ScenarioSpec,
)


@dataclass(frozen=True)
class BackendConformanceReport:
    passed: bool
    checks: tuple[str, ...]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": list(self.checks),
            "issues": list(self.issues),
        }


def assert_backend_conforms(
    *,
    backend: object,
    scenario: ScenarioSpec,
    action: AuthorizedAction,
) -> BackendConformanceReport:
    """Exercise the backend lifecycle and raise when a public contract is broken."""
    checks: list[str] = []
    issues: list[str] = []
    capabilities = getattr(backend, "capabilities", None)
    if not isinstance(capabilities, RuntimeComponentCapabilities):
        issues.append("capabilities must be RuntimeComponentCapabilities")
    elif capabilities.kind is not RuntimeComponentKind.BACKEND:
        issues.append("capabilities.kind must be backend")
    else:
        checks.append("capabilities")

    try:
        try:
            backend.observe()
        except (RuntimeError, ValueError):
            checks.append("pre_connect_guard")
        else:
            issues.append("observe must fail before connect and reset")

        backend.connect()
        health = backend.health()
        if not health.connected or not health.healthy:
            issues.append("health must report connected and healthy after connect")
        else:
            checks.append("connected_health")

        initial = backend.reset(scenario)
        if not isinstance(initial, RawObservation):
            issues.append("reset must return RawObservation")
        else:
            checks.append("reset_observation")

        observed = backend.observe()
        if not isinstance(observed, RawObservation):
            issues.append("observe must return RawObservation")
        else:
            checks.append("observation")

        receipt = backend.execute(action)
        if not isinstance(receipt, ExecutionReceipt):
            issues.append("execute must return ExecutionReceipt")
        elif receipt.command_id != action.command_id:
            issues.append("execution receipt must preserve command_id")
        else:
            checks.append("execution_receipt")
    except Exception as error:
        issues.append(f"backend lifecycle raised {type(error).__name__}")
    finally:
        try:
            backend.close()
            checks.append("close")
        except Exception as error:
            issues.append(f"close raised {type(error).__name__}")

    try:
        if backend.health().connected:
            issues.append("health must report disconnected after close")
        else:
            checks.append("closed_health")
    except Exception as error:
        issues.append(f"post-close health raised {type(error).__name__}")

    report = BackendConformanceReport(
        passed=not issues,
        checks=tuple(checks),
        issues=tuple(issues),
    )
    if issues:
        raise AssertionError("; ".join(issues))
    return report
