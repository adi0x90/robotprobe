# RobotProbe Failure-Semantics Conformance Profile

Spec ID: `robotprobe.failure-semantics/v0` · Status: draft v0 · Last updated: 2026-06-22.

This is a versioned conformance profile for incident traces: the "MCAP of failure semantics."
A RobotProbe JSONL trace is a *well-formed log* once it passes
`robotprobe validate-trace`. This profile defines the additional, named, versioned guarantees a
trace must carry to be a **reproducible, contract-checkable incident** — the property that lets a
second engineer (or another tool) replay it, localize the fault, and audit the verdict.

Conformance is checked with:

```bash
robotprobe conformance trace.jsonl              # report the highest level achieved
robotprobe conformance trace.jsonl --level L1   # exit non-zero unless conformance L1 (`L1_REPRODUCIBLE_INCIDENT`) or higher is met
```

The validator emits a machine-readable JSON report (`spec`, `achieved_level`, `conformant`, and a
per-requirement `checks` list) and exits `0` when conformant.

## Why a profile and not just schema validation

`validate-trace` answers "is this a syntactically valid trace?". This profile answers three
questions: *can I reproduce this failure deterministically, can I point to where and when it
entered the stack, and is the safety/spec verdict auditable rather than asserted?* Those gaps are
named in recent VLA evaluation work (per-subgoal localization in StepEval) and in deterministic
simulation testing (Antithesis DST). Freezing them as a named, versioned level
is what turns RobotProbe from a tool into a citeable standard.

## Levels — the conformance taxonomy

This document defines **conformance levels**: properties of a trace's self-description, checked by
`robotprobe conformance` (`robotprobe/conformance.py`). They are a separate taxonomy from the
**replay levels** (`ReplayLevel` in `robotprobe/core.py`: `L0_TRACE_PLAYBACK`,
`L1_POLICY_REPLAY`, `L2_SIM_REPLAY`, `L3_SHADOW_HARDWARE_REPLAY`), which describe how far a
recorded trace can be re-executed. The numbers are unrelated: a conformance-`L0_WELL_FORMED`
trace may declare replay level `L1_POLICY_REPLAY`.

Conformance levels are ordered and cumulative: `L2_CONTRACT_CHECKABLE` implies
`L1_REPRODUCIBLE_INCIDENT` implies `L0_WELL_FORMED`. The validator reports the
highest **contiguous** level satisfied.

### L0 — `L0_WELL_FORMED`
The trace is a valid RobotProbe log.

| Requirement | Meaning |
|-------------|---------|
| `well_formed_trace` | Passes `validate-trace` with no schema errors. |
| `single_run_id` | Exactly one `run_id` across all events (one episode per trace). |
| `replay_level_declared` | At least one declared, known `replay_level` (a *replay-level* value: `L0_TRACE_PLAYBACK` through `L3_SHADOW_HARDWARE_REPLAY`). |

### L1 — `L1_REPRODUCIBLE_INCIDENT`
The trace carries what is needed to deterministically replay and time-travel the episode.

| Requirement | Meaning |
|-------------|---------|
| `run_seed_present` | An integer `seed` is recorded in the run context (the entropy anchor for deterministic replay; see `TraceRecorder(..., seed=...)`). |
| `provenance_pinned` | The perturbation is pinned: either `intervention_manifest_hash` is present, or `scenario == "clean"` (a clean baseline). |
| `terminal_outcome_present` | At least one `outcome` event — the incident has a terminal verdict to diff against. |
| `monotonic_steps` | `step` indices are non-decreasing, so reverse-step / first-divergence is well-defined. |

### L2 — `L2_CONTRACT_CHECKABLE`
The incident carries the spec it is judged against, so a violation is auditable.

| Requirement | Meaning |
|-------------|---------|
| `contract_bound` | At least one `contract_check` event, or a non-empty declared `contracts` list in the run context. |

## Producing a conformant trace

`TraceRecorder` accepts `seed`. A clean baseline plus a manifest-pinned perturbed run, each
with a seed, an outcome, and (for conformance `L2_CONTRACT_CHECKABLE`) a `contract_check`, will reach the corresponding level:

```python
from robotprobe import ReplayLevel, TraceRecorder

with TraceRecorder(
    "perturbed.jsonl",
    run_id="rp_run_001",
    target="my-robot",
    scenario="gripper_freeze_mid",
    intervention_manifest_ref="interventions/gripper_freeze_mid.yaml",  # -> provenance_pinned
    seed=7,                                                             # -> run_seed_present
    replay_level=ReplayLevel.TRACE_PLAYBACK,
) as recorder:
    recorder.event(35, "postprocessed_action", "action_postprocess", value=[0.1, 0.2, 0.0])
    recorder.contract_check(                                            # -> contract_bound (conformance L2)
        step=35, contract="ActionSpaceContract", check="gripper_freshness",
        status="violated", severity=0.7, evidence_refs=["actions/post/000035.npy"],
    )
    recorder.event(153, "outcome", "outcome", success=True, steps=153)  # -> terminal_outcome
```

## Versioning and compatibility

- The spec ID is `robotprobe.failure-semantics/v<N>`. Requirements are only *added* within a major
  version; tightening or removing a requirement bumps `v<N>`.
- v0 is intentionally minimal. Planned v1 candidates (do not assume present): determinism manifest
  block (recorded entropy sources beyond a single seed), bundle-level conformance across paired
  clean/perturbed traces, evidence-ref resolvability, and an embodiment/units descriptor.

## Non-claims

Conformance is a statement about the **trace's self-description**, not about model robustness,
real-hardware behavior, or the correctness of any diagnosis. An `L2_CONTRACT_CHECKABLE` trace is auditable; it is not
"safe." See [Claims](status-and-claims.md#claims).
