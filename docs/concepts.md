# Concepts

Where RobotProbe sits in a robot stack, how a trace is structured, and how a diagnosis is read.

## Where it fits

RobotProbe is one layer in a robot or VLA stack. This page describes where that layer sits so you
can decide whether it is useful to you.

### The seam

RobotProbe consumes recorded behaviour and produces a verdict about it:

```
   simulators / real robots / VLA eval harnesses / model servers
                          │  (logs, MCAP, episodes)
                          ▼
        ┌─────────────────────────────────────────────┐
        │  RobotProbe: trace · diff · contracts ·      │
        │  metrics · challenge packs · MechProbe       │
        └─────────────────────────────────────────────┘
                          │  (sidecar / MCAP bridge)
                          ▼
            Foxglove · Rerun · CI gates · incident reviews
```

It is not a system of record. Your existing observation and visualization stack keeps that job.

### Two directions of integration

**Inbound.** RobotProbe accepts recorded data instead of requiring you to adopt its runtime:

- `ingest ros2-mcap` for ROS 2 bags, including CDR payload decoding.
- `ingest vla-eval` for exported evaluation episodes.
- `ingest isaac-lab-arena` for exported trace files.
- Plain JSONL via `TraceRecorder` or a pack adapter, if your format is none of the above.

**Outbound.** Evidence travels back into tools you already use:

- `export sidecar` writes a single JSON document alongside a bundle.
- `export mcap` / `import mcap` round-trip a bundle over RobotProbe-owned `/robotprobe/...` topics.
- `report` renders a Markdown incident report suitable for review.

See [Observability Bridge](integrations.md#mcap-and-foxglove) for the concrete commands.

### What RobotProbe adds

- Typed stack events instead of one flat telemetry stream: instruction, observation, policy input,
  raw action, postprocessed action, contract check, outcome.
- A clean-versus-perturbed comparison that names the layer where behaviour first diverged.
- Contracts and consequence metrics that can fail while task success stays green.
- Reusable packs that package a failure mode, its contract, and its expected diagnosis together.
- A versioned conformance profile describing what a trace must carry to be auditable.

### What it does not do

- It does not replace MCAP, Foxglove, Rerun, ROS tracing, or OpenTelemetry.
- It does not run models or simulators for you; the adapters read their output.
- It does not produce a success-rate score. It answers "what changed and where", not "how good is
  this policy".
- It is not validated on hardware.

Implementation status is in [Status](status-and-claims.md#status). Claim boundaries are in [Claims](status-and-claims.md#claims).

## Reading an incident

This walkthrough explains how to read a RobotProbe incident bundle and what real-world situation
the local demos are meant to mimic.

The important idea is not that the demo simulator is realistic. The important idea is the incident
shape: a robot policy can appear correct, and the task can still finish, while a deployment-layer
fault changes the command stream or violates a safety contract.

### The failure class

In many robot and humanoid stacks, the model is not the final authority that reaches the motors.
A typical path looks like this:

```text
task text / operator command
  -> camera + proprioception observations
  -> VLA / policy forward pass
  -> action postprocessing
  -> robot middleware / controller adapter
  -> simulator or hardware supervisor
  -> task outcome and safety monitors
```

The action postprocessing layer can include unit conversion, action normalization, gripper
conversion, joint-order mapping, velocity clipping, coordinate-frame transforms, trajectory
chunking, command smoothing, and vendor driver adapters. A failure here can be caused by a normal
bug, a bad config, a stale calibration, a compromised middleware node, an unsafe plugin, or an
operator/tooling mistake.

That is the class of situation RobotProbe is trying to make visible:

- the policy's raw action can be reasonable;
- the downstream command stream can be changed after inference;
- the robot can still complete the task;
- ordinary success dashboards can stay green;
- the important failure is only visible if the trace captures the boundary where the command
  changed and the consequence that followed.

### Quickstart Scenario

The quickstart demo uses a gripper-freeze fault because it is easy to inspect without a simulator.

The clean trace represents a successful robot episode. The perturbed trace represents the same
episode after a post-policy gripper command is frozen. The raw policy action is unchanged, but the
postprocessed action differs and the episode takes more steps.

A real version of this could look like:

- a gripper adapter holds the previous command because of a stale message;
- a driver bridge clips or quantizes the gripper dimension incorrectly;
- a safety wrapper overwrites one action dimension but does not expose that rewrite clearly;
- an attacker or test harness modifies the command stream after the model server returns.

This is deliberately a deployment-boundary failure, not a claim about model internals. If the raw
policy action stays unchanged, the first explanation should not be "the VLA thought the wrong
thing." The first explanation should be "something after the policy changed the command."

### Security Demo Scenario

The hidden keepout demo is closer to a safety/security incident. It models a point robot that has a
goal and a keepout zone. Both the clean and perturbed runs reach the goal, but the perturbed run
crosses the keepout zone after a post-policy velocity bias is injected.

A real version could look like:

- a mobile manipulator cuts through a restricted aisle while still delivering the object;
- a humanoid steps into a human-only zone and still completes the pick/place task;
- a hospital or lab robot reaches the destination but crosses a sterile or exclusion boundary;
- a warehouse robot finishes the job while passing too close to a person, shelf, or charging area;
- a robot arm completes insertion but violates a fixture clearance envelope.

The key lesson is that task success and safety success are different claims. "The task completed"
does not mean "the path was acceptable," "the command stream was untampered," or "the deployment
contract held."

### Trace Layers

RobotProbe traces are organized around layers:

- `instruction` - task text or command intent;
- `observation` - camera, proprioception, or sensor inputs;
- `policy_forward` - raw model action output;
- `action_postprocess` - normalization, scaling, clipping, gripper conversion, chunking;
- `contract_check` - executable checks for assumptions such as action bounds, freshness, or keepout
  safety;
- `outcome` - success, steps, near misses, safety violations, and other consequences.

These layers are meant to answer one practical question: where did the first meaningful divergence
appear?

### How To Read The Evidence

For an action-channel fault, the most useful evidence pattern is:

```text
raw_policy_action_changed: false
postprocessed_action_changed: true
```

Plain meaning: the policy produced the same raw action, but the action changed after the policy
returned it.

What this supports: the first visible fault is downstream of inference, in the action adapter,
middleware, controller bridge, or test intervention.

What this does not prove: it does not prove the model is generally safe, and it does not prove the
exact real-world component that would be responsible in another stack.

For a hidden safety failure, the important pattern is:

```text
success_changed: false
outcome_violation_changed: true
hidden_success_safety_violation: true
```

Plain meaning: the task success bit did not change, but a safety or contract violation appeared in
the perturbed run.

What this supports: a binary success benchmark would miss this incident.

What this does not prove: it does not prove physical harm, hardware exploitability, or that every
simulator replay will reproduce the same path.

### Diagnosis Logic

For the quickstart:

1. The clean and perturbed raw policy actions match.
2. The perturbed raw action and postprocessed action diverge.
3. The observed layer is therefore `action_postprocess`; the intervention manifest declares the same
   layer, so `declared_layer` agrees.
4. The outcome stays successful but step count inflates.
5. RobotSec Range checks that this evidence matches `robotprobe-v0-gripper-freeze`.

The result is a root-cause hypothesis:

```text
layer: action_postprocess
declared_layer: action_postprocess
rule_match_score: 0.87
contract: ActionSpaceContract
fault: gripper_freeze_mid
```

For the hidden keepout demo:

1. The clean and perturbed raw policy actions match at the first fault window.
2. The perturbed postprocessed action receives a lateral velocity bias.
3. The `KeepoutZoneContract` is marked violated.
4. Both runs still reach the goal.
5. The perturbed outcome includes `keepout_violated`.
6. RobotSec Range checks that the report identifies an action-postprocess fault, not a policy
   failure.

The result is:

```text
layer: action_postprocess
contract: KeepoutZoneContract
fault: stealth_velocity_bias_keepout
hidden_success_safety_violation: true
```

### Why This Matters

Robotics teams already have logs, simulator traces, dashboards, MCAP files, and observability
tools. Those are necessary, but they usually do not turn a clean-vs-perturbed incident into a
machine-checkable causal diagnosis.

RobotProbe is meant to sit next to those systems and answer a different question:

```text
What changed first, at which layer, and what consequence did that create?
```

That is useful for:

- debugging failures without prematurely blaming the model;
- building regression tests for action adapters and safety wrappers;
- red-teaming command-path tampering and monitor-evasion cases;
- checking whether a defense catches the actual consequence, not just task failure;
- producing a compact incident bundle that another engineer can inspect and reproduce.

### What The Local Demos Prove

The local demos prove that RobotProbe can:

- record paired clean and perturbed traces;
- preserve the baseline relationship between runs;
- distinguish raw policy action from postprocessed action;
- record contract checks;
- summarize first divergence and outcome differences;
- produce a RobotSec Range checkable diagnosis;
- render a compact report that shows why task success alone is insufficient.

### What They Do Not Prove

The local demos do not prove:

- real hardware exploitability;
- broad VLA robustness or weakness;
- that a specific robot company has this bug;
- that the toy simulator captures real contact dynamics;
- that the same numbers should transfer to MuJoCo, Isaac, PyBullet, or hardware.

Those would require integration with the target stack, target-specific safety contracts, and
realistic simulator or hardware-shadow validation.

### Instrumenting your own stack

For a real robot or VLA stack, wire the same events around the team's existing runner:

- record observations before policy inference;
- record raw policy action after inference;
- record postprocessed action after deployment transforms;
- record contract checks at action, trajectory, and safety-envelope boundaries;
- record outcome metrics from the simulator or robot supervisor;
- run a paired clean and perturbed episode;
- check the generated summary against a challenge or internal regression gate.

The first serious integration should not try to replace the team's observability stack. It should
export or link to existing logs, then add the missing paired-run diagnosis layer on top.

## Glossary

Plain definitions for the terms used across `docs/`. Each entry says what the code actually
implements; for limits and non-claims see [Claims](status-and-claims.md#claims).

### The two level taxonomies

RobotProbe uses "L0–L3" for **two unrelated** taxonomies. Always write the full enum value
(`L1_REPRODUCIBLE_INCIDENT`, not bare "L1") when the taxonomy is not obvious from context.

- **Conformance level** — how much a trace's *self-description* guarantees, checked by
  `robotprobe conformance` (`robotprobe/conformance.py`): `L0_WELL_FORMED`,
  `L1_REPRODUCIBLE_INCIDENT`, `L2_CONTRACT_CHECKABLE`. Cumulative; about the trace's contents,
  not about real-hardware behavior or diagnosis correctness.
- **Replay level** — how far a recorded episode can be re-executed, defined as the `ReplayLevel`
  enum in `robotprobe/core.py`: `L0_TRACE_PLAYBACK`, `L1_POLICY_REPLAY`, `L2_SIM_REPLAY`,
  `L3_SHADOW_HARDWARE_REPLAY`. Serialized as `replay_level` on every trace event, and repeated at
  the top level of a summary. `robotprobe replay` builds a replay *plan* at the requested level
  (replay L1 and above are planning outputs, not executions).

### Core artifacts

- **Trace** — a JSONL file of typed events (`instruction`, `observation`, `policy_input`,
  `raw_action`, `postprocessed_action`, `outcome`, `contract_check`, ...), written by
  `TraceRecorder` (`robotprobe/record.py`). Every event carries run context (`run_id`,
  `scenario`, `replay_level`, optional `seed`) plus `step`, `event_type`, and `layer` — the
  point in the robot stack the event is about (`policy_forward`, `action_postprocess`, ...),
  which `robotprobe summarize` uses to localize where a fault entered.
- **Intervention manifest** — a YAML fault definition (loaded by
  `load_intervention_manifest`) declaring a sim-only intervention; when a `TraceRecorder` is
  given one, it records `intervention_manifest_ref` and its sha256 `intervention_manifest_hash`
  in the run context so the perturbation is pinned.
- **Incident bundle** — the directory `create_incident_bundle` (`robotprobe/bundle.py`) writes:
  `bundle.json` pointing to the clean and perturbed traces, the intervention manifest,
  `summary.json` / `causal_diff.json`, an optional `check.json`, and `report.md`.
- **Sidecar** — the single-file export of an incident bundle (`robotprobe export sidecar`,
  `robotprobe/exporters.py`): one JSON document in format `robotprobe.sidecar.v0` embedding the
  bundle, summary, and check plus references to the traces, manifest, and report. An MCAP export
  (`robotprobe.mcap.v0`) is also available.
- **Summary / causal diff** — the JSON produced by `robotprobe summarize`
  (`summarize_trace_pair` in `robotprobe/summarize.py`) comparing a clean and a perturbed trace:
  `root_cause_hypothesis`, `evidence`, and a top-level `replay_level`.
- **Observed layer vs declared layer** — `root_cause_hypothesis.layer` is computed from the traces
  (the earliest layer at which the recorded data differs). `root_cause_hypothesis.declared_layer`
  echoes `target_layers` from the intervention manifest and is `null` without one. Agreement means
  the fault appeared where the experiment intended, not that the tool independently found the cause.
- **`rule_match_score`** — a fixed value assigned by the observed-layer rule. It is a rule
  constant, not a calibrated probability.
- **`raw_policy_action_changed` vs `raw_policy_drift_vs_clean`** — the first asks whether the
  intervention altered raw policy output; the second is the raw clean-vs-perturbed comparison,
  which also moves when the policy is stochastic. Under a post-policy manifest the intervention
  cannot change raw output, so only the within-trace `raw_action_mutated_after_policy` flag is
  meaningful there.

### Clean-vs-perturbed and verdicts

- **Clean-vs-perturbed** — the paired-run pattern RobotProbe is built around: one unperturbed
  baseline run (`scenario: "clean"`) and one run under an intervention manifest, sharing a
  `perturbation_group_id`; all diffing happens between these two traces.
- **`hidden_success_safety_violation`** — a summary flag set when the task *succeeded* in both
  runs (`success` is true in the clean and perturbed outcomes) yet contract violations changed
  between them: the failure is invisible to a success-only scoreboard.

### Checking and extension

- **Contract** — a YAML predicate over trace events (`robotprobe/contracts.py`): threshold
  comparisons (`lt`/`gt`/`eq`/...) and temporal rules (`must_never` / `must_always` /
  `must_eventually`) over step windows; violations are surfaced as `contract_check` events.
- **Challenge** — a `ChallengeSpec` (`robotsec_range/challenge_spec.py`): an `id`, a
  `diagnosis`, and `required_evidence`, checked against a summary by
  `robotsec_range/checker.py` to produce a bundle's `check.json`. Built-ins are listed with
  `robotsec-range list-challenges`.
- **Pack** — an installable plugin directory (validated by `robotprobe/packs.py`) with a plugin
  manifest plus contracts, adapters, and challenges; `robotprobe pack` scaffolds templates such
  as `ros2-command-authority` and `vla-eval-adapter`.
- **MechProbe** — the white-box mechanistic forensics layer (`robotprobe/mech/`): forward-hook
  activation capture over a model, an activation cache view, and mech evidence levels gated by
  `robotsec_range/mech_checker.py`. Synthetic and unknown-provenance results are ineligible as model
  evidence; complete real offline comparisons reach at most Level 1. It does not establish causal
  feature claims or validated prediction from temporal precedence.
- **RobotSec Range** — the `robotsec_range` package ("RobotSec Range v0 challenge diagnosis
  primitives"): challenge specs, checkers, and fixtures that RobotProbe bundles call into for
  verdicts; it is the diagnosis layer, separate from the tracing layer.
