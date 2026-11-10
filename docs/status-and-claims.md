# Status and claims

What is implemented, what is claimed, and the runs behind those claims.

## Status

What works today, what is experimental, and what is not built.

### Works

**Traces and diagnosis**

- JSONL trace recording and validation for typed stack events: instruction, observation,
  policy input, raw action, postprocessed action, contract check, outcome, robot state, sim state.
- Clean-vs-perturbed causal summary: observed layer, declared layer, first divergence step,
  action-change flags, outcome deltas, and a rule-based layer match score.
- Consequence metrics: action energy, jerk, trajectory deviation, recovery latency, intervention
  exposure, near-miss scoring.
- Threshold and temporal contract checks (`must_never` / `must_always` / `must_eventually`) over
  step windows.
- Incident bundles: summary, causal diff, check, report, and manifest, with a sidecar export.
- Failure-semantics conformance profile (`robotprobe.failure-semantics/v0`) with ordered
  L0/L1/L2 levels.
- Fault campaigns over a recorded baseline, with post-hoc contract checking.
- World-model faithfulness checks over imagined evidence events.

**Integrations**

- ROS 2 MCAP ingest, with CDR payload decoding behind the `ros2-decoders` extra.
- vla-eval exported episode ingest.
- Isaac Lab-Arena exported trace ingest.
- MCAP export/import and sidecar export, round-trip tested.
- ViFailback and RoboFAC public incident-dataset conversion.

**Packs**

- Pack manifest parsing, validation, listing, testing, claim cards, and a local registry.
- Built-in templates for ROS 2 command authority, vla-eval adapters, and Isaac Lab-Arena.
- Safety classification gate that blocks `hardware_executable` packs by default.
- `--allow-code` gating for any pack whose test path imports its entrypoint.

**Modular runtime**

- One runner over pluggable policy, scheduler, safety gateway, and robot backend components, with
  stable IDs from observation through action chunk, safety decision, execution receipt, and outcome.
- `robotprobe runtime inspect | run | evidence`, and runtime-native evidence bundles.
- Franka pack with a deterministic kinematic backend, a synthetic recorded-replay backend, an
  optional LIBERO backend, and a read-only ROS 2 hardware-shadow backend.
- Three CPU-only Franka demos: Safety Gate, Backend Swap, and Action Journey.

**Local demos**

- `robotprobe demo`, the quickstart script, the Gymnasium CartPole demo, and the hidden-keepout
  security demo, all CPU-only.

### Experimental

- **MechProbe**, the white-box activation layer: activation capture and storage, cache inspection,
  clean-vs-perturbed activation diff, manual feature diff, and offline consequence scoring. The
  fixture path is labelled synthetic through reports and evidence checks. Synthetic or unknown
  provenance is Level 0; complete real offline comparisons reach at most Level 1.
- **PI0.5 policy and capture** (`robotprobe.integrations.lerobot`): LeRobot PI0.5 adapter with
  semantic activation probes, run through the runtime on LIBERO with a GPU for one checkpoint,
  task, and seed. Not a benchmark result.
- **Live in-process instrumentation** (`PolicyProbe`): hooks, fault injection, and watchpoints over
  a Python policy callable. In-process only; there is no cross-process attach.
- **NNsight bridge** (`NNsightActivationAdapter`): captures activations from an NNsight-supported
  PyTorch model, with a Python persistence bridge into the file-backed MechProbe analysis pipeline.
  Verified on a small CPU model, not on a VLA. The CLI does not load live models.

### Not built

- No live OpenVLA or GR00T activation adapter. PI0.5 capture runs only inside the modular runtime.
- No physical actuation. The real-Franka backend observes only and cannot send motion commands.
- No temporal activation patching, activation steering, or SAE dictionary support.
- No automatic inclusion of MechProbe tensors, indexes, or reports in bundles or MCAP exports.
- No remote pack registry, and no signing beyond a manifest SHA256.
- No native Foxglove or Rerun visualization schemas (the MCAP bridge is the interop path).
- No hardware execution. Every bundled intervention is `sim_only`.
- No full Dockerized LIBERO benchmark run. The attempt and its blocker are recorded in
  [Evidence](status-and-claims.md#evidence).

### Verifying

```bash
make ci          # lint, tests, doctor, demo smoke
make test        # pytest only
```

GitHub Actions runs the same checks on push and pull request.

## Claims

RobotProbe provides a reusable instrumentation and incident-forensics layer for robot-stack
security experiments. It can record typed traces, summarize clean-vs-perturbed causal
differences, validate summaries against RobotSec Range challenge specs, and render compact
incident bundles.

The included fixtures show that the tooling can represent instruction, observation,
action-postprocess, hidden success/safety divergence, real Pi0.5/LIBERO trace, and
imagined-evidence integrity scenarios.

The current repository includes evidence that:

- synthetic challenge traces can be checked locally;
- a local sim-only keepout demo can keep binary task success unchanged while RobotProbe surfaces a
  post-policy safety-contract violation;
- real Pi0.5/LIBERO trace fixtures can drive metric-gated checks;
- OpenVLA and GR00T model servers start through official `vla-eval` configs and complete short
  instrumented episodes on real weights (the injected fault produced no observable action change in
  those runs, so this validates integration, not detection);
- a full LIBERO Docker benchmark-container attempt was blocked by the host's container privilege
  model.

Details, commands, and the limits of each result are in [Evidence](status-and-claims.md#evidence).

### How The Diagnosis Fields Are Produced

`robotprobe summarize` reports both an **observed** layer and the **declared** layer from the
intervention manifest:

- `root_cause_hypothesis.layer` is computed from the traces: the earliest layer at which the
  recorded data differs. It is a heuristic over trace fields, not a causal inference.
- The manifest chooses which comparison feeds that heuristic. Under an `action_postprocess`
  manifest, cross-run raw-action differences are ignored and only the raw-to-postprocessed mutation
  inside the perturbed trace counts, so `layer` cannot be `policy_forward`. Without a manifest, or
  with any other target layer, both raw and postprocessed actions are compared across runs.
- `root_cause_hypothesis.declared_layer` echoes `target_layers` from the intervention manifest, and
  is `null` when no manifest is supplied. Agreement between the two means the fault appeared where
  the experiment intended it to, not that the tool independently found the cause.
- `root_cause_hypothesis.rule_match_score` is a fixed score assigned by that heuristic rule. It is
  not calibrated against labelled incidents and should not be read as a probability.
- There is no counterfactual claim. The summary does not re-run the episode with the fault removed,
  and no field should be read as if it did.
- The summary reads instruction, observation, `raw_action`, `postprocessed_action`, `contract_check`,
  and `outcome` events. It ignores runtime events such as `safety_decision` and `execution_receipt`.
  For a guarded run, where the safety gateway rejected a command before dispatch, the summary still
  reports the post-processing change and does not say the command was blocked. Use
  `robotprobe runtime evidence` for the gateway's decisions.
- The summary compares whole episodes. Once a fault changes what the robot does, later
  observations and raw actions differ too. Without an `action_postprocess` manifest, a long enough
  rollout can therefore report `policy_forward` for a fault injected downstream of the policy; check `first_divergence_step` and
  the earliest changed command rather than the layer alone.

Two evidence fields are worth distinguishing. `raw_policy_action_changed` asks whether the
intervention altered raw policy output; `raw_policy_drift_vs_clean` is the raw clean-versus-perturbed
comparison, which also moves when the policy is stochastic. Under a post-policy manifest the
intervention cannot change raw output, so only `raw_action_mutated_after_policy` is meaningful
there.

### Non-Claims

RobotProbe is not a robotics simulator, not a policy-training framework, and not a replacement
for benchmark harnesses such as LIBERO or vla-eval.

The current alpha is not evidence of broad model robustness. A passing or failing RobotSec
challenge is evidence about the recorded trace and the stated contract, not a universal claim
about all robots, all VLAs, or all deployment conditions.

RobotProbe does not currently claim full Dockerized LIBERO benchmark compatibility.

RobotProbe does not ship real-hardware destructive interventions. Public examples are
simulation-first, dry-run, or trace-only.

RobotProbe does not replace ROS tracing, Foxglove, Rerun, OpenTelemetry, simulator logs, or model
evaluation harnesses. It is designed to sit on top of those systems as a fault-injection and
causal-diagnosis layer.

### MechProbe Claims

MechProbe provides generic forward-hook capture, a NNsight adapter tested on a small CPU model,
and a Python bridge that persists captures with explicit run/incident metadata. It compares selected
internal activations against external RobotProbe traces. Pi0.5 has semantic capture probes inside
the modular runtime, exercised on one LIBERO checkpoint; see
[Evidence](status-and-claims.md#evidence). OpenVLA and GR00T still need model-specific adapters.

The current MechProbe slice can validate capture configs, represent `model_activation` trace
events, write activation indexes, compute simple layerwise activation divergence, and render
reports that preserve synthetic provenance and comparison coverage. Synthetic or unknown-provenance
outputs are Level 0 and ineligible as model evidence. Complete real offline comparisons reach at
most Level 1; temporal precedence alone is not a predictive finding.

### MechProbe Non-Claims

MechProbe does not prove that an SAE feature label is true by inspection.

MechProbe does not work on closed model-server-only systems unless activation probes are exposed.

MechProbe does not turn internal feature steering into a safety guarantee.

MechProbe findings below evidence Level 3 are candidate research findings, not root-cause claims.

MechProbe is not required for RobotProbe bundles. External RobotProbe trace diagnosis remains the
primary path when white-box model access is unavailable. Bundle creation and MCAP export do not
automatically include MechProbe indexes, tensors, or reports; share those files separately.

## Evidence

What has actually been run, and what each run does and does not establish. If a result is not
listed here, it has not been validated. The GPU runs used rented hardware; their raw logs and
recordings are not part of this repository.

| Run | Date | Result | What it does not establish |
| --- | --- | --- | --- |
| Local suite and demos (CPU) | any | Unit suite, `robotprobe demo`, simulator demos, and pack checks pass. | Anything about a real robot or model. |
| vla-eval harness, synthetic benchmark | 2026-06-16 | The instrumented run produced a valid paired trace with `action_postprocess` divergence at step 1. | Model behaviour; the benchmark is synthetic. |
| OpenVLA and GR00T model servers | 2026-06-16 | Both started from official `vla-eval` configs and completed 3-step clean and perturbed episodes with valid traces. | Fault detection: synthetic frames and a constant gripper output meant the fault had no observable effect. |
| Pi0.5/LIBERO recorded traces | 2026-06-16 | Real traces, reduced to `robotsec_range/fixtures/robotprobe-v1-libero-realtrace/` (numeric trajectories and image hashes only; no images, model weights, or LIBERO assets are redistributed), drive a passing metric-gated challenge. | Anything beyond these traces and this contract. |
| Full Dockerized LIBERO benchmark | 2026-06-16 | The benchmark did not run: the host lacked `cap_sys_admin` and `/dev/fuse`, so nested Docker failed. | Benchmark compatibility. |
| PI0.5/LIBERO through the modular runtime | 2026-10-29 | On real weights in simulation: an instruction swap changed the first action and late language and action-output activations; the safety gateway rejected a command before dispatch; a downstream gripper fault was mislocated as `policy_forward` by the whole-episode summary. | Reliability across checkpoints, tasks, or seeds; each result is one checkpoint, task, and seed. |

### What these runs do not establish

- No real-hardware execution, and no claim about physical robot behaviour.
- No LIBERO benchmark success-rate comparison between policies.
- No evidence that fault localization is reliable across policies, tasks, or seeds. The
  2026-10-29 run includes a mislocated fault.
- No causal claim from activation differences. An L2 difference does not say what a feature means
  or that it caused an action.

### Reproducing

The local rows reproduce from a clean checkout with `make setup` and `uv run robotprobe demo`. The
GPU rows need a CUDA host and network access for model weights. `scripts/gpu_smoke.sh` runs the
CPU-safe subset, and [the PI0.5/LIBERO runbook](franka.md#pi05-on-libero-gpu) covers the
GPU run.
