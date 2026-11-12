# Changelog

Notable changes to RobotProbe. The project follows a loose
[semantic versioning](https://semver.org/) scheme while it is in alpha.

## v0.2.0-alpha

### Added

- Modular robot runtime (`robotprobe.runtime`): one runner over pluggable policy, scheduler, safety
  gateway, and backend components, with `robotprobe runtime inspect | run | evidence`.
- Franka pack and backends under `robotprobe.integrations.franka`: deterministic kinematic,
  synthetic recorded replay, optional LIBERO, and read-only ROS 2 hardware shadow. Three CPU-only
  demos (Safety Gate, Backend Swap, Action Journey) in `examples/demos/franka`.
- LeRobot PI0.5 policy adapter and semantic activation probes under
  `robotprobe.integrations.lerobot`, plus a bounded PI0.5/LIBERO experiment driver in `scripts/`.
- `persist_capture_pair` connects in-memory adapter captures to the file-backed MechProbe pipeline
  using explicit incident, run, target, scenario, and replay metadata.
- A focused CI job installs the optional NNsight runtime and runs MechProbe tests without silently
  skipping that adapter.
- An evidence table in `docs/status-and-claims.md`: what has actually been run, and what each run
  does and does not establish.
- `SECURITY.md` with a disclosure path and the pack/code-execution threat surface.
- `CODE_OF_CONDUCT.md`, and contribution licensing terms in `CONTRIBUTING.md`.
- A third-party names notice in `README.md`: product names describe interoperability only.
- GitHub Actions workflow running lint, tests, and demo and pack smoke checks on Python 3.10, 3.12,
  and 3.13.
- A repository hygiene test that fails if a private path, cloud endpoint, maintainer-note reference,
  or personal email address is reintroduced.
- Failure-semantics conformance profile (`robotprobe.failure-semantics/v0`) with ordered L0/L1/L2
  levels and seeded trace recording.
- Fault-injection campaigns over a recorded baseline, with post-hoc contract checking.
- World-model faithfulness verification for imagined-evidence events.
- NNsight activation-capture bridge for MechProbe (optional `nnsight` extra).
- In-process instrumentation (`PolicyProbe`): hooks, live fault injection, and watchpoints over a
  Python policy callable.
- `robotprobe demo`, a single command that runs the bundled incident and writes a full bundle from
  package data.
- ROS 2 CDR decoding for `robotprobe ingest ros2-mcap` behind the `ros2-decoders` extra.
- `make ci`, a local lint/test/doctor/demo gate, and `mcap-ros2-support` in the dev dependency group
  so `uv sync && pytest` works from a clean checkout.

### Changed

- RobotSec challenge checks validate both traces and independently recompute the summary and required
  metrics before awarding a pass. Supplied summaries must agree with the recomputed evidence;
  missing trace evidence fails closed unless packaged challenge fixtures provide it. This establishes
  consistency with the traces, not their authenticity.
- Documentation is consolidated from 29 pages into 12, grouped by task: getting started, concepts,
  CLI, packs, integrations, runtime, Franka, MechProbe, experimental features, status and claims,
  safety, and the failure-semantics spec. Development setup moved into `CONTRIBUTING.md`.
- MechProbe preserves synthetic provenance through feature analysis, reports, and evidence checks.
  Fixtures and unknown provenance are Level 0; real offline comparisons reach at most Level 1.
- MechProbe capture uses runtime steps and immutable activation snapshots, supports bfloat16
  conversion with source dtype metadata, and enforces sampling, storage dtype, and tensor budgets.
- Activation event output is split into one file per run; `activation_events` now maps `clean` and
  `perturbed` to their paths. Capture persistence requires a fresh or empty output directory.
- MechProbe rejects nonfinite or incompatible tensors, marks missing pairs as insufficient
  evidence, and includes candidate feature results in Markdown reports.
- The OpenVLA scaffolding pack advertises no implemented capture or replay capabilities.
- `root_cause_hypothesis` reports the **observed** layer measured from the traces, plus
  `declared_layer` echoed from the intervention manifest. Previously the reported layer was the
  manifest's declared target, so a manifest that injected nothing still produced a "diagnosis".
- `root_cause_hypothesis.confidence` is now `rule_match_score`, reflecting what it always was: a
  fixed value assigned by the layer heuristic. It is not calibrated and not a probability.
- `counterfactuals.replay_confidence` is now a top-level `replay_level` field. It holds a
  replay-level enum value and was never a confidence score.
- `counterfactuals.fault_removed_recovered` has been removed. It reported `true` whenever both
  traces existed, implying a counterfactual experiment that never ran.
- `conformance` requires `L1_REPRODUCIBLE_INCIDENT` by default rather than treating any
  schema-valid trace as conformant.
- `worldmodel faithfulness` reports `faithfulness_rate: null` when a trace contains no imagined
  events, instead of `1.0`.
- Traces no longer carry `expected_layer_of_effect` / `expected_downstream_layers`. The fields were
  never read and put an expectation into the evidence itself.
- `integrations` is now `robotprobe.integrations`, so the distribution no longer installs a
  top-level `integrations` package that can collide with another project.
- MechProbe synthetic capture is labelled in its index (`"synthetic": true`) and in the activation
  diff (`"evidence_eligible": false`). It exercises the pipeline and is not evidence about a model.
- License metadata uses the SPDX form (`Apache-2.0`) and the build requires setuptools 77.
- `make setup` installs the core library plus dev tooling; simulator and white-box extras live in
  `make setup-simulators` and `make setup-nnsight`.

### Removed

- The long-form demo narratives (presentation talk tracks and the detailed run write-ups). The
  runnable commands and claim boundaries stay in the docs.
- The `mech train-sae` and `mech steer` subcommands. Neither was implemented; both existed only to
  print a refusal and exit non-zero.
- Maintainer working notes, dated run artifacts, and the private-release policy from the published
  tree.
- The checked-in `examples/public-demo/security-keepout` bundle. It was generated output kept in
  version control; `robotprobe demo` and `examples/simulators/security_keepout_demo.py` reproduce
  the same layout on demand.

## v0.1.0-alpha

- First alpha: trace recording, clean-vs-perturbed causal summaries, RobotSec challenge checking,
  incident reports and bundles, and a local quickstart demo that needs no GPU, robot, or simulator.
