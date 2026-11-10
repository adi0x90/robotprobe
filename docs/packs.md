# Packs

Encode a failure mode as a reusable pack, and the plugin architecture underneath it.

## Authoring a pack

This page is for a robotics, VLA, or safety team that already has logs, traces, simulator output,
or a failure they can describe. The goal is to turn that material into a RobotProbe pack without
changing RobotProbe core code.

### What A Pack Mimics

In a real robotics deployment, failures usually live at boundaries:

- the instruction changed but the policy ignored it;
- the observation stream became stale;
- an action chunk was replayed too long;
- post-processing changed the policy action;
- units, joint order, gripper semantics, or camera freshness drifted between systems.

A RobotProbe pack captures one such boundary as data:

```text
pack manifest
  -> contracts or adapter definitions
  -> optional fixtures and golden diagnoses
  -> claim card
  -> reusable check in CI, demos, or review
```

The pack should make the diagnostic claim reviewable without requiring the original author to
explain it.

### Start From The Built-In Templates

Create a local pack skeleton:

```bash
uv run robotprobe pack init /tmp/robotprobe-pack-demo \
  --id robotprobe-pack-demo \
  --plugin-id robotprobe.contracts.demo \
  --plugin-type contract
```

Or start from a specific integration template:

```bash
uv run robotprobe pack init /tmp/robotprobe-pack-ros2 \
  --id robotprobe-pack-ros2 \
  --template ros2-command-authority

uv run robotprobe pack init /tmp/robotprobe-pack-vla-eval \
  --id robotprobe-pack-vla-eval \
  --template vla-eval-adapter

uv run robotprobe pack init /tmp/robotprobe-pack-isaac \
  --id robotprobe-pack-isaac \
  --template isaac-lab-arena-adapter
```

Then inspect the generated files:

```bash
find /tmp/robotprobe-pack-demo -maxdepth 3 -type f -print
cat /tmp/robotprobe-pack-demo/manifest.yaml
```

### Minimal Pack Layout

```text
robotprobe-pack-example/
  manifest.yaml
  contracts/
  interventions/
  adapters/
  fixtures/
  golden/
  docs/
```

Only `manifest.yaml` is required, but a useful pack should usually include docs and at least one
fixture or golden diagnosis.

### Manifest Fields That Matter Most

```yaml
id: robotprobe-pack-example
version: 0.1.0
trust_level: community_pack
publisher:
  name: your-team
compatibility:
  robotprobe_min: 0.2.0
  schema_version: robotprobe.pack.v0.2
execution:
  imports_code_during_validate: false
  imports_code_during_test: false
  network_access_required: false
plugins:
  - id: robotprobe.contracts.example
    type: contract
    version: 0.1.0
    files:
      - contracts/example.yaml
    safety:
      release_classification: trace_only
      hardware_allowed: false
    claims:
      proves:
        - detects the declared contract condition in recorded traces
      does_not_prove:
        - physical exploitability
        - full model-family robustness
        - hardware safety certification
```

Use `trace_only` unless the pack actually runs in a simulator. Use `sim_only` for synthetic or
simulator mutation. Use `hardware_shadow` only for non-commanding hardware envelope checks.
Public alpha packs should not be `hardware_executable`.

### Validate, Test, And Render Claims

```bash
uv run robotprobe pack validate /tmp/robotprobe-pack-demo
uv run robotprobe pack test /tmp/robotprobe-pack-demo
uv run robotprobe pack claims /tmp/robotprobe-pack-demo
```

These commands answer different questions:

- `validate`: is the manifest structurally valid and are declared files present?
- `test`: do static checks and golden fixture comparisons pass?
- `claims`: what does this pack say it proves, and what does it refuse to prove?

For packs that import code during tests:

```bash
uv run robotprobe pack test path/to/pack --allow-code
```

Only use `--allow-code` when the pack entrypoint is intentionally tested and does not perform
network, hardware, or heavy model-runtime side effects at import time.

### Add A Golden Diagnosis

Golden diagnoses make a pack useful for regression checks. The pattern is:

```text
fixtures/
  scenario-name/
    summary.json
golden/
  scenario-name/
    summary.json
```

`robotprobe pack test` compares matching JSON files under `fixtures/` and `golden/`.

Use this for stable expected outputs such as:

- changed layer;
- contract name;
- fault name;
- first divergence step;
- action-change flags;
- outcome delta fields.

Do not include large raw datasets in the pack unless redistribution is clear. Prefer small fixtures
and source references.

### Start From A Plain JSONL Trace

If you already have RobotProbe JSONL traces:

```bash
uv run robotprobe validate-trace traces/clean.jsonl
uv run robotprobe validate-trace traces/perturbed.jsonl

uv run robotprobe summarize \
  --clean traces/clean.jsonl \
  --perturbed traces/perturbed.jsonl \
  --manifest interventions/fault.yaml \
  --metrics action_energy,jerk,trajectory_deviation,recovery_latency,near_miss \
  > fixtures/my-incident/summary.json
```

Review the summary, then copy it to `golden/my-incident/summary.json` once the expected diagnosis is
stable.

`pack test` compares every file under `golden/` with the same relative path under `fixtures/`. That
makes the pair a drift check, not a correctness check: it catches an output that changed after the
diagnosis was reviewed, but it cannot tell you the diagnosis was right. Keep `fixtures/` as the
recorded output and `golden/` as the reviewed one, and update them together.

### Start From External Logs

RobotProbe can ingest exported files from several ecosystems:

```bash
uv run robotprobe ingest ros2-mcap \
  --input /path/to/ros2_bag.mcap \
  --out traces/ros2_command_authority.jsonl \
  --run-id ros2_run_001 \
  --target ros2 \
  --scenario command_authority_replay

uv run robotprobe ingest vla-eval \
  --input /path/to/vla_eval_episode.json \
  --out traces/vla_eval_episode.jsonl \
  --run-id vla_eval_run_001 \
  --scenario libero_spatial

uv run robotprobe ingest isaac-lab-arena \
  --input /path/to/isaac_trace.jsonl \
  --out traces/isaac_lab_arena_episode.jsonl \
  --run-id isaac_run_001 \
  --scenario lift_cube
```

The first version of an adapter pack can be trace-only: prove that the exported artifact can be
normalized into RobotProbe events before adding interventions or challenge checks.

### Submission Checklist

Before opening a pack submission:

```bash
uv run robotprobe pack validate path/to/pack
uv run robotprobe pack test path/to/pack
uv run robotprobe pack claims path/to/pack
```

Also include:

- a short `README.md` in the pack's `docs/` directory;
- the source or provenance of any trace fixture;
- safety classification for every plugin;
- one paragraph of claims and non-claims;
- whether the pack is trace-only, sim-only, or hardware-shadow.

### What Not To Claim

Pack evidence is intentionally narrow. A pack can show a reusable diagnosis pattern for a boundary
or incident class. It does not, by itself, prove broad VLA robustness, real-world exploitability,
or that a model's internal features caused the behavior.

## Plugin architecture

RobotProbe v0.2 keeps the evidence kernel small and moves domain-specific behavior into plugin
packs.

For a practical contribution walkthrough, read [Pack Authoring Walkthrough](packs.md#authoring-a-pack) first. This
page is the lower-level reference for manifest shape, commands, and extension boundaries.

The core repository owns:

- trace and bundle schemas;
- intervention and plugin manifest validation;
- paired-run causal summaries;
- contract-check results;
- RobotSec challenge checks;
- incident reports and sidecar exports.

Packs own:

- simulator or log adapters;
- interventions and attacks;
- safety/security contracts;
- metrics;
- defense evaluators;
- exporters;
- challenge fixtures and golden diagnoses.

### Pack Layout

```text
robotprobe-pack-example/
  manifest.yaml
  adapters/
  interventions/
  contracts/
  metrics/
  defenses/
  exporters/
  challenges/
  fixtures/
  golden/
  docs/
  tests/
```

Only `manifest.yaml` is required. Other directories are optional, but declared files must exist.

### Manifest Shape

```yaml
id: robotprobe-pack-keepout-safety
version: 0.1.0
trust_level: trusted_core_pack
publisher:
  name: robotprobe
compatibility:
  robotprobe_min: 0.2.0
  schema_version: robotprobe.pack.v0.2
execution:
  imports_code_during_validate: false
  imports_code_during_test: false
  network_access_required: false
challenges:
  - challenges/robotprobe-v1-security-keepout-bypass.json
plugins:
  - id: robotprobe.contracts.keepout_margin
    type: contract
    version: 0.1.0
    files:
      - contracts/keepout_margin.yaml
    safety:
      release_classification: sim_only
      hardware_allowed: false
    claims:
      proves:
        - detects keepout-zone violations in recorded simulator traces
      does_not_prove:
        - hardware certification
        - physical exploitability
```

Supported plugin types:

- `adapter`
- `intervention`
- `contract`
- `metric`
- `defense`
- `diagnoser`
- `exporter`
- `challenge_pack`

Supported release classifications:

- `trace_only`
- `sim_only`
- `hardware_shadow`
- `hardware_executable`

By default, `hardware_executable` plugins fail validation. This prevents a public pack from quietly
shipping hardware-mutating behavior.

### Commands

Scaffold a safe pack skeleton:

```bash
robotprobe pack init /tmp/robotprobe-pack-demo \
  --id robotprobe-pack-demo \
  --plugin-id robotprobe.contracts.demo \
  --plugin-type contract
```

Scaffold a pack from a built-in integration template:

```bash
robotprobe pack init /tmp/robotprobe-pack-ros2 \
  --id robotprobe-pack-ros2 \
  --template ros2-command-authority

robotprobe pack init /tmp/robotprobe-pack-vla-eval \
  --id robotprobe-pack-vla-eval \
  --template vla-eval-adapter

robotprobe pack init /tmp/robotprobe-pack-isaac \
  --id robotprobe-pack-isaac \
  --template isaac-lab-arena-adapter
```

Validate a pack:

```bash
robotprobe pack validate examples/packs/keepout-safety
```

List pack contents:

```bash
robotprobe pack list examples/packs/keepout-safety
```

Run static conformance checks:

```bash
robotprobe pack test examples/packs/keepout-safety
```

`pack test` runs validation plus golden fixture checks. Every JSON file under `golden/` is compared
with the matching relative JSON file under `fixtures/`.

If a pack declares `execution.imports_code_during_test: true`, `pack test` refuses to import plugin
entrypoints unless the caller passes `--allow-code`:

```bash
robotprobe pack test examples/my-pack --allow-code
```

Render a human-readable claim card:

```bash
robotprobe pack claims examples/packs/libero-action-contracts
```

Build and inspect a local pack registry:

```bash
robotprobe pack registry build examples/packs --out /tmp/robotprobe-pack-registry.json
robotprobe pack registry list /tmp/robotprobe-pack-registry.json
```

The registry output is a machine-readable index of pack IDs, versions, plugin IDs, safety
classification, trust metadata, manifest SHA256, challenge refs, and validation issues. It is
a local file: publishers can commit it, serve it statically, or import it
into internal package catalogs without RobotProbe needing a hosted registry service.

Evaluate a contract against a trace:

```bash
robotprobe contracts check \
  --contract examples/packs/keepout-safety/contracts/keepout_margin.yaml \
  --trace /tmp/robotprobe-keepout/traces/stealth_velocity_bias_keepout.jsonl
```

Export a bundle sidecar or MCAP file:

```bash
robotprobe export sidecar \
  --bundle /tmp/robotprobe-keepout \
  --out /tmp/robotprobe-keepout/robotprobe.sidecar.json

robotprobe export mcap \
  --bundle /tmp/robotprobe-keepout \
  --out /tmp/robotprobe-keepout/robotprobe.mcap

robotprobe import mcap \
  --mcap /tmp/robotprobe-keepout/robotprobe.mcap \
  --out /tmp/robotprobe-keepout-restored
```

Ingest ecosystem logs into RobotProbe traces:

```bash
robotprobe ingest ros2-mcap \
  --input /path/to/ros2_bag.mcap \
  --out traces/ros2_command_authority.jsonl \
  --run-id ros2_run_001 \
  --target ros2 \
  --scenario command_authority_replay

robotprobe ingest vla-eval \
  --input /path/to/vla_eval_episode.json \
  --out traces/vla_eval_episode.jsonl \
  --run-id vla_eval_run_001 \
  --scenario libero_spatial

robotprobe ingest isaac-lab-arena \
  --input /path/to/isaac_trace.jsonl \
  --out traces/isaac_lab_arena_episode.jsonl \
  --run-id isaac_run_001 \
  --scenario lift_cube
```

Seed public incident datasets into RobotProbe traces:

```bash
robotprobe incident-pack download vifailback \
  --out "$HOME/datasets/robotprobe-incidents/raw/vifailback" \
  --task cut_sausage

robotprobe incident-pack convert vifailback \
  --annotation "$HOME/datasets/robotprobe-incidents/raw/vifailback/annotated_data/cut_sausage/cut_sausage_annotations.json" \
  --out "$HOME/datasets/robotprobe-incidents/packs/vifailback-cut-sausage" \
  --limit 3

robotprobe incident-pack download robofac \
  --out "$HOME/datasets/robotprobe-incidents/raw/robofac" \
  --leaf simulation_data/LiftPegUpright-box/view0

robotprobe incident-pack convert robofac \
  --metadata "$HOME/datasets/robotprobe-incidents/raw/robofac/simulation_data/LiftPegUpright-box/view0/view0.json" \
  --h5 "$HOME/datasets/robotprobe-incidents/raw/robofac/simulation_data/LiftPegUpright-box/view0/view0.h5" \
  --out "$HOME/datasets/robotprobe-incidents/packs/robofac-liftpeg" \
  --limit 3 \
  --max-steps 50
```

See [Public Incident Packs](integrations.md#public-incident-packs). These dataset-backed packs are
`trace_only`: they represent annotated public robot incidents without pretending they are
clean-vs-perturbed causal runs.

Inspect environment readiness:

```bash
robotprobe doctor
robotprobe doctor --pack examples/packs/libero-action-contracts
robotprobe doctor --integration vla-eval
```

### Contract DSL

The first contract DSL is deliberately small. It supports threshold checks over recorded trace
fields:

```yaml
id: keepout_margin_violation
type: threshold_contract
when:
  field: keepout_violated
  eq: true
require:
  field: keepout_violated
  eq: false
violation:
  contract: KeepoutZoneContract
  check: recorded_keepout_violation
  severity: 0.9
```

This says: when a trace records `keepout_violated=true`, the requirement that it remain false has
failed, so emit a `KeepoutZoneContract` violation.

Current operators:

- `lt`
- `lte`
- `gt`
- `gte`
- `eq`
- `neq`

Temporal contracts are also supported for common incident-forensics checks over a step window:

```yaml
id: command_authority_window
type: temporal_contract
window:
  start_step: 10
  end_step: 120
must_never:
  field: command_authority.source
  eq: bypass_proxy
violation:
  contract: CommandAuthorityContract
  check: bypass_proxy_inside_supervised_window
  severity: 0.95
```

Supported temporal operators:

- `must_never`: violation if the predicate is observed inside the window.
- `must_always`: violation if any event inside the window fails the predicate.
- `must_eventually`: violation if no event inside the window satisfies the predicate.

This is not a full LTL/STL engine. It covers the high-value safety/security cases RobotProbe needs
for v0.2 while keeping contract files small and inspectable.

### Built-In Pack Templates

RobotProbe ships three executable adapter pack templates:

- `ros2-command-authority`: trace-only command-authority audit pack with
  `robotprobe.integrations.ros2.mcap_adapter:ingest_ros2_mcap`.
- `vla-eval-adapter`: simulator-only adapter pack with
  `robotprobe.integrations.vla_eval.episode_ingest:ingest_vla_eval_episode`.
- `isaac-lab-arena-adapter`: simulator-only adapter pack with
  `robotprobe.integrations.isaac_lab_arena.trace_ingest:ingest_isaac_lab_arena_trace`.

The committed examples live under:

```text
examples/packs/ros2-command-authority
examples/packs/vla-eval-adapter
examples/packs/isaac-lab-arena-adapter
```

These adapters ingest exported artifacts. They do not launch ROS 2, `vla-evaluation-harness`, Isaac
Sim, or Isaac Lab-Arena directly. That keeps RobotProbe installable in a lightweight incident-review
environment while still allowing teams to convert real logs into RobotProbe traces.

### Python Entrypoints

A plugin can declare a Python entrypoint:

```yaml
entrypoint: sample_pack:SampleContract
```

RobotProbe resolves entrypoints as `module:object`. When loading a pack-local entrypoint, callers
can provide the pack directory as a search path. This keeps the manifest portable while allowing
community packs to ship lightweight Python implementations.

Entrypoint loading is explicit. Validation does not import plugin code by default. This keeps
`pack validate` safe for untrusted packs and reserves execution for commands or integrations that
need it.

### Golden Fixtures

Packs can include fixture/golden pairs:

```text
fixtures/keepout_summary.json
golden/keepout_summary.json
```

`robotprobe pack test` compares the normalized JSON. A mismatch fails the pack test and reports the
relative path. This is the first conformance layer for community packs: they should ship examples
that prove their expected diagnosis shape.

### Sidecar Export

The sidecar exporter writes RobotProbe evidence into a portable JSON envelope:

```json
{
  "format": "robotprobe.sidecar.v0",
  "bundle": {},
  "summary": {},
  "check": {},
  "refs": {}
}
```

This remains dependency-free and useful for internal tooling or environments that want the
RobotProbe evidence envelope without writing an MCAP file.

### MCAP Bridge

RobotProbe also writes bundle evidence to MCAP using the public `mcap` Python library. The bridge
uses schemaless JSON channels so MCAP-aware tools can inspect RobotProbe evidence without requiring
a custom binary schema:

```text
/robotprobe/bundle
/robotprobe/summary
/robotprobe/check
/robotprobe/report
/robotprobe/manifest
/robotprobe/traces/clean
/robotprobe/traces/perturbed
```

Trace topics contain one JSON message per recorded event. Bundle/report/manifest topics contain
the same evidence files RobotProbe uses for causal diagnosis. Import is intentionally scoped to
RobotProbe-owned `/robotprobe/...` topics; this is not a generic ROS bag or arbitrary MCAP decoder.

The practical workflow is:

```bash
robotprobe export mcap --bundle incident/ --out incident/robotprobe.mcap
robotprobe import mcap --mcap incident/robotprobe.mcap --out restored/
```

This is the compatibility layer for teams that already store robot incidents in MCAP-oriented
tooling. Rich native Foxglove/Rerun visual schemas can be added as exporter plugins later, but the
current bridge keeps the first integration portable and easy to verify.

### Extension Rule

If a contribution is specific to one simulator, model server, robot stack, defense, or attack
family, it should start as a pack. Move behavior into core only when multiple independent packs need
the same evidence primitive.
