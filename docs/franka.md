# Franka and PI0.5

The bundled Franka pack: CPU demos, the GPU PI0.5/LIBERO run, and the read-only path to a real robot.

## CPU demos

RobotProbe provides three audience-specific Franka demonstrations over one runtime and one
evidence format. The center of gravity is not a polished motion clip by itself: it is the ability
to show exactly what the policy proposed, what safety decided, what the selected backend did, and
what the evidence does and does not establish.

### Choose a demo

| Audience | Demo | Primary question | Local runtime |
| --- | --- | --- | --- |
| Security assessor | **Safety Gate** | Was an unsafe proposal stopped before it reached the backend? | CPU-only |
| Robotics engineer | **Backend Swap** | Can the environment or robot-facing backend change without rewriting policy instrumentation? | CPU-only, with optional targets inspected |
| Public or classroom | **Action Journey** | Can we follow one action from observation to outcome? | CPU-only |

Run the complete local suite from the repository root:

```bash
uv sync
uv run python examples/demos/franka/run_demos.py \
  --demo all \
  --out .artifacts/franka-demos
```

The output directory must be absent or empty. Use a different directory for a second run.

To run one view:

```bash
uv run python examples/demos/franka/run_demos.py \
  --demo safety-gate \
  --out .artifacts/franka-safety-gate

uv run python examples/demos/franka/run_demos.py \
  --demo backend-swap \
  --out .artifacts/franka-backend-swap

uv run python examples/demos/franka/run_demos.py \
  --demo action-journey \
  --out .artifacts/franka-action-journey
```

### What gets generated

`--demo all` reuses exactly three bounded runtime runs rather than producing three disconnected
stories:

```text
.artifacts/franka-demos/
├── suite.json
├── demos/
│   ├── safety-gate.md
│   ├── backend-swap.md
│   ├── backend-matrix.json
│   └── action-journey.md
└── runs/
    ├── safe/
    ├── rejected/
    └── replay/
```

Each run contains its generated `runtime.yaml`, `runtime.jsonl`, `run-result.json`, and an
`evidence/` directory. RobotProbe runtime evidence includes:

```text
evidence/
├── evidence.json
├── report.md
├── timeline.md
├── manifest.json
└── inputs/
    ├── trace.jsonl
    ├── baseline.jsonl        # only for a paired comparison
    └── runtime-config.yaml
```

`manifest.json` records relative paths and SHA256 digests. `evidence.json` keeps runtime integrity,
task outcome, and motion disposition separate, so a valid safety rejection is not mislabeled as a
broken run and a valid trace is not mislabeled as task success.

### Demo 1: Safety Gate

Open:

```bash
cat .artifacts/franka-demos/demos/safety-gate.md
cat .artifacts/franka-demos/runs/rejected/evidence/report.md
```

The baseline proposes an in-envelope delta. The candidate proposes an oversized delta. The same
scheduler and safety gateway evaluate both; the candidate ends with
`safety_rejected:action_limit_exceeded`, motion disposition `rejected_pre_backend`, zero execution
receipts, and zero applied commands.

### Demo 2: Backend Swap

Open:

```bash
cat .artifacts/franka-demos/demos/backend-swap.md
cat .artifacts/franka-demos/demos/backend-matrix.json
```

The runner, Franka embodiment, command space, scenario contract, safety gateway, and evidence
renderer stay fixed. Configuration selects either
`robotprobe.backends.franka_deterministic` or
`robotprobe.backends.franka_recorded_replay`. The replay path uses a synthetic recorded-format
fixture and always reports `command_applied: false`.

The two runnable configurations are:

```text
examples/packs/franka-pi05-runtime/configs/deterministic.yaml
examples/packs/franka-pi05-runtime/configs/franka-recorded-replay.yaml
```

### Demo 3: Action Journey

Open:

```bash
cat .artifacts/franka-demos/demos/action-journey.md
cat .artifacts/franka-demos/runs/safe/evidence/timeline.md
```

The timeline follows stable observation, policy-call, action-chunk, and command identifiers. It
then shows the safety decision, execution receipt, measured state, and task outcome. Activation
captures, when present, are summarized beside their policy call without assigning causal meaning.

### Optional PI0.5 and LIBERO execution

The CPU demos prove local contracts; they do not load
`lerobot/pi05_libero_finetuned_v044`. On a compatible Linux/CUDA host, run:

```bash
ROBOTPROBE_SMOKE_OUT=/tmp/robotprobe-pi05-libero-smoke \
bash scripts/run_pi05_libero_franka_smoke.sh
```

A `completed` smoke exercises PI0.5 inference, semantic activation capture, and LIBERO simulation
for the bounded task. A `skipped` result is not a pass, and task success remains separate from
integration completion. The script produces `/tmp/robotprobe-pi05-libero-smoke/runtime-evidence/`
only after runtime execution, trace validation, and pack validation succeed.

Any completed runtime trace can also be instrumented after the fact:

```bash
uv run robotprobe runtime evidence \
  --trace /tmp/robotprobe-pi05-libero-smoke/pi05-libero-trace.jsonl \
  --config /tmp/robotprobe-pi05-libero-smoke/generated-runtime.yaml \
  --out /tmp/robotprobe-pi05-libero-smoke/manual-runtime-evidence
```

See [PI0.5/LIBERO validation](franka.md#pi05-on-libero-gpu) for prerequisites, checkpoint pins, exact
terminal states, and interpretation.

### Real Franka: retain the seam, change the transport

For a physical Franka, implement the four-method read-only transport described in
[real Franka shadow migration](franka.md#real-franka-read-only), then select:

```text
examples/packs/franka-pi05-runtime/configs/franka-ros2-shadow.yaml
```

The policy and evidence layers remain unchanged. The site package supplies timestamped robot and
camera state only. This is physical observation only: the transport has no command publication
method, every receipt must say `command_applied: false`, and physical actuation is not implemented.
A future actuator requires a distinct hardware-executable plugin, calibration and controller
evidence, independent safety enforcement, operator arming, and a separate review.

For a future simulator or robot transport, follow
[Franka backend authoring](runtime.md#writing-a-backend). The runner does not need a backend-name
branch; compatibility is declared through execution-mode and command-space capabilities.

### Evidence claim matrix

| Evidence level | What is available now | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| Static inspection | All four checked-in configurations | Manifest/config compatibility without optional imports | Runtime execution |
| Deterministic kinematic contract | CPU demo | Runtime composition, lineage, safety, receipts, and evidence rendering | Physics, learned-policy behavior, or hardware |
| Synthetic recorded-format fixture | CPU replay demo | Backend swapping and read-only proposal evaluation | A physical capture or physical Franka behavior |
| LIBERO simulation | Optional Linux/CUDA smoke, not the CPU demo | The exact completed checkpoint/simulator path for that retained run | Broad robustness or physical transfer |
| Physical observation only | Site-specific Franka shadow transport, still an external validation gate | Timestamped physical state plus shadow policy/safety evaluation | Motion, physical task success, or safe control |
| Physical actuation | Not implemented | Nothing yet | Any hardware motion or safety claim |

The architecture and command-space limitations are documented in
[the modular Franka runtime](runtime.md#architecture). Current verified and open evidence gates are
tracked in [RobotProbe status](status-and-claims.md#status).

## PI0.5 on LIBERO (GPU)

This runbook distinguishes static compatibility, mocked contract tests, a real checkpoint smoke,
benchmark task success, and physical transfer. They are different evidence levels.

### Automated local evidence

The default suite verifies:

- lazy imports without Torch, LeRobot, LIBERO, MuJoCo, or ROS 2;
- the LeRobot 0.5.x PI0.5 processor and complete action-chunk boundary using injected fakes;
- semantic module discovery against a representative named-module tree;
- bounded summary/full activation storage;
- the LIBERO environment adapter through an injected vector environment;
- the deterministic Franka episode through the public CLI; and
- read-only hardware-shadow invariants.

These tests do not load `lerobot/pi05_libero_finetuned_v044` or execute MuJoCo.

### Checkpoint identity

Both PI0.5 example configurations pin the Hugging Face checkpoint revision
`dbf8a3f794a9c4297b44f40b752712f50073d945`. RobotProbe passes that same revision,
cache directory, and offline-mode setting to the LeRobot config and policy-weight loaders. It also
resolves the preprocessor JSON, postprocessor JSON, and their normalization state from that exact
local checkpoint snapshot. This prevents a moving `main` artifact from being combined with weights
from a different revision.

The checkpoint's preprocessor names `google/paligemma-3b-pt-224` as a transitive tokenizer
dependency. The examples pin that tokenizer separately at
`35e4f46485b4d07967e7e9935bc3786aad50687c`; RobotProbe preloads it with the same cache and
offline-mode controls and injects the resulting tokenizer into LeRobot's processor pipeline.

Treat a revision update as an explicit compatibility change: inspect the upstream `config.json`,
preprocessor and postprocessor configurations, rerun the adapter and runtime suites, and retain the
new revisions in the generated trace metadata. These pins identify Hugging Face model artifacts;
they do not pin the separately installed LeRobot, Torch, CUDA, LIBERO, or MuJoCo packages.

### Heavyweight prerequisites

The smoke script requires:

- Linux;
- `uv`;
- sufficient writable disk;
- a CUDA-capable Torch runtime with `torch.cuda.is_available()` true;
- compatible `lerobot`, `libero`, and `mujoco` imports;
- the current LeRobot `LiberoEnv`, environment factory, and `PI05Policy` symbols;
- checkpoint availability locally or through configured Hugging Face access; and
- a working headless MuJoCo graphics backend.

The script exports `MUJOCO_GL=egl` before any optional imports unless the caller selects another
known-good backend such as `osmesa`.

RobotProbe intentionally does not install or pin a site CUDA stack. Prepare that environment using
the compatibility guidance of the selected LeRobot/LIBERO versions, then install RobotProbe in it.

### Bounded smoke

From the repository root:

```bash
ROBOTPROBE_SMOKE_OUT=/tmp/robotprobe-pi05-libero-smoke \
bash scripts/run_pi05_libero_franka_smoke.sh
```

Defaults:

- suite `libero_spatial`;
- task ID `0`;
- seed `7`;
- maximum `10` environment steps;
- one selected task and one environment;
- PI0.5 inference on CUDA;
- semantic summary capture; and
- a 20 GB free-disk preflight.

Useful explicit overrides:

```bash
ROBOTPROBE_LIBERO_SUITE=libero_spatial \
ROBOTPROBE_LIBERO_TASK_ID=0 \
ROBOTPROBE_LIBERO_SEED=7 \
ROBOTPROBE_LIBERO_MAX_STEPS=10 \
ROBOTPROBE_CAPTURE_STORAGE=summary \
MUJOCO_GL=egl \
ROBOTPROBE_SMOKE_OUT=/tmp/robotprobe-pi05-libero-smoke \
bash scripts/run_pi05_libero_franka_smoke.sh
```

For an already cached checkpoint:

```bash
ROBOTPROBE_LOCAL_FILES_ONLY=1 \
ROBOTPROBE_SMOKE_OUT=/tmp/robotprobe-pi05-libero-local \
bash scripts/run_pi05_libero_franka_smoke.sh
```

Full storage requires `ROBOTPROBE_CAPTURE_STORAGE=npy` or `safetensors`; the generated runtime
configuration writes tensors below the selected smoke output and enforces
`ROBOTPROBE_CAPTURE_MAX_BYTES`.

The source configuration is:

```text
examples/packs/franka-pi05-runtime/configs/pi05-libero.yaml
```

The selected profile is:

```text
examples/packs/franka-pi05-runtime/capture/pi05-semantic.yaml
```

The script writes a generated config rather than editing either source file.

### Evidence states

`evidence-state.json` has exactly one terminal state:

- `completed`: runtime command, trace validation, pack validation, and required trace-event checks
  finished. Task success remains a separate value in `smoke-summary.json`.
- `failed`: prerequisites were present but runtime execution or evidence validation failed. Inspect
  the named stage and log files.
- `skipped`: a named prerequisite was unavailable. This is neither a pass nor a failed model.

A completed smoke requires at least one `model_activation`, completed policy inference, action
chunk, execution receipt, and outcome event. It does not require task success because a short,
bounded smoke is an integration check rather than a benchmark run. The evidence check also
requires both activation and action-chunk metadata to match the configured checkpoint and
tokenizer revisions; `smoke-summary.json` records those revisions and the two provenance checks.
Only after runtime execution, trace validation, and pack validation succeed does the script create
the generic `runtime-evidence/` bundle. Its manifest path is recorded in `smoke-summary.json`.

Outputs include:

```text
evidence-state.json
generated-runtime.yaml
runtime-output.log
trace-validation.json
pack-validation.json
smoke-summary.json
pi05-libero-trace.jsonl
runtime-evidence/
├── evidence.json
├── report.md
├── timeline.md
├── manifest.json
└── inputs/
activations/                  # only for full storage
```

### Interpreting the result

A completed run demonstrates the exact installed checkpoint/runtime path, selected semantic
hooks, simulated environment interaction, and trace validity for that run.

It does not prove broad policy robustness, benchmark-level performance, causal meaning of
activation summaries, safe simulator-to-hardware action conversion, or physical task success. It
does not prove physical Franka safety.

A task-success claim requires the full benchmark protocol, task/seed counts, success aggregation,
environment/version identity, and retained receipts. A physical-transfer claim additionally
requires the calibration and controller gates in
[real Franka shadow migration](franka.md#real-franka-read-only).

### Static checks

These checks are safe without optional dependencies:

```bash
uv run robotprobe pack validate examples/packs/franka-pi05-runtime
uv run robotprobe runtime inspect \
  --config examples/packs/franka-pi05-runtime/configs/pi05-libero.yaml
uv run robotprobe pack claims examples/packs/franka-pi05-runtime
```

See [the modular Franka runtime](runtime.md#architecture) for the complete architecture and evidence
boundary. See [Franka evidence demos](franka.md#cpu-demos) to present a completed smoke through the
same evidence spine as the CPU-only demonstrations.

## Real Franka, read-only

The initial real-Franka path is deliberately observation-only. It lets RobotProbe capture physical
state, run a policy, inspect PI0.5 internals, schedule proposed actions, and evaluate safety
envelopes without sending motion commands.

### Hard invariant

The backend capability is:

```text
is_hardware: true
supports_shadow: true
can_apply_hardware_commands: false
```

Every execution receipt contains `command_applied: false`. The backend requires an authorization
disposition of `shadow_only`, and its transport protocol has no publish, send, execute, or command
method. The runtime and receipt types independently reject an applied hardware-shadow command.

This is not a controller with a dry-run flag. It is a read-only design boundary.

### Implement the site transport

The backend lazily imports a small external package named `robotprobe_franka_ros2` at `connect()`.
That package exposes:

```python
def create_shadow_transport(**options):
    return SiteFrankaReadOnlyTransport(**options)
```

The returned object implements only:

```python
def connect() -> None: ...
def health() -> FrankaTransportHealth: ...
def read_snapshot() -> FrankaShadowSnapshot: ...
def close() -> None: ...
```

Use `franka_ros2`, a read-only `libfranka` state bridge, or a site observability service behind
that contract. Keep the package deployment-specific; RobotProbe needs no core change.

Do not add a motion publisher to the same object. If code can publish, a configuration mistake can
turn an observation component into an actuator. A future actuator belongs in a
new hardware-executable plugin with separate review and explicit opt-in.

### Snapshot requirements

`FrankaShadowSnapshot` carries:

- source timestamp in the same epoch as the configured clock;
- named agent and wrist camera frames;
- end-effector position and `xyzw` quaternion;
- two gripper positions;
- seven joint positions, velocities, and efforts;
- base/task frame ID;
- camera/robot calibration ID;
- robot mode;
- controller authority; and
- transport metadata.

RobotProbe rejects disconnected, unhealthy, future-dated, stale, malformed, or incomplete
snapshots. `reset()` starts a logical observation session only; it does not reset or move the
robot.

### Deployment data that remains mandatory

Simulator-to-hardware reuse is not achieved by changing one backend ID. Record and review:

- camera calibration, intrinsics, extrinsics, resolution, crop, rotation, and semantic role;
- robot base, flange, end-effector, camera, and task frame transforms;
- clock source and maximum state age;
- joint and gripper sign/order mapping;
- robot mode, faults, recovery state, and controller authority semantics;
- task instruction, object layout, and operator-controlled reset procedure;
- action normalization and denormalization;
- translation and rotation units;
- control rate, latency, queue horizon, saturation, and rate limits;
- workspace, collision, force, velocity, and acceleration limits; and
- an independently measured physical success condition.

The current `franka.cartesian_delta_gripper.v0` boundary does not yet prove equivalence between
LIBERO normalized relative-controller values and calibrated physical Franka motion. Action
normalization is therefore an open gate, not an implementation detail.

### Inspect the configuration

```bash
uv run robotprobe runtime inspect \
  --config examples/packs/franka-pi05-runtime/configs/franka-ros2-shadow.yaml
```

Inspection succeeds without ROS 2, Torch, a checkpoint, or hardware. Running requires the
read-only transport package plus the PI0.5 runtime:

```bash
uv run robotprobe runtime run \
  --config examples/packs/franka-pi05-runtime/configs/franka-ros2-shadow.yaml
```

The default uses one scheduled action per observation so a queued model chunk does not hide
physical state changes. The 5-second observation-age envelope accommodates non-real-time VLA
inference but is not a certified latency bound; tune it from measured deployment data.

Before a site transport is available, use the synthetic recorded-format replay in
[Franka evidence demos](franka.md#cpu-demos) to rehearse configuration swapping and read-only receipt
handling. Replay is simulation evidence; it is not a substitute for a timestamped physical
observation.

### Before connecting a robot

1. Verify the transport package contains no command publication API.
2. Run its unit tests with a recorded state source.
3. Inspect the pack and confirm `hardware_shadow`.
4. Confirm the backend advertises `can_apply_hardware_commands: false`.
5. Confirm the selected scenario says `reset_semantics: logical_session_only`.
6. Validate timestamp age and camera calibration on a stationary robot.
7. Run with the physical controller disabled or under an independent operator-safe state.
8. Inspect the trace and confirm every receipt has `command_applied: false`.

Operator supervision and emergency-stop readiness remain required around powered hardware even
for read-only software.

### Executable motion is a separate project

Do not extend `FrankaRos2ShadowBackend.execute()` to publish. Motion requires a
new hardware-executable plugin and, at minimum:

- calibrated action conversion with explicit units and frames;
- independent workspace, velocity, acceleration, force, and collision enforcement;
- controller lifecycle, watchdog, fault, and emergency-stop evidence;
- command acknowledgement and measured-state receipts;
- stale-observation and stale-command rejection at the controller boundary;
- operator arming and explicit opt-in;
- staged no-load and low-energy physical validation; and
- a target-specific safety review.

Until those gates are closed, RobotProbe's real-Franka capability is deep observation and shadow
assessment, not autonomous actuation.

See [Franka backend authoring](runtime.md#writing-a-backend) and
[the safety policy](safety.md). The [Franka demo guide](franka.md#cpu-demos) provides the audience
walkthrough and the full evidence claim matrix.
