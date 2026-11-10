# Modular runtime

The runner that drives a policy, scheduler, safety gateway, and robot backend under one trace, and how to write a backend for it.

## Architecture

RobotProbe now has a composable policy-to-robot runtime for Franka-shaped observations and
seven-value relative Cartesian action chunks. The center of gravity is the evidence boundary:
policy, embodiment, backend, scenario, safety, capture, and tracing remain separable so the
backend can change without a model-specific runner.

### What is implemented

The example pack provides four paths:

| Configuration | Policy | Backend | Evidence level |
| --- | --- | --- | --- |
| `deterministic.yaml` | deterministic reach | deterministic kinematic Franka | local executable contract test |
| `franka-recorded-replay.yaml` | deterministic reach | read-only snapshot replay | synthetic recorded-format fixture |
| `pi05-libero.yaml` | LeRobot PI0.5 | LeRobot LIBERO/MuJoCo | optional simulated integration |
| `franka-ros2-shadow.yaml` | LeRobot PI0.5 | read-only Franka transport | physical observation and shadow evaluation only |

The deterministic backend is a deterministic kinematic contract. It is not a physics simulator:
it has no contact dynamics, object model, controller dynamics, or benchmark assets. Its purpose is
to prove component resolution, observation-to-action lineage, safety decisions, receipts, cleanup,
and trace validity on an ordinary development machine.

The optional LIBERO path uses `lerobot/pi05_libero_finetuned_v044`, two camera roles, a Franka
state boundary, a complete action chunk, and configurable semantic activation probes. Static
inspection does not load Torch, LeRobot, LIBERO, MuJoCo, a checkpoint, or a GPU.

The hardware-shadow path reads timestamped physical state through a separate transport contract.
It has no command method and every receipt must contain `command_applied: false`.

For the three audience-specific walkthroughs built on these paths, start with
[Franka evidence demos](franka.md#cpu-demos).

### Local deterministic run

From the repository root:

```bash
uv sync
uv run robotprobe pack validate examples/packs/franka-pi05-runtime
uv run robotprobe runtime inspect \
  --config examples/packs/franka-pi05-runtime/configs/deterministic.yaml
uv run robotprobe runtime run \
  --config examples/packs/franka-pi05-runtime/configs/deterministic.yaml
uv run robotprobe validate-trace \
  examples/packs/franka-pi05-runtime/artifacts/deterministic-trace.jsonl
```

`runtime inspect` is static. It checks the manifest, selected component kinds, shared command
space, mode compatibility, capture path, and option names without importing the entry points.
`runtime run` imports only the selected components and runs one bounded episode.

The deterministic trace contains:

- instruction and canonical observation identity;
- policy-inference start and completion;
- complete proposed action chunks;
- scheduled chunk indices;
- per-action safety decisions;
- backend execution receipts;
- measured robot state; and
- the final outcome.

These events share stable observation, policy-call, action-chunk, and command identifiers.

Turn any completed runtime trace into a self-contained RobotProbe runtime evidence bundle:

```bash
uv run robotprobe runtime evidence \
  --trace examples/packs/franka-pi05-runtime/artifacts/deterministic-trace.jsonl \
  --config examples/packs/franka-pi05-runtime/configs/deterministic.yaml \
  --out /tmp/robotprobe-franka-evidence
```

The bundle separates runtime integrity, task outcome, and motion disposition and includes
deterministic JSON, Markdown, copied inputs, and a SHA256 manifest.

### PI0.5 plus LIBERO

Inspect the optional runtime without loading it:

```bash
uv run robotprobe runtime inspect \
  --config examples/packs/franka-pi05-runtime/configs/pi05-libero.yaml
```

Run the prerequisite-aware smoke on a suitable Linux/CUDA host:

```bash
bash scripts/run_pi05_libero_franka_smoke.sh
```

The script defaults to one suite task, one seed, ten environment steps, summary capture, and a
bounded output directory. See [PI0.5/LIBERO validation](franka.md#pi05-on-libero-gpu) before treating
its result as evidence.

This path does not prove checkpoint task success merely because the model loaded, one inference
completed, or a trace validated. A completed smoke proves the exercised runtime path and records
the task outcome separately.

### Configurable deep capture

Both PI0.5 configurations select:

```text
examples/packs/franka-pi05-runtime/capture/pi05-semantic.yaml
```

The profile chooses semantic roles rather than depending on one wrapper prefix:

- early, middle, and late vision blocks;
- early, middle, and late language blocks;
- early, middle, and late action-expert MLPs;
- action input/output projections;
- time-conditioning input/output projections; and
- optional exact or glob user selectors.

The profile is resolved only after the selected model is loaded. A requested role that is absent
is a runtime error, not a guessed substitute. The default records summaries and caps each selected
module at 16 calls per policy inference. Full `.npy` or safetensors capture is opt-in and requires
an output directory plus a byte budget:

```yaml
capture:
  profile: ../capture/pi05-semantic.yaml
  storage: safetensors
  output_dir: ../artifacts/pi05-activations
  max_bytes_per_run: 67108864
```

Every activation event includes:

- probe and concrete module identity;
- exact capture-profile SHA256;
- tensor shape, dtype, finite-value statistics, norm, and content digest;
- environment step;
- policy-call, action-chunk, and observation IDs;
- module call index; and
- denoising step only for roles whose call relationship is established.

These summaries are observational. They do not establish that a captured activation caused an
action or failure.

### Swapping backends

The runner does not branch on PI0.5, Franka, LIBERO, ROS 2, or backend names. A configuration
selects four plugin IDs:

```yaml
components:
  policy:
    plugin: robotprobe.policies.pi05_lerobot
  embodiment:
    plugin: robotprobe.embodiments.franka
  backend:
    plugin: robotprobe.backends.libero_franka
  scenario:
    plugin: robotprobe.scenarios.libero_task
```

Changing the backend still requires compatible command-space and execution-mode capabilities.
It does not imply that simulator coordinates, action scaling, camera geometry, reset behavior, or
success criteria transfer to hardware.

For a new simulator or transport, follow [Franka backend authoring](runtime.md#writing-a-backend).
For physical observation, follow [real Franka shadow migration](franka.md#real-franka-read-only).
For runnable examples, follow [Franka evidence demos](franka.md#cpu-demos).

### Command-space limitation

The current shared command-space identifier is:

```text
franka.cartesian_delta_gripper.v0
```

Its shape is `[dx, dy, dz, drx, dry, drz, gripper]`, with axis-angle rotation delta and gripper
range `[-1, 1]`. The LIBERO checkpoint and environment use their native relative-controller
scaling. The deterministic backend interprets translation values as its synthetic position
deltas. Those are not yet a calibrated physical-unit equivalence.

Therefore the current physical configuration is shadow-only. Action normalization, controller
scaling, frame transforms, rate limits, and workspace limits remain explicit gates before any
executable hardware adapter can be designed.

### Evidence boundary

Locally passing tests and the deterministic run prove the modular contracts and evidence path.
Injected fake environments prove adapter behavior without claiming the heavyweight dependency is
installed. A `skipped` heavyweight smoke is not a pass. A completed LIBERO smoke is simulation
evidence, not physical Franka evidence. Hardware shadow is observation evidence, not motion or
task-success evidence.

The pack claim card is the concise machine-derived boundary:

```bash
uv run robotprobe pack claims examples/packs/franka-pi05-runtime
```

## Writing a backend

This guide describes how to add a simulator, replay source, or read-only hardware transport
without changing RobotProbe's episode runner or PI0.5 adapter.

### The contract

A backend implements `RuntimeBackend` from `robotprobe/runtime/protocols.py`:

```python
class RuntimeBackend:
    capabilities = ...

    def connect(self) -> None: ...
    def reset(self, scenario: ScenarioSpec) -> RawObservation: ...
    def observe(self) -> RawObservation: ...
    def execute(self, action: AuthorizedAction) -> ExecutionReceipt: ...
    def health(self) -> BackendHealth: ...
    def close(self) -> None: ...
```

Only `AuthorizedAction` crosses the execution boundary. A backend must reject a raw scheduled
action, a rejected authorization, an undeclared command space, a wrong action dimension, or
non-finite values.

### Declare exact capabilities

The class and pack manifest must declare identical `RuntimeComponentCapabilities`. Runtime
construction compares the complete capability object and fails if the imported object drifts from
the static manifest.

At minimum declare:

- a stable component ID and runtime family;
- supported command spaces and observation streams;
- simulation versus hardware;
- reset and ground-truth behavior;
- shadow support;
- whether hardware commands can be applied; and
- optional dependencies.

Do not claim a generic Franka command space if the backend silently changes axis order, units,
rotation convention, gripper polarity, clipping, or controller scaling. Add an embodiment codec or
a distinct command-space identifier instead.

### Observation boundary

The Franka embodiment expects a `RawObservation.payload` containing:

```text
images: mapping with configured semantic camera roles
eef_position: finite vector[3]
eef_quaternion: finite vector[4], declared as xyzw
gripper_qpos: finite vector[2]
joint_positions: finite vector[7]
joint_velocities: finite vector[7]
joint_efforts: finite vector[7]
instruction: non-empty string
frame_id: non-empty string
calibration_id: non-empty string
quaternion_convention: xyzw
```

Copy mutable arrays at the boundary. Use a source capture timestamp, not the time at which a
delayed message was decoded. Preserve simulator ground truth versus physical measured/estimated
state in metadata.

If a signal is unavailable, do not fabricate it as measured data. The LIBERO adapter currently
uses zero joint efforts only with `joint_efforts_available: false`; a production backend should
either provide the signal or evolve the canonical optionality explicitly.

### Lifecycle rules

- `connect()` owns optional runtime creation and imports heavyweight dependencies lazily.
- `reset()` must state whether it changes the environment or starts a logical session.
- `observe()` must fail before connection/reset and return a new sequence for a new sample.
- `execute()` returns a receipt even when a backend safely refuses an in-envelope command.
- `close()` is idempotent and invalidates cached observations even if dependency cleanup fails.
- Instance state must not be kept in a process-global environment cache.

Simulator auto-reset behavior must not erase the terminal observation used in a receipt. Normalize
external API drift behind private helpers and test every accepted return variant.

### Receipt rules

An `ExecutionReceipt` preserves the command ID, backend ID, execution mode, applied flag, measured
state, terminal state, success, and a bounded reason.

For simulation, `command_applied: true` means the simulator accepted and stepped the action. It
does not mean a physical command was issued.

For hardware shadow:

```text
execution_mode: hardware_shadow
command_applied: false
```

A shadow backend reporting an applied command is a fatal runtime invariant violation. The safest
shadow transport interface omits publish/send methods entirely.

### Add the pack entry

Add an `adapter` entry to:

```text
examples/packs/franka-pi05-runtime/manifest.yaml
```

Use `release_classification: sim_only` for a simulator and `hardware_shadow` for a non-commanding
physical reader. Keep `hardware_allowed: false` unless a separate future executable plugin has
passed its own safety review.

Add a runtime configuration selecting the new plugin. The core registry resolves the manifest
entry point; do not add a backend-name switch to `robotprobe/runtime/config.py` or
`robotprobe/runtime/orchestrator.py`.

### Required tests

Use an injected fake dependency and cover:

- import without optional dependencies;
- exact constructor/config mapping;
- lifecycle guards;
- seeded reset semantics where applicable;
- observation shapes, frame, calibration, and timestamps;
- accepted and rejected action behavior;
- terminal/success variants;
- stale/disconnected state;
- cleanup and reconnect;
- capability equality with the manifest;
- execution-mode incompatibility; and
- the public conformance helper.

The reusable check is:

```python
from robotprobe.runtime.conformance import assert_backend_conforms

report = assert_backend_conforms(
    backend=backend,
    scenario=scenario,
    action=authorized_action,
)
assert report.passed
```

Then run:

```bash
uv run pytest tests/test_libero_franka_backend.py \
  tests/test_franka_ros2_shadow_backend.py -q
uv run robotprobe pack validate examples/packs/franka-pi05-runtime
uv run robotprobe runtime inspect \
  --config examples/packs/franka-pi05-runtime/configs/deterministic.yaml
```

See [the modular runtime guide](runtime.md#architecture) for the user path and
[the real shadow guide](franka.md#real-franka-read-only) for the non-commanding hardware
boundary.
