# Getting started

Install RobotProbe, run the bundled incident, and try the CPU simulator demos.

## Quickstart

This walkthrough runs entirely locally and does not require a GPU, simulator, or model checkpoint.
It demonstrates the core RobotProbe loop:

```text
clean trace + perturbed trace + intervention manifest
  -> causal summary
  -> RobotSec challenge check
  -> incident report bundle
```

### 1. Install

From the repository root:

```bash
python -m pip install -e .
```

### 2. Run The Five-Minute Demo

From a clean checkout:

```bash
make setup
uv run robotprobe doctor
bash examples/quickstart/run_demo.sh
uv run robotprobe pack claims examples/packs/libero-action-contracts
uv run robotprobe pack test examples/packs/libero-action-contracts
```

If you are not using uv, the equivalent consumer path is `python -m pip install -e .` followed by
the same `robotprobe ...` commands. The demo script itself prefers `uv run` when uv is installed and
falls back to `python` otherwise.

If you already installed the package, the demo command alone is:

```bash
bash examples/quickstart/run_demo.sh
```

By default this writes:

```text
examples/quickstart/out/
```

The output includes:

- `summary.json` - causal-diff summary;
- `causal_diff.json` - machine-readable causal diff;
- `check.json` - RobotSec challenge result;
- `report.md` - human-readable incident report;
- `bundle.json` - bundle manifest;
- copied traces and intervention manifest.

### 3. Inspect The Result

```bash
cat examples/quickstart/out/report.md
cat examples/quickstart/out/check.json
```

Expected diagnosis:

```text
layer: action_postprocess
contract: ActionSpaceContract
fault: gripper_freeze_mid
RobotSec check: PASS
```

If you want a hidden-success safety violation rather than an action-contract fault, run the keepout
demo from [Local Simulator Demos](getting-started.md#simulator-demos):

```bash
uv run python examples/simulators/security_keepout_demo.py \
  --out examples/simulators/out/security-keepout
```

It writes the same bundle layout from a different failure mode.

Next useful docs:

- [Pack Authoring Walkthrough](packs.md#authoring-a-pack) if you want to turn your own trace or contract into a pack.
- [Observability Bridge](integrations.md#mcap-and-foxglove) if you want to export sidecar/MCAP artifacts.
- [How It Fits](concepts.md#where-it-fits) if you are deciding whether this fits an existing stack.

### In a container

To avoid using the host Python environment:

```bash
docker build -t robotprobe:local .
docker run --rm robotprobe:local bash examples/quickstart/run_demo.sh /tmp/robotprobe-demo
```

See [Docker](getting-started.md#docker).

### 4. Run The Pieces Manually

Validate traces:

```bash
robotprobe validate-trace examples/quickstart/traces/clean.jsonl
robotprobe validate-trace examples/quickstart/traces/gripper_freeze_mid.jsonl
```

Summarize:

```bash
robotprobe summarize \
  --clean examples/quickstart/traces/clean.jsonl \
  --perturbed examples/quickstart/traces/gripper_freeze_mid.jsonl \
  --manifest interventions/gripper_freeze_mid.yaml \
  --metrics action_energy,jerk,trajectory_deviation,recovery_latency,near_miss
```

Bundle:

```bash
robotprobe bundle \
  --clean examples/quickstart/traces/clean.jsonl \
  --perturbed examples/quickstart/traces/gripper_freeze_mid.jsonl \
  --manifest interventions/gripper_freeze_mid.yaml \
  --challenge robotprobe-v0-gripper-freeze \
  --metrics action_energy,jerk,trajectory_deviation,recovery_latency,near_miss \
  --title "Quickstart: Gripper Freeze" \
  --out examples/quickstart/out
```

## Simulator demos

RobotProbe can be tried locally against CPU-only simulator episodes. The implemented demos are
small on purpose: they exercise the same `reset` / `step` / policy / action-postprocess loop shape
that appears in larger robotics simulators without requiring a GPU or robot stack.

### Install Optional Simulator Dependencies

```bash
python -m pip install -e '.[simulators]'
```

This installs `gymnasium[classic-control]`. RobotProbe core does not depend on Gymnasium.

MuJoCo and DeepMind Control Suite are optional:

```bash
python -m pip install -e '.[mujoco]'
```

PyBullet is separate because pip wheels vary by platform:

```bash
python -m pip install -e '.[pybullet]'
```

On macOS arm64, `pybullet` may fall back to a source build. If that fails, use a Linux/Docker
environment or `conda-forge`:

```bash
conda create -n robotprobe-pybullet -c conda-forge python=3.10 pybullet
conda run -n robotprobe-pybullet python -c "import pybullet as p; print('pybullet ok')"
```

### Run Gymnasium CartPole

```bash
python examples/simulators/gymnasium_cartpole_demo.py \
  --out examples/simulators/out/gymnasium-cartpole
```

Inspect the result:

```bash
cat examples/simulators/out/gymnasium-cartpole/report.md
cat examples/simulators/out/gymnasium-cartpole/check.json
```

The demo runs two episodes:

- clean CartPole policy episode;
- perturbed episode where a discrete action is flipped after policy output.

RobotProbe records:

- observations;
- raw policy actions;
- postprocessed actions;
- a `DiscreteActionContract` violation;
- outcome and consequence metrics.

Expected diagnosis:

```text
layer: action_postprocess
contract: DiscreteActionContract
fault: action_flip_discrete
RobotSec check: PASS
```

### Run Hidden Keepout Security Demo

```bash
python examples/simulators/security_keepout_demo.py \
  --out examples/simulators/out/security-keepout
```

Inspect the result:

```bash
cat examples/simulators/out/security-keepout/report.md
cat examples/simulators/out/security-keepout/check.json
```

This demo runs two point-robot episodes:

- clean route that reaches the goal while avoiding a keepout zone;
- perturbed route where a post-policy velocity bias crosses the keepout zone but still reaches
  the goal.

The security point is that a deployment dashboard that only watches binary task success would mark
both runs as successful. RobotProbe records the raw policy action, postprocessed action,
`KeepoutZoneContract` check, and outcome fields, so the report can show:

```text
Success changed: false
Raw policy action changed: false
Postprocessed action changed: true
Contract violated: true
Outcome violation changed: true
Hidden success safety violation: true
RobotSec check: PASS
```

The claim boundary matches the other demos: this is a simulator-only demonstration of diagnostic
value, not evidence of hardware exploitability or physical harm.

### Run In Docker

The base RobotProbe Docker image stays small and does not install simulator extras. To run the
Gymnasium demo inside the image:

```bash
docker build -t robotprobe:local .
docker run --rm robotprobe:local bash -lc \
  "python -m pip install '.[simulators]' && python examples/simulators/gymnasium_cartpole_demo.py --out /tmp/gym-cartpole && sed -n '1,80p' /tmp/gym-cartpole/report.md"
```

To write output to the host:

```bash
mkdir -p out/gym-cartpole
docker run --rm \
  -v "$PWD/out/gym-cartpole:/out" \
  robotprobe:local \
  bash -lc "python -m pip install '.[simulators]' && python examples/simulators/gymnasium_cartpole_demo.py --out /out"
```

### Relation To MuJoCo, dm_control, And PyBullet

The local demos are not special. RobotProbe needs three things from a simulator:

1. a reset function;
2. a step function;
3. a place to observe policy input, raw action, postprocessed action, and outcome.

That means the same wrapper pattern applies to:

- Gymnasium classic-control and Gymnasium MuJoCo environments;
- DeepMind Control Suite (`dm_control.suite.load(...).reset()/step()`);
- PyBullet environments, including Gymnasium-compatible wrappers;
- IsaacLab-Arena and other richer robotics eval loops.

RobotProbe should not replace those simulators. It should instrument their episode loops and turn
clean-vs-perturbed simulator runs into challenge-checkable incident bundles.

### Current Boundary

Implemented now:

- Gymnasium-style discrete action demo;
- local hidden keepout-violation security demo;
- ROS 2 MCAP log ingest;
- `vla-eval` episode/result artifact ingest;
- Isaac Lab-Arena JSON/JSONL/CSV trace ingest;
- RobotProbe bundle MCAP export/import;
- sim-only `action_flip_discrete` intervention manifest;
- sim-only `stealth_velocity_bias_keepout` intervention manifest;
- `robotprobe-v1-gymnasium-cartpole-action-flip` RobotSec challenge;
- `robotprobe-v1-security-keepout-bypass` RobotSec challenge.

Not implemented yet:

- first-class MuJoCo adapter;
- first-class dm_control adapter;
- first-class PyBullet adapter;
- rendered video export;
- native Foxglove/Rerun annotation schemas on top of MCAP.

## Docker

RobotProbe can run as a small CPU-only Docker image. This is for local demos, tests, and
host-independent incident-bundle generation. It is not the Isaac Sim / IsaacLab-Arena GPU
container path.

### Build

```bash
docker build -t robotprobe:local .
```

### Run The CLI

```bash
docker run --rm robotprobe:local robotprobe --help
docker run --rm robotprobe:local robotsec-range list-challenges
```

### Run The Quickstart Demo In The Container

```bash
docker run --rm robotprobe:local bash examples/quickstart/run_demo.sh /tmp/robotprobe-demo
```

### Write Demo Output To The Host

```bash
mkdir -p out/docker-quickstart
docker run --rm \
  -v "$PWD/out/docker-quickstart:/out" \
  robotprobe:local \
  bash examples/quickstart/run_demo.sh /out
```

Then inspect:

```bash
cat out/docker-quickstart/report.md
cat out/docker-quickstart/check.json
```

### Run Tests

```bash
docker run --rm robotprobe:local python -m unittest discover -s tests
```

### Scope

This image proves RobotProbe's alpha package, CLI, challenge pack, quickstart traces, and incident
bundle path work without depending on the host Python environment.

This image does not:

- run Isaac Sim;
- run IsaacLab-Arena;
- run GPU VLA inference;
- run Dockerized LIBERO benchmarks;
- provide hardware execution support.
