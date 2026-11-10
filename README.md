# RobotProbe

RobotProbe records typed traces from a robot or VLA stack, compares a clean run against a
perturbed one, and reports where they first diverged and which layer changed.

It is a diagnosis layer that sits beside MCAP, Foxglove, Rerun, and ROS tracing. It does not replace
them, and it is not a simulator or a benchmark.

## Quickstart

No account, no GPU, no robot. `demo` runs a bundled incident end to end and writes a report.

```bash
python -m pip install --pre robotprobe
robotprobe demo             # writes ./robotprobe-demo-out
cat robotprobe-demo-out/report.md
```

The report looks like this (abridged):

```text
# Quickstart: Gripper Freeze

## Diagnosis

- Layer: `action_postprocess`
- Declared by manifest: `action_postprocess`
- Contract: `ActionSpaceContract`
- Fault: `gripper_freeze_mid`
- Layer match score: `0.87`
- RobotSec check: `PASS`

## Evidence

- First divergence step: `35`
- Raw policy action changed: `false`
- Postprocessed action changed: `true`
- Contract violated: `true`
```

The task still succeeded, the raw policy output was unchanged, and the fault entered in action
post-processing at step 35.

The bundled demo traces are synthetic: short hand-written fixtures for a placeholder target
(`quickstart-vla`), not a recording from a real policy. They exist to show the shape of a diagnosis.

`Layer` is computed from the traces, but the manifest decides which comparison is used. Under an
`action_postprocess` manifest, raw-policy differences between the two runs are treated as rollout
noise, so the observed layer can only be `action_postprocess` or `unknown`. Agreement with
`Declared by manifest` is therefore expected by construction for a post-policy fault; it is not
independent confirmation.

`Layer match score` is a fixed value assigned by the layer rule, not a calibrated probability. See
[Claims](https://github.com/adi0x90/robotprobe/blob/main/docs/status-and-claims.md#claims) for exactly what the fields mean.

## Install

Install the alpha package from PyPI:

```bash
python -m pip install --pre robotprobe
```

For the source examples and development tools, clone the repository:

```bash
git clone https://github.com/adi0x90/robotprobe && cd robotprobe
make setup                  # core library + dev tooling (this is all most people need)
```

Simulators and the white-box activation bridge are opt-in:

```bash
make setup-simulators       # gymnasium / mujoco / pybullet
make setup-nnsight          # nnsight + torch (MechProbe only)
make setup-all              # everything
```

Or with pip, from a checkout:

```bash
python -m pip install -e .                    # core
python -m pip install -e '.[simulators]'      # + Gymnasium classic control
python -m pip install -e '.[ros2-decoders]'   # + ROS 2 MCAP/CDR decoding (no ROS install)
```

The `--pre` option includes alpha releases. The source checkout is needed for
commands below that reference `examples/` or `scripts/`.

## What You Can Do

| Task | Command |
| --- | --- |
| Diagnose a regression | `robotprobe summarize --clean ... --perturbed ... --manifest ...` |
| Check a trace | `robotprobe validate-trace trace.jsonl` |
| Score a diagnosis against a challenge | `robotsec-range check --challenge ... --summary ...` |
| Run a fault campaign over a baseline | `robotprobe campaign run --campaign ... --out ...` |
| Author a reusable pack | `robotprobe pack init /tmp/my-pack --template ros2-command-authority` |
| Move evidence into MCAP and back | `robotprobe export mcap` / `robotprobe import mcap` |
| Ingest a ROS 2 bag | `robotprobe ingest ros2-mcap --input bag.mcap --out traces/run.jsonl` |
| Diff model activations (experimental) | `robotprobe mech collect` / `diff-activations` |

Full command reference: [Command Reference](https://github.com/adi0x90/robotprobe/blob/main/docs/cli.md).

## Franka runtime

RobotProbe also ships a modular runtime that runs one policy, scheduler, safety gateway, and robot
backend under a single trace. The bundled pack targets a Franka arm with PI0.5.

### Choose your Franka demo

All three demos are CPU-only views over the same runtime evidence:

| Audience | Demo | What it shows |
| --- | --- | --- |
| Security and assurance | **Safety Gate** | An oversized action is rejected before backend execution, with zero receipts and zero applied commands. |
| Robotics engineering | **Backend Swap** | One runner swaps between a deterministic kinematic contract and a synthetic recorded-format fixture. PI0.5/LIBERO and real-Franka shadow stay optional. |
| Teaching | **Action Journey** | Stable IDs follow one observation through policy, action chunk, safety, execution, measured state, and outcome. |

```bash
uv run robotprobe pack validate examples/packs/franka-pi05-runtime
uv run python examples/demos/franka/run_demos.py --demo all --out .artifacts/franka-demos
```

The local suite does not load a learned model or simulate physics. LIBERO simulation needs a
separate Linux/CUDA run. The real-Franka path observes only; physical actuation is not implemented.

See [`docs/franka.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/franka.md) for the demos, the GPU PI0.5/LIBERO run, and the read-only
real-robot path, and [`docs/runtime.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/runtime.md) for the runtime architecture and backend
authoring.

## Documentation

| If you want to… | Read |
| --- | --- |
| install and run the demos | [`docs/getting-started.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/getting-started.md) |
| understand traces, diffs, and how a diagnosis is read | [`docs/concepts.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/concepts.md) |
| look up a command | [`docs/cli.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/cli.md) |
| encode your own failure mode as a pack | [`docs/packs.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/packs.md) |
| bring in ROS 2, MCAP, vla-eval, or dataset recordings | [`docs/integrations.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/integrations.md) |
| run a policy through the modular runtime | [`docs/runtime.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/runtime.md) |
| use the Franka pack or run PI0.5 on LIBERO | [`docs/franka.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/franka.md) |
| capture and compare model activations | [`docs/mechprobe.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/mechprobe.md) |
| try campaigns, live instrumentation, or world-model checks | [`docs/experimental.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/experimental.md) |
| know what works, what is claimed, and the evidence | [`docs/status-and-claims.md`](https://github.com/adi0x90/robotprobe/blob/main/docs/status-and-claims.md) |

## Status

`v0.2.0-alpha` is a research preview. The local path, the CLI, pack authoring, the simulators, the
Franka runtime demos, and offline MechProbe analysis work on a CPU. The PI0.5/LIBERO path has run on
a GPU for one checkpoint, task, and seed, with mixed diagnostic results; see
[Evidence](https://github.com/adi0x90/robotprobe/blob/main/docs/status-and-claims.md#evidence) for what was and was not established.

## Development

```bash
make test     # uv run pytest
make lint     # uv run ruff check
make ci       # lint, tests, doctor, and a demo smoke test
```

GitHub Actions runs the same checks on push and pull request. See
[`CONTRIBUTING.md`](https://github.com/adi0x90/robotprobe/blob/main/CONTRIBUTING.md).

Maintainers: see [Publishing](https://github.com/adi0x90/robotprobe/blob/main/docs/publishing.md) for the PyPI release workflow.

## Safety

Bundled interventions are simulation-only: every manifest declares
`release_classification: sim_only` and `hardware_allowed: false`. Nothing here is validated for real
hardware. See [Safety Policy](https://github.com/adi0x90/robotprobe/blob/main/docs/safety.md).

## Security

Report suspected vulnerabilities privately; see [`SECURITY.md`](https://github.com/adi0x90/robotprobe/blob/main/SECURITY.md).

## Third-party names

Franka, LIBERO, PI0.5, LeRobot, MuJoCo, Isaac, OpenVLA, GR00T, Hugging Face, Foxglove, Rerun, and
other product names are trademarks of their respective owners. They appear here only to describe
what this code interoperates with. This project is not affiliated with, sponsored by, or endorsed by
any of them, and it redistributes no vendor SDKs, model weights, or benchmark datasets.

## License

Apache-2.0. See [LICENSE](https://github.com/adi0x90/robotprobe/blob/main/LICENSE).
