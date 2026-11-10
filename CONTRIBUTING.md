# Contributing To RobotProbe

RobotProbe is pack-first. Most useful contributions should be a pack, fixture, adapter, contract,
incident report, or documentation improvement before they become core library changes.

## What RobotProbe Is Trying To Preserve

- Evidence first: every diagnosis should point to traces, manifests, checks, or reports.
- Claim boundaries: local/synthetic evidence must not be described as live-model benchmark evidence.
- Safety by default: public contributions must not ship hardware-destructive interventions.
- Compatibility: RobotProbe should sit beside ROS 2, MCAP, Foxglove, Rerun, vla-eval, Isaac, and
  simulator tooling rather than replacing them.

## Best First Contributions

1. Add a new incident pack from a public or recorded trace source.
2. Add a contract pack for a specific robot-stack boundary.
3. Add a golden diagnosis fixture for an existing pack.
4. Add an adapter from an exported log format into RobotProbe JSONL.
5. Improve a walkthrough with commands that you actually ran.

Use [Pack Authoring Walkthrough](docs/packs.md#authoring-a-pack) before changing core library code.

## Safety Classes

Every pack plugin declares a release classification:

- `trace_only`: parses or diagnoses recorded data only.
- `sim_only`: mutates simulator or synthetic traces only.
- `hardware_shadow`: checks commands or envelopes without commanding hardware.
- `hardware_executable`: can command or mutate real hardware behavior.

Public RobotProbe packs should be `trace_only`, `sim_only`, or `hardware_shadow`. `hardware_executable`
packs are blocked by default and are not accepted as public alpha contributions.

## Development Environment

```bash
make setup
uv run robotprobe doctor
uv run python -m pytest
uv run ruff check .
```

If you do not use `uv`, install the package with:

```bash
python -m pip install -e .
```

### Dependency groups

- core: `mcap`, `numpy`, `PyYAML`, `safetensors` (always installed).
- `dev`: `pytest`, `ruff`, and `mcap-ros2-support` — a PEP 735 **dependency-group**, not a
  published extra, installed by `uv sync` by default. The library stays `pip install`-able with no
  uv requirement for consumers.
- `ros2-decoders`: `mcap-ros2-support`, for users who want CDR decoding at runtime.
- `simulators`: Gymnasium classic control (for the simulator demos and their tests).
- `mujoco` / `pybullet` / `incident-datasets` / `nnsight`: heavier optional extras, on demand.

The ROS 2 CDR tests import `mcap_ros2` at module scope, which is why it is in the dev group rather
than only in an extra. Without it, collection fails instead of skipping.

### Test layout

Tests are `unittest.TestCase` classes under `tests/`, runnable with `pytest` or
`python -m unittest discover -s tests`. Bare-function collection is disabled in
`[tool.pytest.ini_options]` so the `test_pack()` entrypoint imported from `robotprobe.packs` is not
mistaken for a test.

`tests/test_repo_hygiene.py` scans the tree for private paths, cloud endpoints, and maintainer-note
references. If it fails, it is usually telling you that a document or fixture picked up something
that should not be published.

## Required Checks Before A Pull Request

Run the smallest relevant checks plus the full local suite:

```bash
uv run robotprobe doctor
bash examples/quickstart/run_demo.sh /tmp/robotprobe-contrib-quickstart
uv run robotprobe pack claims examples/packs/libero-action-contracts
uv run robotprobe pack test examples/packs/libero-action-contracts
uv run python -m pytest
uv run ruff check .
```

For pack contributions, also run:

```bash
uv run robotprobe pack validate path/to/your-pack
uv run robotprobe pack test path/to/your-pack
uv run robotprobe pack claims path/to/your-pack
```

If your pack declares `execution.imports_code_during_test: true`, use `--allow-code` only when the
entrypoint is intentionally imported during tests and the code does not perform network, hardware,
or model-runtime side effects.

## Evidence Expectations

For every new incident, include:

- the source trace or a clear source reference;
- intervention manifest if the incident is perturbed;
- `summary.json` or golden expected summary;
- `check.json` or challenge result when applicable;
- `report.md` for human review;
- claim/non-claim notes that say what the trace proves and what it does not prove.

Do not claim:

- broad model robustness from one pack;
- live OpenVLA/GR00T/Pi0.5 behavior from synthetic traces;
- MechProbe causal explanations from offline activation correlation alone;
- real hardware exploitability from simulator-only traces.

## Documentation Style

Every walkthrough should answer four questions:

1. What real-world situation is this mimicking?
2. What command should a user run?
3. What artifacts should appear?
4. What should the user not claim from this result?

Prefer short runnable command blocks over architecture prose that cannot be verified.

## Licensing

RobotProbe is licensed under Apache-2.0. By submitting a pull request you agree that your
contribution is licensed under the same terms. If you are contributing on behalf of an employer,
make sure you have permission to do so.

A pack you publish separately keeps its own license. The pack format, the example packs, and the
code in this repository are Apache-2.0.
