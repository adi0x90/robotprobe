# Franka evidence demos

These demos are three views over one RobotProbe runtime-evidence path:

- `safety-gate` compares an accepted deterministic action with an oversized action rejected before
  backend execution;
- `backend-swap` executes deterministic and read-only replay backends, then statically inspects the
  optional LIBERO and physical-shadow configurations; and
- `action-journey` follows stable observation, policy-call, action-chunk, command, safety, receipt,
  and outcome identifiers.

Run the complete CPU-only suite from the repository root:

```bash
uv run python examples/demos/franka/run_demos.py \
  --demo all \
  --out .artifacts/franka-demos
```

The output directory must be absent or empty. Generated traces, evidence JSON, Markdown reports,
configuration copies, and SHA256 manifests are written beneath it.

The deterministic backend is a kinematic contract, not a physics simulator. The replay backend
uses a synthetic recorded-format snapshot and is not physical-robot evidence. PI0.5/LIBERO and
Franka ROS 2 appear as `inspect_only` unless their separate prerequisite-aware paths are actually
run.
