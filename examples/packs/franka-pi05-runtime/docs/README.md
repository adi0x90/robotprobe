# Franka modular runtime pack

This pack composes policy, embodiment, backend, and scenario plugins through one traced episode
runner. It contains:

- a deterministic Franka contract episode;
- a synthetic recorded-format, read-only replay episode;
- an optional PI0.5 plus LIBERO/MuJoCo simulated path;
- configurable PI0.5 semantic activation capture; and
- a read-only real-Franka shadow seam.

Inspect any configuration without importing its heavyweight entry points:

```bash
uv run robotprobe runtime inspect \
  --config examples/packs/franka-pi05-runtime/configs/deterministic.yaml
uv run robotprobe runtime inspect \
  --config examples/packs/franka-pi05-runtime/configs/franka-recorded-replay.yaml
uv run robotprobe runtime inspect \
  --config examples/packs/franka-pi05-runtime/configs/pi05-libero.yaml
uv run robotprobe runtime inspect \
  --config examples/packs/franka-pi05-runtime/configs/franka-ros2-shadow.yaml
```

Run the local deterministic contract episode:

```bash
uv run robotprobe runtime run \
  --config examples/packs/franka-pi05-runtime/configs/deterministic.yaml
uv run robotprobe validate-trace \
  examples/packs/franka-pi05-runtime/artifacts/deterministic-trace.jsonl
```

The deterministic backend proves runtime composition and trace lineage. It is not a physics
simulator and does not model contacts. Injected tests prove optional adapter contracts without
claiming that the checkpoint, LIBERO, or a real robot ran locally.

The recorded replay backend is also simulation-only. It evaluates actions against a synthetic,
recorded-format snapshot and always records `command_applied: false`; it is not physical Franka
evidence.

The hardware-shadow backend has no command method, advertises no hardware-command capability, and
must record `command_applied: false`. It does not prove physical task success or safe action
conversion.

For the complete walkthrough, see the repository-level
[Franka evidence demo guide](../../../../docs/franka.md#cpu-demos). Architecture, authoring,
heavyweight validation, and physical-shadow migration remain in the other Franka guides under
`docs/`.
