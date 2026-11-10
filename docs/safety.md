# Safety Policy

## Simulation-first

RobotProbe alpha artifacts are simulation-first. Built-in interventions are intended for trace
playback, offline replay, simulator replay, or hardware-shadow validation with command envelopes.

No hardware-destructive intervention is shipped. Built-in manifests must keep:

```yaml
release_classification: sim_only
safety:
  sim_only: true
  hardware_allowed: false
```

Any hardware-connected use requires a separate target-specific safety review, explicit command
envelopes, emergency stop handling, and operator supervision. Those controls are outside the
scope of this alpha.

## Modular Franka runtime

The Franka runtime has three distinct evidence modes:

- the deterministic backend is a kinematic contract test, not physics or contact evidence;
- the LIBERO backend applies actions only inside simulation; and
- the physical Franka backend is hardware-shadow and read-only.

The shadow transport interface exposes health and timestamped snapshots only. It cannot publish or
send a motion command. Shadow authorization is recorded as `shadow_only`; its receipt must contain
`command_applied: false`. The receipt type, safety gateway, runner, backend, manifest, and tests all
enforce that invariant.

The current pack contains no `hardware_executable` backend. Its only hardware component declares:

```yaml
release_classification: hardware_shadow
hardware_allowed: false
capabilities:
  is_hardware: true
  supports_shadow: true
  can_apply_hardware_commands: false
```

Connecting a powered robot still requires operator supervision, emergency-stop readiness, verified
camera calibration, clock freshness, and controller authority review. Shadow mode does not certify
those controls.

## No configuration-only promotion to motion

Hardware execution cannot be enabled by changing `execution.mode`. Compatibility checks reject the
current shadow backend in simulation and hardware-executable modes.

A motion-capable Franka integration must be a new plugin with a separate safety review. It must
close action normalization, frame, calibration, rate, workspace, collision, force, watchdog,
fault-recovery, command-acknowledgement, and operator-arming gates before physical validation.

See [real Franka shadow migration](franka.md#real-franka-read-only) for the full boundary.
