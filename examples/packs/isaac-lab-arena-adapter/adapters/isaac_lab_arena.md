# Isaac Lab-Arena Adapter

This template is for mapping Isaac Lab-Arena episodes into RobotProbe traces and incident bundles.
It should stay simulator-only until a project adds explicit hardware-shadow safety envelopes.

Executable ingest path:

```bash
robotprobe ingest isaac-lab-arena --input isaac_trace.jsonl --out trace.jsonl --run-id isaac_run --scenario lift_cube
```

The adapter accepts exported JSON, JSONL, or CSV episode rows with state/action/success fields. It
does not launch Isaac Sim or Isaac Lab-Arena locally.
