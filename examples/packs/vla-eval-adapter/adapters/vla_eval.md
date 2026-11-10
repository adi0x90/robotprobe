# vla-eval Adapter

This template is for wrapping AllenAI-style VLA evaluation loops with RobotProbe trace hooks,
paired run IDs, intervention manifests, and RobotSec checks.

Executable ingest path:

```bash
robotprobe ingest vla-eval --input episode.json --out trace.jsonl --run-id vla_eval_run --scenario libero_spatial
```

The adapter accepts merged per-episode JSON/JSONL-style artifacts and converts task text,
observations, actions, and success into RobotProbe trace events. It does not launch the upstream
benchmark container.
