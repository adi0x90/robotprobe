# robotprobe-pack-isaac-lab-arena-adapter

This RobotProbe pack is scaffolded as `sim_only`.

Executable entrypoint:

```text
robotprobe.integrations.isaac_lab_arena.trace_ingest:ingest_isaac_lab_arena_trace
```

Ingest an Isaac Lab-Arena exported trace:

```bash
robotprobe ingest isaac-lab-arena \
  --input /path/to/isaac_trace.jsonl \
  --out traces/isaac_lab_arena_episode.jsonl \
  --run-id isaac_run_001 \
  --target isaac-lab-arena \
  --scenario lift_cube
```

Validate it with:

```bash
robotprobe pack validate examples/packs/isaac-lab-arena-adapter
```
