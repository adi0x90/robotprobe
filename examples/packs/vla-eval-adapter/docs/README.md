# robotprobe-pack-vla-eval-adapter

This RobotProbe pack is scaffolded as `sim_only`.

Executable entrypoint:

```text
robotprobe.integrations.vla_eval.episode_ingest:ingest_vla_eval_episode
```

Ingest a vla-eval episode/result artifact:

```bash
robotprobe ingest vla-eval \
  --input /path/to/vla_eval_episode.json \
  --out traces/vla_eval_episode.jsonl \
  --run-id vla_eval_run_001 \
  --target vla-eval \
  --scenario libero_spatial
```

Validate it with:

```bash
robotprobe pack validate examples/packs/vla-eval-adapter
```
