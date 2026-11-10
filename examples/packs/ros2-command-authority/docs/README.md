# robotprobe-pack-ros2-command-authority

This RobotProbe pack is scaffolded as `trace_only`.

Executable entrypoint:

```text
robotprobe.integrations.ros2.mcap_adapter:ingest_ros2_mcap
```

Ingest a ROS 2 MCAP log:

```bash
robotprobe ingest ros2-mcap \
  --input /path/to/ros2_bag.mcap \
  --out traces/ros2_command_authority.jsonl \
  --run-id ros2_run_001 \
  --target ros2 \
  --scenario command_authority_replay
```

Validate it with:

```bash
robotprobe pack validate examples/packs/ros2-command-authority
```
