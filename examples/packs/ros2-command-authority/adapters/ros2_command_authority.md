# ROS 2 Command Authority Adapter

This template is for trace-only ROS 2 command-authority audits. It is meant to record and inspect
who produced command messages, where supervision gates sit, and whether a bypass proxy appears in
the command path.

Executable ingest path:

```bash
robotprobe ingest ros2-mcap --input bag.mcap --out trace.jsonl --run-id ros2_run --scenario command_authority
```

JSON-encoded MCAP messages are decoded into RobotProbe values. Binary ROS 2 messages are still
ingested with topic, schema, timing, payload hash, and payload size metadata so authority/freshness
analysis can proceed without a local ROS 2 workspace.
