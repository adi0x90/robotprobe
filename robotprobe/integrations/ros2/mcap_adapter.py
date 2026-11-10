from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mcap.reader import make_reader

from robotprobe.integrations.ros2.cdr_decode import decode_ros2_payload
from robotprobe.core import ReplayLevel
from robotprobe.record import TraceRecorder


COMMAND_TOPIC_HINTS = ("cmd", "command", "action", "trajectory", "twist")
STATE_TOPIC_HINTS = ("joint_states", "state", "odom", "tf", "pose")


def ingest_ros2_mcap(
    input_path: str | Path,
    out_path: str | Path,
    *,
    run_id: str,
    target: str,
    scenario: str,
    replay_level: ReplayLevel = ReplayLevel.TRACE_PLAYBACK,
) -> dict[str, Any]:
    source = Path(input_path)
    out = Path(out_path)
    event_count = 0
    topics: dict[str, int] = {}

    with TraceRecorder(
        out,
        run_id=run_id,
        target=target,
        scenario=scenario,
        replay_level=replay_level,
    ) as recorder, source.open("rb") as f:
        reader = make_reader(f)
        for step, (schema, channel, message) in enumerate(reader.iter_messages()):
            payload = _decode_payload(schema, message, channel.message_encoding)
            event_type, layer = _classify_topic(channel.topic)
            fields: dict[str, Any] = {
                "topic": channel.topic,
                "message_encoding": channel.message_encoding,
                "schema_name": getattr(schema, "name", None),
                "log_time_ns": message.log_time,
                "publish_time_ns": message.publish_time,
            }
            if payload is not None:
                fields["value"] = payload
                fields["command_authority"] = _command_authority(channel.topic, payload)
            else:
                fields["payload_sha256"] = hashlib.sha256(message.data).hexdigest()
                fields["payload_size_bytes"] = len(message.data)
                fields["command_authority"] = {"source": f"topic:{channel.topic}"}
            recorder.event(step, event_type, layer, **fields)
            event_count += 1
            topics[channel.topic] = topics.get(channel.topic, 0) + 1

    return {
        "format": "robotprobe.trace.v0",
        "source_format": "ros2.mcap",
        "input": str(source),
        "out": str(out),
        "event_count": event_count,
        "topics": topics,
    }


def _decode_payload(schema: Any, message: Any, message_encoding: str) -> Any | None:
    """Decode a message payload, JSON first, then CDR (ros2msg schemas).

    Returns ``None`` when the payload cannot be decoded, in which case the
    caller records a hash fallback instead of a structured payload.
    """
    if message_encoding.lower() == "json":
        try:
            return json.loads(message.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    return decode_ros2_payload(schema, message)


def _classify_topic(topic: str) -> tuple[str, str]:
    normalized = topic.lower()
    if any(hint in normalized for hint in COMMAND_TOPIC_HINTS):
        return "postprocessed_action", "ros2_command"
    if any(hint in normalized for hint in STATE_TOPIC_HINTS):
        return "robot_state", "ros2_state"
    return "observation", "ros2_topic"


def _command_authority(topic: str, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        authority = payload.get("command_authority")
        if isinstance(authority, dict):
            return authority
        source = payload.get("source")
        if isinstance(source, str):
            return {"source": source}
    return {"source": f"topic:{topic}"}
