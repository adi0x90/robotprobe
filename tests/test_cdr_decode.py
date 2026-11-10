"""Tests for robotprobe/integrations/ros2/cdr_decode.py and the CDR ingest path."""

from __future__ import annotations

import io
import math
import unittest

from mcap.reader import make_reader
from mcap_ros2.writer import Writer as McapRos2Writer

from robotprobe.integrations.ros2.cdr_decode import decode_ros2_payload, payload_to_dict


def _cdr_roundtrip(schema_name: str, schema_text: str, payload: dict) -> tuple[dict, dict, bytes]:
    """Build a real CDR-encoded MCAP and return (decoded dict, schema, message)."""
    buf = io.BytesIO()
    writer = McapRos2Writer(buf)
    schema = writer.register_msgdef(schema_name, schema_text)
    writer.write_message("probe_topic", schema, payload, log_time=1, publish_time=2, sequence=3)
    writer.finish()
    blob = buf.getvalue()
    reader = make_reader(io.BytesIO(blob))
    schemas, messages = [], []
    for schema, _channel, message in reader.iter_messages():
        schemas.append(schema)
        messages.append(message)
    assert len(schemas) == 1 and len(messages) == 1
    return decode_ros2_payload(schemas[0], messages[0]), schemas[0], messages[0]


class TestDecodeRos2Payload(unittest.TestCase):
    """CDR decode round trips for representative rosbag2 message shapes."""

    def test_header_roundtrip(self):
        schema_text = (
            "std_msgs/Header header\n===\n"
            "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n"
        )
        decoded, _schema, _message = _cdr_roundtrip(
            "test_msgs/HeaderProbe",
            schema_text,
            {"header": {"stamp": {"sec": 1, "nanosec": 2}, "frame_id": "map"}},
        )
        self.assertEqual(decoded["header"]["stamp"]["sec"], 1)
        self.assertEqual(decoded["header"]["stamp"]["nanosec"], 2)
        self.assertEqual(decoded["header"]["frame_id"], "map")

    def test_nested_type_with_fixed_array(self):
        schema_text = (
            "example_interfaces/Nested child\n===\n"
            "MSG: example_interfaces/Nested\n"
            "builtin_interfaces/Time stamp\nstring frame_id\nfloat64[3] vec\n"
        )
        decoded, _schema, _message = _cdr_roundtrip(
            "test_msgs/NestedProbe",
            schema_text,
            {"child": {"stamp": {"sec": 5, "nanosec": 6}, "frame_id": "base", "vec": [1.0, 2.0, 3.0]}},
        )
        self.assertEqual(
            decoded["child"],
            {"stamp": {"sec": 5, "nanosec": 6}, "frame_id": "base", "vec": [1, 2, 3]},
        )

    def test_primitive_arrays_and_bytes(self):
        decoded, _schema, _message = _cdr_roundtrip(
            "test_msgs/ArraysProbe",
            "uint8[] data\nfloat64[] samples\nbool[] flags\n",
            {"data": b"abc", "samples": [1.0, 2.5, -3.75], "flags": [True, False, True]},
        )
        self.assertEqual(decoded["data"], {"bytes_len": 3})
        self.assertEqual(decoded["samples"], [1.0, 2.5, -3.75])
        self.assertEqual(decoded["flags"], [True, False, True])

    def test_large_bytes_marker_has_no_hex(self):
        decoded, _schema, _message = _cdr_roundtrip(
            "test_msgs/BigBytesProbe", "uint8[] data\n", {"data": bytes(range(256))}
        )
        self.assertEqual(decoded["data"], {"bytes_len": 256})

    def test_multiple_messages_all_decoded(self):
        buf = io.BytesIO()
        writer = McapRos2Writer(buf)
        schema = writer.register_msgdef("test_msgs/TimeProbe", "int32 sec\nuint32 nanosec\n")
        writer.write_message("t", schema, {"sec": 1, "nanosec": 0}, log_time=1)
        writer.write_message("t", schema, {"sec": 2, "nanosec": 5}, log_time=2)
        writer.finish()
        reader = make_reader(io.BytesIO(buf.getvalue()))
        decoded = [decode_ros2_payload(schema, m) for _s, _c, m in reader.iter_messages()]
        self.assertEqual(decoded, [{"sec": 1, "nanosec": 0}, {"sec": 2, "nanosec": 5}])

    def test_bad_schema_text_returns_none(self):
        result = decode_ros2_payload(
            type(
                "S",
                (),
                {"encoding": "ros2msg", "name": "pkg/BadMsg", "data": b"no_such_type/NoSuchType field"},
            )(),
            type("M", (), {"encoding": "cdr", "data": b"\x00\x00\x00\x00"})(),
        )
        self.assertIsNone(result)

    def test_non_ros2msg_schema_encoding_returns_none(self):
        result = decode_ros2_payload(
            type("S", (), {"encoding": "ros2idl", "name": "pkg/X", "data": b"x"})(),
            type("M", (), {"encoding": "cdr", "data": b"\x00\x00\x00\x00"})(),
        )
        self.assertIsNone(result)

    def test_missing_schema_name_returns_none(self):
        result = decode_ros2_payload(
            type("S", (), {"encoding": "ros2msg", "name": "", "data": b"int32 a\n"})(),
            type("M", (), {"encoding": "cdr", "data": b"\x00\x00\x00\x00"})(),
        )
        self.assertIsNone(result)


class TestPayloadToDict(unittest.TestCase):
    """Normalisation of decoded dynamic message objects into plain Python."""

    def test_bytes_represented_as_bytes_len_dict(self):
        self.assertEqual(payload_to_dict(b"\x00\x01\x02"), {"bytes_len": 3})
        self.assertEqual(payload_to_dict(b"\x00" * 65), {"bytes_len": 65})
        self.assertEqual(payload_to_dict(bytearray(b"\x00" * 4)), {"bytes_len": 4})

    def test_large_array_becomes_array_len(self):
        self.assertEqual(payload_to_dict(list(range(200))), {"array_len": 200})

    def test_scalar_passthrough(self):
        self.assertEqual(payload_to_dict(1), 1)
        self.assertEqual(payload_to_dict(1.5), 1.5)
        self.assertEqual(payload_to_dict("x"), "x")
        self.assertEqual(payload_to_dict(True), True)
        self.assertIsNone(payload_to_dict(None))

    def test_nan_inf_list_elements_not_objectified(self):
        self.assertEqual(payload_to_dict([1, 2, 3]), [1, 2, 3])
        result = payload_to_dict([1.0, float("nan")])
        self.assertEqual(result[0], 1.0)
        self.assertIsInstance(result[1], float)
        self.assertTrue(math.isnan(result[1]))
        self.assertEqual(payload_to_dict([1.0, float("inf")]), [1.0, float("inf")])

    def test_float_summary_not_dict(self):
        self.assertEqual(payload_to_dict([1.5, 2.5]), [1.5, 2.5])

    def test_slots_object_normalised(self):
        class Inner:
            __slots__ = ("c",)

            def __init__(self):
                self.c = "x"

        class Outer:
            __slots__ = ("a", "b", "d", "e")

            def __init__(self):
                self.a = 1
                self.b = Inner()
                self.d = [1.5, 2.5]
                self.e = None

        self.assertEqual(payload_to_dict(Outer()), {"a": 1, "b": {"c": "x"}, "d": [1.5, 2.5], "e": None})


class TestIngestRos2McapCdr(unittest.TestCase):
    """End-to-end: CDR-encoded MCAP ingests with structured payloads, not hashes."""

    def test_cdr_ingest_records_value_and_authority(self):
        from robotprobe.integrations.ros2.mcap_adapter import ingest_ros2_mcap
        from robotprobe.core import ReplayLevel

        buf = io.BytesIO()
        writer = McapRos2Writer(buf)
        schema = writer.register_msgdef("test_msgs/StateProbe", "float64[] samples\n")
        writer.write_message("joint_states", schema, {"samples": [0.5, 1.5]}, log_time=1)
        writer.finish()
        blob = buf.getvalue()

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mcap"
            out = Path(tmp) / "out.trace"
            src.write_bytes(blob)
            result = ingest_ros2_mcap(
                src, out, run_id="r1", target="t1", scenario="s1", replay_level=ReplayLevel.TRACE_PLAYBACK
            )
            self.assertEqual(result["event_count"], 1)
            self.assertEqual(result["topics"], {"joint_states": 1})
            content = out.read_text()
            self.assertNotIn("payload_sha256", content)
            self.assertIn("joint_states", content)
            self.assertIn("robot_state", content)
            self.assertIn("ros2_state", content)
            self.assertIn("samples", content)
            self.assertIn("command_authority", content)

    def test_undecodable_cdr_falls_back_to_hash(self):
        from mcap.writer import Writer as RawMcapWriter

        from robotprobe.integrations.ros2.mcap_adapter import ingest_ros2_mcap
        from robotprobe.core import ReplayLevel

        buf = io.BytesIO()
        writer = RawMcapWriter(buf)
        writer.start()
        schema_id = writer.register_schema(
            name="no_such_msgs/BadProbe", encoding="ros2msg", data=b"no_such_type/NoSuchType field\n"
        )
        channel_id = writer.register_channel(
            topic="unknown_topic", message_encoding="cdr", schema_id=schema_id
        )
        payload = b"\x00\x00\x00\x00"
        writer.add_message(
            channel_id=channel_id,
            log_time=1,
            data=payload,
            publish_time=1,
            sequence=0,
        )
        writer.finish()
        blob = buf.getvalue()

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.mcap"
            out = Path(tmp) / "out.trace"
            src.write_bytes(blob)
            result = ingest_ros2_mcap(
                src, out, run_id="r1", target="t1", scenario="s1", replay_level=ReplayLevel.TRACE_PLAYBACK
            )
            self.assertEqual(result["event_count"], 1)
            content = out.read_text()
            self.assertIn("payload_sha256", content)
            self.assertIn("payload_size_bytes", content)
            self.assertNotIn('"value"', content)


if __name__ == "__main__":
    unittest.main()
