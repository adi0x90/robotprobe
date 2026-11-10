"""CDR decoding for ROS 2 MCAP payloads.

Real rosbag2 files store message payloads CDR-encoded (``message_encoding="cdr"``)
against a ``ros2msg`` schema. The optional ``mcap-ros2-support`` extra provides the
decoder; this module makes it optional: every failure mode here degrades to
``None`` so the ingest adapter falls back to hashing the payload.
"""

from __future__ import annotations

from typing import Any

# Numeric sequences large enough that per-element JSON numbers are not worth
# emitting; the trace keeps the element count instead.
_ARRAY_ELEMENT_LIMIT = 128


def decode_ros2_payload(schema: Any, message: Any) -> dict | None:
    """Decode one MCAP ``(Schema, Message)`` pair to a plain dict.

    Returns ``None`` when the payload cannot be decoded: unknown schema
    encoding, absent optional dependency, or any decode error. Never raises.
    """
    try:
        schema_encoding = str(getattr(schema, "encoding", "") or "")
        type_name = str(getattr(schema, "name", "") or "")
        schema_data = getattr(schema, "data", b"")
        message_data = getattr(message, "data", None)
    except Exception:
        return None
    if schema_encoding.lower() != "ros2msg" or not type_name or not schema_data:
        return None
    if not isinstance(message_data, (bytes, bytearray, memoryview)):
        return None
    try:
        # Lazy import: the optional ``ros2-decoders`` extra may be absent.
        from mcap_ros2.decoder import generate_dynamic

        decoders = generate_dynamic(type_name, bytes(schema_data).decode("utf-8"))
        decoder = decoders.get(type_name)
        if decoder is None:
            if not decoders:
                return None
            decoder = next(iter(decoders.values()))
        obj = decoder(bytes(message_data))
    except Exception:
        return None
    try:
        result = payload_to_dict(obj)
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def payload_to_dict(obj: Any) -> dict | list | str | int | float | bool | None:
    """Normalise a decoded dynamic ROS 2 message into JSON-serialisable Python."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return {"bytes_len": len(obj)}
    if isinstance(obj, dict):
        return {key: payload_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, bytearray):
        return {"bytes_len": len(obj)}
    if isinstance(obj, (list, tuple)):
        return _normalise_sequence(obj)
    # Dynamic ROS 2 message classes expose fields via ``__slots__`` attributes.
    return _normalise_fields(obj)


def _normalise_sequence(obj: list | tuple) -> list:
    if not obj:
        return []
    first = obj[0]
    if isinstance(first, (list, tuple, bytes, bytearray, dict)):
        return [payload_to_dict(element) for element in obj]
    flat = _flatten_sequence(obj)
    if flat is None:
        return [payload_to_dict(element) for element in obj]
    return _numeric_list_summary(flat)


def _flatten_sequence(obj: list | tuple) -> list | None:
    """Flatten a 1D/2D numeric sequence into scalars; ``None`` if unflattenable."""
    flat: list = []
    for element in obj:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            return None
        flat.append(element)
    return flat


def _numeric_list_summary(flat: list) -> list | dict:
    if len(flat) > _ARRAY_ELEMENT_LIMIT:
        return {"array_len": len(flat)}
    values = [float(element) for element in flat]
    if any(value != value or value in (float("inf"), float("-inf")) for value in values):
        return [element if isinstance(element, int) else float(element) for element in flat]
    # Exact integral values stay ints; otherwise round to 6 decimals to kill
    # float64 repr noise while preserving distinguishable magnitudes.
    rounded = [round(value, 6) for value in values]
    if all(value == int(value) for value in rounded):
        return [int(value) for value in rounded]
    return rounded


def _normalise_fields(obj: Any) -> dict:
    slots = getattr(type(obj), "__slots__", None)
    names = slots if isinstance(slots, (list, tuple)) else None
    if names is None:
        names = [name for name in dir(obj) if not name.startswith("_")]
    fields: dict[str, Any] = {}
    for name in names:
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        if callable(value):
            continue
        fields[name] = payload_to_dict(value)
    return fields
