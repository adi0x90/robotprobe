from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap.writer import Writer


MCAP_FORMAT = "robotprobe.mcap.v0"
MCAP_TOPICS = {
    "bundle": "/robotprobe/bundle",
    "summary": "/robotprobe/summary",
    "causal_diff": "/robotprobe/causal_diff",
    "check": "/robotprobe/check",
    "report": "/robotprobe/report",
    "manifest": "/robotprobe/manifest",
    "clean_trace": "/robotprobe/traces/clean",
    "perturbed_trace": "/robotprobe/traces/perturbed",
}


def export_sidecar(bundle_dir: str | Path, out_path: str | Path) -> dict[str, Any]:
    bundle_root = Path(bundle_dir)
    bundle = _read_json(bundle_root / "bundle.json")
    summary = _read_optional_json(bundle_root / str(bundle.get("summary_ref", "")))
    check = _read_optional_json(bundle_root / str(bundle.get("check_ref", "")))
    sidecar = {
        "format": "robotprobe.sidecar.v0",
        "producer": "robotprobe",
        "bundle": bundle,
        "summary": summary,
        "check": check,
        "refs": {
            "report": bundle.get("report_ref"),
            "clean_trace": bundle.get("clean_trace_ref"),
            "perturbed_trace": bundle.get("perturbed_trace_ref"),
            "manifest": bundle.get("manifest_ref"),
        },
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def export_mcap_bundle(bundle_dir: str | Path, out_path: str | Path) -> dict[str, Any]:
    bundle_root = Path(bundle_dir)
    bundle = _read_json(bundle_root / "bundle.json")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    message_count = 0
    topics: dict[str, int] = {}
    with out.open("wb") as f:
        writer = Writer(f)
        writer.start()
        channels = {
            name: writer.register_channel(
                topic=topic,
                message_encoding="json",
                schema_id=0,
            )
            for name, topic in MCAP_TOPICS.items()
        }

        def write_json(name: str, payload: dict[str, Any], timestamp_ns: int = 0) -> None:
            nonlocal message_count
            writer.add_message(
                channel_id=channels[name],
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            )
            message_count += 1
            topics[MCAP_TOPICS[name]] = topics.get(MCAP_TOPICS[name], 0) + 1

        write_json("bundle", bundle)
        _write_optional_json_payload(bundle_root, bundle, "summary_ref", "summary", write_json)
        _write_optional_json_payload(bundle_root, bundle, "causal_diff_ref", "causal_diff", write_json)
        _write_optional_json_payload(bundle_root, bundle, "check_ref", "check", write_json)
        _write_optional_text_payload(bundle_root, bundle, "report_ref", "report", write_json)
        _write_optional_text_payload(bundle_root, bundle, "manifest_ref", "manifest", write_json)
        _write_trace_payloads(bundle_root, bundle, "clean_trace_ref", "clean_trace", write_json)
        _write_trace_payloads(bundle_root, bundle, "perturbed_trace_ref", "perturbed_trace", write_json)
        writer.finish()

    return {
        "format": MCAP_FORMAT,
        "path": str(out),
        "message_count": message_count,
        "topics": topics,
    }


def import_mcap_bundle(mcap_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    source = Path(mcap_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    causal_diff: dict[str, Any] | None = None
    check: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    traces: dict[str, list[dict[str, Any]]] = {"clean": [], "perturbed": []}
    imported_topics: dict[str, int] = {}

    with source.open("rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages():
            if not channel.topic.startswith("/robotprobe/"):
                continue
            payload = json.loads(message.data.decode("utf-8"))
            imported_topics[channel.topic] = imported_topics.get(channel.topic, 0) + 1
            if channel.topic == MCAP_TOPICS["bundle"]:
                bundle = payload
            elif channel.topic == MCAP_TOPICS["summary"]:
                summary = payload
            elif channel.topic == MCAP_TOPICS["causal_diff"]:
                causal_diff = payload
            elif channel.topic == MCAP_TOPICS["check"]:
                check = payload
            elif channel.topic == MCAP_TOPICS["report"]:
                report = payload
            elif channel.topic == MCAP_TOPICS["manifest"]:
                manifest = payload
            elif channel.topic == MCAP_TOPICS["clean_trace"]:
                traces["clean"].append(payload)
            elif channel.topic == MCAP_TOPICS["perturbed_trace"]:
                traces["perturbed"].append(payload)

    restored_bundle = _default_bundle(bundle)
    if causal_diff is not None:
        restored_bundle.setdefault("causal_diff_ref", "causal_diff.json")
    _write_json_file(out / "bundle.json", restored_bundle)
    if summary is not None:
        _write_json_file(_safe_bundle_path(out, restored_bundle.get("summary_ref"), "summary.json"), summary)
    if causal_diff is not None:
        _write_json_file(
            _safe_bundle_path(out, restored_bundle.get("causal_diff_ref"), "causal_diff.json"), causal_diff
        )
    if check is not None:
        _write_json_file(_safe_bundle_path(out, restored_bundle.get("check_ref"), "check.json"), check)
    if report is not None:
        _write_text_file(
            _safe_bundle_path(out, restored_bundle.get("report_ref"), str(report.get("path", "report.md"))),
            str(report.get("text", "")),
        )
    if manifest is not None:
        _write_text_file(
            _safe_bundle_path(out, restored_bundle.get("manifest_ref"), str(manifest.get("path", "manifests/manifest.yaml"))),
            str(manifest.get("text", "")),
        )
    if traces["clean"]:
        _write_jsonl_file(
            _safe_bundle_path(out, restored_bundle.get("clean_trace_ref"), "traces/clean.jsonl"),
            traces["clean"],
        )
    if traces["perturbed"]:
        _write_jsonl_file(
            _safe_bundle_path(out, restored_bundle.get("perturbed_trace_ref"), "traces/perturbed.jsonl"),
            traces["perturbed"],
        )

    return {
        "format": "robotprobe.bundle.v0",
        "path": str(out),
        "topics": imported_topics,
        "message_count": sum(imported_topics.values()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.name:
        return None
    return _read_json(path)


def _write_optional_json_payload(
    bundle_root: Path,
    bundle: dict[str, Any],
    ref_key: str,
    topic_name: str,
    write_json: Any,
) -> None:
    path = _bundle_ref(bundle_root, bundle.get(ref_key))
    if path is None or not path.exists():
        return
    write_json(topic_name, _read_json(path))


def _write_optional_text_payload(
    bundle_root: Path,
    bundle: dict[str, Any],
    ref_key: str,
    topic_name: str,
    write_json: Any,
) -> None:
    path = _bundle_ref(bundle_root, bundle.get(ref_key))
    if path is None or not path.exists():
        return
    write_json(topic_name, {"path": str(path.relative_to(bundle_root)), "text": path.read_text(encoding="utf-8")})


def _write_trace_payloads(
    bundle_root: Path,
    bundle: dict[str, Any],
    ref_key: str,
    topic_name: str,
    write_json: Any,
) -> None:
    path = _bundle_ref(bundle_root, bundle.get(ref_key))
    if path is None or not path.exists():
        return
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        step = event.get("step", index)
        timestamp_ns = _step_to_timestamp_ns(step, index)
        write_json(topic_name, event, timestamp_ns=timestamp_ns)


def _bundle_ref(bundle_root: Path, ref: Any) -> Path | None:
    if not ref:
        return None
    return bundle_root / str(ref)


def _step_to_timestamp_ns(step: Any, fallback_index: int) -> int:
    if isinstance(step, bool):
        return fallback_index * 1_000_000_000
    if isinstance(step, (int, float)):
        return int(step * 1_000_000_000)
    return fallback_index * 1_000_000_000


def _default_bundle(bundle: dict[str, Any] | None) -> dict[str, Any]:
    restored = dict(bundle or {})
    restored.setdefault("summary_ref", "summary.json")
    restored.setdefault("check_ref", "check.json")
    restored.setdefault("report_ref", "report.md")
    restored.setdefault("manifest_ref", "manifests/manifest.yaml")
    restored.setdefault("clean_trace_ref", "traces/clean.jsonl")
    restored.setdefault("perturbed_trace_ref", "traces/perturbed.jsonl")
    return restored


def _safe_bundle_path(root: Path, ref: Any, fallback: str) -> Path:
    candidate = Path(str(ref or fallback))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe bundle ref: {candidate}")
    return root / candidate


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text_file(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _write_jsonl_file(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(event, sort_keys=True) for event in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
