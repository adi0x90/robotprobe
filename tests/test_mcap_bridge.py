import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mcap.reader import make_reader

from robotprobe.exporters import export_mcap_bundle, import_mcap_bundle


class McapBridgeTests(unittest.TestCase):
    def test_exports_bundle_to_mcap_json_topics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle_dir = _write_bundle(root / "incident")
            out = root / "incident.mcap"

            result = export_mcap_bundle(bundle_dir, out)

            self.assertEqual(result["format"], "robotprobe.mcap.v0")
            self.assertTrue(out.exists())
            topics = _read_mcap_topics(out)

        self.assertIn("/robotprobe/bundle", topics)
        self.assertIn("/robotprobe/summary", topics)
        self.assertIn("/robotprobe/check", topics)
        self.assertIn("/robotprobe/report", topics)
        self.assertIn("/robotprobe/manifest", topics)
        self.assertIn("/robotprobe/traces/clean", topics)
        self.assertIn("/robotprobe/traces/perturbed", topics)
        self.assertEqual(topics["/robotprobe/summary"][0]["root_cause_hypothesis"]["layer"], "action_postprocess")
        self.assertEqual(topics["/robotprobe/traces/perturbed"][0]["step"], 0)
        self.assertTrue(topics["/robotprobe/traces/perturbed"][1]["keepout_violated"])

    def test_imports_robotprobe_mcap_topics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle_dir = _write_bundle(root / "incident")
            mcap_path = root / "incident.mcap"
            restored = root / "restored"
            export_mcap_bundle(bundle_dir, mcap_path)

            result = import_mcap_bundle(mcap_path, restored)

            self.assertEqual(result["format"], "robotprobe.bundle.v0")
            self.assertTrue((restored / "bundle.json").exists())
            self.assertTrue((restored / "summary.json").exists())
            self.assertTrue((restored / "check.json").exists())
            self.assertTrue((restored / "report.md").exists())
            self.assertTrue((restored / "manifests" / "fault.yaml").exists())
            self.assertTrue((restored / "traces" / "clean.jsonl").exists())
            self.assertTrue((restored / "traces" / "perturbed.jsonl").exists())
            summary = json.loads((restored / "summary.json").read_text(encoding="utf-8"))
            perturbed_events = [
                json.loads(line)
                for line in (restored / "traces" / "perturbed.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["root_cause_hypothesis"]["fault"], "stealth_velocity_bias_keepout")
        self.assertEqual(len(perturbed_events), 2)
        self.assertTrue(perturbed_events[1]["keepout_violated"])

    def test_cli_exports_and_imports_mcap_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle_dir = _write_bundle(root / "incident", with_causal_diff=True)
            mcap_path = root / "incident.mcap"
            restored = root / "restored"

            export_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "export",
                    "mcap",
                    "--bundle",
                    str(bundle_dir),
                    "--out",
                    str(mcap_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(export_proc.returncode, 0, export_proc.stderr)
            self.assertTrue(mcap_path.exists())

            import_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "import",
                    "mcap",
                    "--mcap",
                    str(mcap_path),
                    "--out",
                    str(restored),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(import_proc.returncode, 0, import_proc.stderr)
            self.assertTrue((restored / "bundle.json").exists())
            self.assertTrue((restored / "traces" / "perturbed.jsonl").exists())
            self.assertEqual(
                json.loads((restored / "analysis" / "causal_diff.json").read_text()),
                json.loads((bundle_dir / "analysis" / "causal_diff.json").read_text()),
            )

    def test_roundtrip_preserves_distinct_causal_diff_and_resolves_every_reference(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original = _write_bundle(root / "incident", with_causal_diff=True)
            mcap_path = root / "incident.mcap"
            restored = root / "restored"

            export_mcap_bundle(original, mcap_path)
            import_mcap_bundle(mcap_path, restored)

            bundle = json.loads((restored / "bundle.json").read_text())
            for field, ref in bundle.items():
                if field.endswith("_ref") and ref:
                    with self.subTest(field=field):
                        self.assertTrue((restored / ref).is_file())
            causal_diff_ref = bundle["causal_diff_ref"]
            self.assertEqual(
                json.loads((restored / causal_diff_ref).read_text()),
                json.loads((original / causal_diff_ref).read_text()),
            )
            self.assertNotEqual(
                json.loads((restored / causal_diff_ref).read_text()),
                json.loads((restored / bundle["summary_ref"]).read_text()),
            )
            self.assertIn("/robotprobe/causal_diff", _read_mcap_topics(mcap_path))


def _write_bundle(path: Path, *, with_causal_diff: bool = False) -> Path:
    path.mkdir(parents=True)
    (path / "traces").mkdir()
    (path / "manifests").mkdir()
    (path / "summary.json").write_text(
        json.dumps(
            {
                "root_cause_hypothesis": {
                    "layer": "action_postprocess",
                    "contract": "KeepoutZoneContract",
                    "fault": "stealth_velocity_bias_keepout",
                },
                "evidence": {"first_divergence_step": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "check.json").write_text(
        json.dumps({"passed": True, "diagnosis": {"layer": "action_postprocess"}}) + "\n",
        encoding="utf-8",
    )
    (path / "report.md").write_text("# RobotProbe Incident Report\n\nEvidence.\n", encoding="utf-8")
    (path / "manifests" / "fault.yaml").write_text(
        "id: ACTION.KEEP_OUT.BIAS.001\nrelease_classification: sim_only\n",
        encoding="utf-8",
    )
    (path / "traces" / "clean.jsonl").write_text(
        json.dumps({"step": 0, "action": [0.0], "keepout_violated": False}) + "\n"
        + json.dumps({"step": 1, "action": [0.1], "keepout_violated": False}) + "\n",
        encoding="utf-8",
    )
    (path / "traces" / "perturbed.jsonl").write_text(
        json.dumps({"step": 0, "action": [0.0], "keepout_violated": False}) + "\n"
        + json.dumps({"step": 1, "action": [0.4], "keepout_violated": True}) + "\n",
        encoding="utf-8",
    )
    bundle = {
        "summary_ref": "summary.json",
        "check_ref": "check.json",
        "report_ref": "report.md",
        "manifest_ref": "manifests/fault.yaml",
        "clean_trace_ref": "traces/clean.jsonl",
        "perturbed_trace_ref": "traces/perturbed.jsonl",
        "replay_level": "L2_SIM_REPLAY",
    }
    if with_causal_diff:
        (path / "analysis").mkdir()
        (path / "analysis" / "causal_diff.json").write_text(
            json.dumps({"format": "test.causal_diff", "layers": [{"name": "action_postprocess", "step": 1}]})
            + "\n",
            encoding="utf-8",
        )
        bundle["causal_diff_ref"] = "analysis/causal_diff.json"
    (path / "bundle.json").write_text(
        json.dumps(bundle) + "\n",
        encoding="utf-8",
    )
    return path


def _read_mcap_topics(path: Path) -> dict[str, list[dict]]:
    topics: dict[str, list[dict]] = {}
    with path.open("rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages():
            topics.setdefault(channel.topic, []).append(json.loads(message.data.decode("utf-8")))
    return topics


if __name__ == "__main__":
    unittest.main()
