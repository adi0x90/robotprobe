import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mcap.writer import Writer

from robotprobe.integrations.isaac_lab_arena.trace_ingest import ingest_isaac_lab_arena_trace
from robotprobe.integrations.ros2.mcap_adapter import ingest_ros2_mcap
from robotprobe.integrations.vla_eval.episode_ingest import ingest_vla_eval_episode
from robotprobe.packs import list_pack
from robotprobe.plugins import load_plugin_entrypoint, load_plugin_manifest
from robotprobe.schema import validate_trace_file


class Ros2McapAdapterTests(unittest.TestCase):
    def test_ingests_json_ros2_mcap_topics_into_robotprobe_trace(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mcap_path = root / "ros2.mcap"
            trace_path = root / "trace.jsonl"
            _write_json_mcap(
                mcap_path,
                [
                    (
                        "/cmd_vel",
                        {
                            "linear": {"x": 0.4, "y": 0.0},
                            "angular": {"z": 0.1},
                            "command_authority": {"source": "nav_stack"},
                        },
                    ),
                    (
                        "/joint_states",
                        {
                            "name": ["joint1"],
                            "position": [0.2],
                        },
                    ),
                ],
            )

            result = ingest_ros2_mcap(
                mcap_path,
                trace_path,
                run_id="ros2_real_log",
                target="ros2",
                scenario="command_authority_replay",
            )

            events = _read_jsonl(trace_path)
            validation = validate_trace_file(trace_path)

        self.assertEqual(result["format"], "robotprobe.trace.v0")
        self.assertEqual(result["event_count"], 2)
        self.assertTrue(validation.valid, validation.to_dict())
        self.assertEqual(events[0]["event_type"], "postprocessed_action")
        self.assertEqual(events[0]["topic"], "/cmd_vel")
        self.assertEqual(events[0]["value"]["linear"]["x"], 0.4)
        self.assertEqual(events[0]["command_authority"]["source"], "nav_stack")
        self.assertEqual(events[1]["event_type"], "robot_state")
        self.assertEqual(events[1]["topic"], "/joint_states")

    def test_cli_ingests_ros2_mcap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mcap_path = root / "ros2.mcap"
            trace_path = root / "trace.jsonl"
            _write_json_mcap(mcap_path, [("/cmd_vel", {"linear": {"x": 0.1}})])

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "ingest",
                    "ros2-mcap",
                    "--input",
                    str(mcap_path),
                    "--out",
                    str(trace_path),
                    "--run-id",
                    "ros2_cli",
                    "--target",
                    "ros2",
                    "--scenario",
                    "cli_smoke",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            result = json.loads(proc.stdout) if proc.stdout else {}
            trace_exists = trace_path.exists()

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["event_count"], 1)
        self.assertTrue(trace_exists)


class VlaEvalEpisodeAdapterTests(unittest.TestCase):
    def test_ingests_vla_eval_episode_json_into_robotprobe_trace(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            episode_path = root / "episode.json"
            trace_path = root / "trace.jsonl"
            episode_path.write_text(
                json.dumps(
                    {
                        "episode_id": "libero_spatial_0001",
                        "task": {"language_instruction": "pick up the black bowl"},
                        "steps": [
                            {
                                "observation": {"state": [0.0, 0.1]},
                                "raw_action": [0.1, 0.0, 0.2],
                                "postprocessed_action": [0.1, 0.0, 0.2],
                            },
                            {
                                "observation": {"state": [0.1, 0.2]},
                                "action": [0.2, 0.0, 0.3],
                            },
                        ],
                        "success": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = ingest_vla_eval_episode(
                episode_path,
                trace_path,
                run_id="vla_eval_real_episode",
                target="vla-eval",
                scenario="libero_spatial",
            )
            events = _read_jsonl(trace_path)
            validation = validate_trace_file(trace_path)

        self.assertEqual(result["format"], "robotprobe.trace.v0")
        self.assertEqual(result["step_count"], 2)
        self.assertTrue(validation.valid, validation.to_dict())
        self.assertEqual(events[0]["event_type"], "instruction")
        self.assertEqual(events[0]["value"], "pick up the black bowl")
        self.assertTrue(any(event["event_type"] == "raw_action" for event in events))
        self.assertTrue(any(event["event_type"] == "postprocessed_action" for event in events))
        self.assertEqual(events[-1]["event_type"], "outcome")
        self.assertTrue(events[-1]["success"])

    def test_cli_ingests_vla_eval_episode_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            episode_path = root / "episode.json"
            trace_path = root / "trace.jsonl"
            episode_path.write_text(
                json.dumps(
                    {
                        "task_description": "stack the blocks",
                        "steps": [{"action": [0.1]}],
                        "success": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "ingest",
                    "vla-eval",
                    "--input",
                    str(episode_path),
                    "--out",
                    str(trace_path),
                    "--run-id",
                    "vla_eval_cli",
                    "--target",
                    "vla-eval",
                    "--scenario",
                    "cli_smoke",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout) if proc.stdout else {}
            trace_exists = trace_path.exists()

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["step_count"], 1)
        self.assertTrue(trace_exists)


class IsaacLabArenaTraceAdapterTests(unittest.TestCase):
    def test_ingests_isaac_lab_arena_jsonl_trace(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "isaac_trace.jsonl"
            trace_path = root / "trace.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "step": 0,
                                "joint_positions": [0.0, 0.2],
                                "action": [0.1, 0.0],
                                "reward": 0.0,
                            }
                        ),
                        json.dumps(
                            {
                                "step": 1,
                                "state": {"eef_pos": [0.2, 0.3, 0.4]},
                                "action": [0.2, 0.1],
                                "success": True,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = ingest_isaac_lab_arena_trace(
                input_path,
                trace_path,
                run_id="isaac_lab_arena_real_trace",
                target="isaac-lab-arena",
                scenario="lift_cube",
            )
            events = _read_jsonl(trace_path)
            validation = validate_trace_file(trace_path)

        self.assertEqual(result["format"], "robotprobe.trace.v0")
        self.assertEqual(result["step_count"], 2)
        self.assertTrue(validation.valid, validation.to_dict())
        self.assertEqual(events[0]["event_type"], "sim_state")
        self.assertEqual(events[1]["event_type"], "postprocessed_action")
        self.assertEqual(events[-1]["event_type"], "outcome")
        self.assertTrue(events[-1]["success"])

    def test_cli_ingests_isaac_lab_arena_trace(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "isaac_trace.json"
            trace_path = root / "trace.jsonl"
            input_path.write_text(
                json.dumps({"steps": [{"step": 0, "state": {"x": 1}, "action": [0.1]}], "success": True}) + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "ingest",
                    "isaac-lab-arena",
                    "--input",
                    str(input_path),
                    "--out",
                    str(trace_path),
                    "--run-id",
                    "isaac_cli",
                    "--target",
                    "isaac-lab-arena",
                    "--scenario",
                    "cli_smoke",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout) if proc.stdout else {}
            trace_exists = trace_path.exists()

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["step_count"], 1)
        self.assertTrue(trace_exists)


class ExecutableAdapterPackTests(unittest.TestCase):
    def test_example_adapter_packs_expose_executable_entrypoints(self):
        root = Path(__file__).resolve().parents[1] / "examples" / "packs"
        expected = {
            "ros2-command-authority": "robotprobe.integrations.ros2.mcap_adapter:ingest_ros2_mcap",
            "vla-eval-adapter": "robotprobe.integrations.vla_eval.episode_ingest:ingest_vla_eval_episode",
            "isaac-lab-arena-adapter": "robotprobe.integrations.isaac_lab_arena.trace_ingest:ingest_isaac_lab_arena_trace",
        }

        for pack_name, entrypoint in expected.items():
            pack = root / pack_name
            listing = list_pack(pack)
            manifest = load_plugin_manifest(pack / "manifest.yaml")
            loaded = load_plugin_entrypoint(manifest.plugins[0])

            self.assertEqual(listing["plugins"][0]["entrypoint"], entrypoint)
            self.assertTrue(callable(loaded))


def _write_json_mcap(path: Path, messages: list[tuple[str, dict]]) -> None:
    with path.open("wb") as f:
        writer = Writer(f)
        writer.start()
        channels: dict[str, int] = {}
        for index, (topic, payload) in enumerate(messages):
            if topic not in channels:
                channels[topic] = writer.register_channel(topic=topic, message_encoding="json", schema_id=0)
            timestamp_ns = index * 1_000_000_000
            writer.add_message(
                channel_id=channels[topic],
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=json.dumps(payload).encode("utf-8"),
            )
        writer.finish()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
