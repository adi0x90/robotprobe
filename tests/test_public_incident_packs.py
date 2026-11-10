import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe.packs import list_pack, validate_pack
from robotprobe.plugins import load_plugin_entrypoint, load_plugin_manifest
from robotprobe.schema import validate_trace_file


class ViFailbackIncidentPackTests(unittest.TestCase):
    def test_converts_vifailback_annotation_json_to_valid_incident_traces(self):
        from robotprobe.integrations.incident_datasets.vifailback import convert_vifailback_annotations

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            annotation_path = root / "cut_sausage_annotations.json"
            out_dir = root / "out"
            annotation_path.write_text(
                json.dumps(
                    [
                        {
                            "video": "annotated_data/cut_sausage/videos/episode_23_cam_high.mp4",
                            "images": [
                                "annotated_data/cut_sausage/frames/episode_23_cam_high/0.jpg",
                                "annotated_data/cut_sausage/frames/episode_23_cam_high/250_avoid_keyframe.jpg",
                                "annotated_data/cut_sausage/frames/episode_23_cam_high/302_correct_keyframe.jpg",
                            ],
                            "task": "Use the right arm to pick up the knife.",
                            "failure_detection": "yes",
                            "subtasks": ["1. Move to the knife holder.", "2. Grasp the knife."],
                            "failure_subtask": "1",
                            "failure_type": "gripper 6d-pose",
                            "keyframe": [
                                {
                                    "avoid_keyframe": "annotated_data/cut_sausage/frames/episode_23_cam_high/250_avoid_keyframe.jpg",
                                    "correct_keyframe": "annotated_data/cut_sausage/frames/episode_23_cam_high/302_correct_keyframe.jpg",
                                }
                            ],
                            "avoidance": [{"low_level": "Move the right gripper downward slightly."}],
                            "correction": [{"low_level": "Move the right gripper right and down."}],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = convert_vifailback_annotations(annotation_path, out_dir, limit=1)
            trace_path = Path(result["traces"][0]["trace"])
            events = _read_jsonl(trace_path)
            validation = validate_trace_file(trace_path)

        self.assertEqual(result["dataset"], "ViFailback")
        self.assertEqual(result["trace_count"], 1)
        self.assertTrue(validation.valid, validation.to_dict())
        self.assertEqual(events[0]["event_type"], "instruction")
        self.assertTrue(any(event["event_type"] == "observation" for event in events))
        contract_events = [event for event in events if event["event_type"] == "contract_check"]
        self.assertEqual(contract_events[0]["contract"], "DatasetFailureAnnotation")
        self.assertEqual(contract_events[0]["check"], "gripper_6d_pose")
        self.assertEqual(events[-1]["event_type"], "outcome")
        self.assertFalse(events[-1]["success"])
        self.assertEqual(events[-1]["failure_type"], "gripper 6d-pose")


class RoboFACIncidentPackTests(unittest.TestCase):
    def test_converts_robofac_metadata_without_h5_to_valid_trace(self):
        from robotprobe.integrations.incident_datasets.robofac import convert_robofac_episodes

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "view0.json"
            out_dir = root / "out"
            metadata_path.write_text(
                json.dumps(
                    {
                        "env_info": {"env_id": "LiftPegUpright-box", "max_episode_steps": 50},
                        "source_type": "motionplanning",
                        "source_desc": "official motion planning solution",
                        "episodes": [
                            {
                                "episode_id": 0,
                                "episode_seed": 0,
                                "control_mode": "pd_joint_pos",
                                "elapsed_steps": 171,
                                "unique_id": "episode-zero",
                                "success": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = convert_robofac_episodes(metadata_path, out_dir, limit=1)
            trace_path = Path(result["traces"][0]["trace"])
            events = _read_jsonl(trace_path)
            validation = validate_trace_file(trace_path)

        self.assertEqual(result["dataset"], "RoboFAC")
        self.assertEqual(result["trace_count"], 1)
        self.assertTrue(validation.valid, validation.to_dict())
        self.assertEqual(events[0]["event_type"], "instruction")
        self.assertTrue(any(event["event_type"] == "contract_check" for event in events))
        self.assertEqual(events[-1]["event_type"], "outcome")
        self.assertFalse(events[-1]["success"])
        self.assertEqual(events[-1]["env_id"], "LiftPegUpright-box")

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py optional dependency is not installed")
    def test_converts_robofac_h5_actions_into_action_events(self):
        import h5py
        import numpy as np

        from robotprobe.integrations.incident_datasets.robofac import convert_robofac_episodes

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "view0.json"
            h5_path = root / "view0.h5"
            out_dir = root / "out"
            metadata_path.write_text(
                json.dumps(
                    {
                        "env_info": {"env_id": "LiftPegUpright-box"},
                        "episodes": [
                            {
                                "episode_id": 0,
                                "elapsed_steps": 3,
                                "unique_id": "episode-zero",
                                "success": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with h5py.File(h5_path, "w") as f:
                traj = f.create_group("traj_0")
                traj.create_dataset("actions", data=np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype="float32"))
                traj.create_dataset("rewards", data=np.array([0.0, 0.0, 0.0], dtype="float32"))
                traj.create_dataset("success", data=np.array([False, False, False]))
                articulations = traj.create_group("env_states").create_group("articulations")
                articulations.create_dataset("panda", data=np.ones((4, 4), dtype="float32"))

            result = convert_robofac_episodes(metadata_path, out_dir, h5_path=h5_path, limit=1, max_steps=2)
            trace_path = Path(result["traces"][0]["trace"])
            events = _read_jsonl(trace_path)
            validation = validate_trace_file(trace_path)

        self.assertTrue(validation.valid, validation.to_dict())
        self.assertEqual(result["traces"][0]["h5_used"], True)
        self.assertTrue(any(event["event_type"] == "raw_action" for event in events))
        self.assertTrue(any(event["event_type"] == "postprocessed_action" for event in events))
        self.assertTrue(any(event["event_type"] == "robot_state" for event in events))


class PublicIncidentPackCliTests(unittest.TestCase):
    def test_cli_converts_vifailback_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            annotation_path = root / "annotations.json"
            out_dir = root / "out"
            annotation_path.write_text(
                json.dumps(
                    [
                        {
                            "video": "annotated_data/task/videos/episode_1_cam_high.mp4",
                            "images": ["annotated_data/task/frames/episode_1_cam_high/0.jpg"],
                            "task": "Pick the cube.",
                            "failure_detection": "no",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robotprobe",
                    "incident-pack",
                    "convert",
                    "vifailback",
                    "--annotation",
                    str(annotation_path),
                    "--out",
                    str(out_dir),
                    "--limit",
                    "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = json.loads(proc.stdout) if proc.stdout else {}

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(result["trace_count"], 1)


class PublicIncidentPackManifestTests(unittest.TestCase):
    def test_example_incident_packs_validate_and_load_entrypoints(self):
        root = Path(__file__).resolve().parents[1] / "examples" / "incident_packs"
        for pack_name in ("vifailback", "robofac"):
            pack = root / pack_name
            report = validate_pack(pack)
            listing = list_pack(pack)
            manifest = load_plugin_manifest(pack / "manifest.yaml")
            loaded = load_plugin_entrypoint(manifest.plugins[0])

            self.assertTrue(report.valid, report.to_dict())
            self.assertEqual(listing["plugins"][0]["type"], "adapter")
            self.assertTrue(callable(loaded))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
