import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robotprobe import ReplayLevel, TraceRecorder
from robotprobe.campaign import CampaignSpec, FaultSpec, run_campaign, run_campaign_file


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "campaigns" / "campaign.json"


def _baseline(path):
    with TraceRecorder(
        path, run_id="rp_base", target="arm", scenario="clean",
        seed=0, replay_level=ReplayLevel.TRACE_PLAYBACK,
    ) as r:
        for step in range(6):
            a = [0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.4]
            r.event(step, "raw_action", "policy_forward", value=a)
            r.event(step, "postprocessed_action", "action_postprocess", value=a)
        r.event(20, "outcome", "outcome", success=True, steps=20)


def _envelope_contract(path):
    path.write_text(
        "id: env\ntype: temporal_contract\nwindow:\n  start_step: 0\n  end_step: 200\n"
        "must_never:\n  field: action_max_abs\n  gt: 0.9\n"
        "violation:\n  contract: ActionEnvelopeContract\n  check: env\n  severity: 0.7\n",
        encoding="utf-8",
    )


class CampaignTests(unittest.TestCase):
    def test_saturating_fault_violates_envelope_benign_does_not(self):
        with tempfile.TemporaryDirectory() as raw_td:
            td = Path(raw_td)
            _baseline(td / "baseline.jsonl")
            _envelope_contract(td / "env.yaml")
            spec = CampaignSpec(
                id="t",
                baseline="baseline.jsonl",
                contracts=["env.yaml"],
                metrics=["trajectory_deviation"],
                faults=[
                    FaultSpec(name="benign", channel="action", mode="scale_dims", dims=(0, 1, 2), scale=0.5),
                    FaultSpec(name="saturate", channel="action", mode="set_dims", dims=(0,), value=5.0, start=2, end=6),
                ],
            )
            report = run_campaign(spec, base_dir=td, out_dir=td / "out")

            self.assertEqual(report["n_faults"], 2)
            self.assertEqual(report["n_faults_violating_contracts"], 1)
            self.assertEqual(report["violating_faults"], ["saturate"])
            by_name = {f["name"]: f for f in report["faults"]}
            self.assertFalse(by_name["benign"]["violated_any_contract"])
            self.assertTrue(by_name["saturate"]["violated_any_contract"])
            self.assertEqual(by_name["saturate"]["diagnosis"]["layer"], "action_postprocess")
            self.assertEqual(by_name["saturate"]["diagnosis"]["first_divergence_step"], 2)
            # bundle artifacts written (nemesis manifest + history trace + report)
            self.assertTrue((td / "out" / "campaign_report.json").exists())
            self.assertTrue((td / "out" / "campaign_report.md").exists())
            self.assertTrue((td / "out" / "traces" / "saturate.jsonl").exists())
            self.assertTrue((td / "out" / "manifests" / "saturate.yaml").exists())

    def test_instruction_channel_fault_is_diagnosed_as_instruction(self):
        with tempfile.TemporaryDirectory() as raw_td:
            td = Path(raw_td)
            with TraceRecorder(
                td / "baseline.jsonl", run_id="rp_b2", target="arm", scenario="clean",
                seed=0, replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as r:
                r.event(0, "instruction", "instruction", before="pick red", after="pick red", value="pick red")
                r.event(10, "outcome", "outcome", success=True, steps=10)
            spec = CampaignSpec(
                id="t2",
                baseline="baseline.jsonl",
                faults=[FaultSpec(name="swap", channel="instruction", payload="pick blue", start=0, end=5)],
            )
            report = run_campaign(spec, base_dir=td, out_dir=td / "out")
            self.assertEqual(report["faults"][0]["diagnosis"]["layer"], "instruction")

    def test_example_campaign_runs_via_cli_and_fails_on_violation(self):
        with tempfile.TemporaryDirectory() as raw_td:
            proc = subprocess.run(
                [sys.executable, "-m", "robotprobe", "campaign", "run",
                 "--campaign", str(EXAMPLE), "--out", str(Path(raw_td) / "out"), "--fail-on-violation"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)  # example contains a violating fault
            report = json.loads(proc.stdout)
            self.assertEqual(report["campaign_id"], "action-envelope-sweep")
            self.assertIn("saturate_x", report["violating_faults"])

    def test_example_campaign_file_loads(self):
        with tempfile.TemporaryDirectory() as raw_td:
            report = run_campaign_file(EXAMPLE, out_dir=Path(raw_td) / "out")
            self.assertEqual(report["n_faults"], 2)


if __name__ == "__main__":
    unittest.main()
