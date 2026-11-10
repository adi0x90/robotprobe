import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotprobe import ReplayLevel, TraceRecorder
from robotprobe.attach import PolicyProbe, Watchpoint
from robotprobe.attacks import ActionFaultAttack, InstructionOverrideAttack, Observation
from robotprobe.schema import validate_trace_file


def _obs(instruction="pick red"):
    return Observation(image=np.zeros((4, 4, 3), dtype=np.uint8), instruction=instruction)


def _fixed_policy(_obs):
    return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8]


class PolicyProbeTests(unittest.TestCase):
    def test_output_fault_injection_and_watchpoint_fire_live(self):
        hits = []
        wp = Watchpoint(
            name="gripper_frozen",
            event_type="postprocessed_action",
            predicate=lambda e: e["value"][6] == 0.0,
            on_hit=lambda e: hits.append(e["step"]),
        )
        freeze = ActionFaultAttack(mode="set_dims", dims=(6,), value=0.0, start=2)
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "live.jsonl"
            with TraceRecorder(
                trace, run_id="rp_live", target="bot", scenario="live_freeze",
                seed=0, replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as rec:
                probe = PolicyProbe(_fixed_policy, rec, output_attack=freeze, watchpoints=[wp])
                actions = probe.run([_obs() for _ in range(4)])

            # gripper frozen from step 2 onward -> watchpoint hits at 2 and 3
            self.assertEqual(hits, [2, 3])
            self.assertEqual([h.step for h in probe.hits], [2, 3])
            self.assertEqual(actions[0][6], 0.8)   # before window: untouched
            self.assertEqual(actions[2][6], 0.0)   # in window: frozen
            report = validate_trace_file(trace)
            self.assertTrue(report.valid, report.issues)

    def test_input_instruction_override_is_streamed(self):
        swap = InstructionOverrideAttack(payload="pick blue")
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "live.jsonl"
            with TraceRecorder(
                trace, run_id="rp_live2", target="bot", scenario="live_swap",
                seed=0, replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as rec:
                probe = PolicyProbe(_fixed_policy, rec, input_attack=swap)
                probe.step(0, _obs(instruction="pick red"))
            events = [__import__("json").loads(line) for line in trace.read_text().splitlines()]
            instr = [e for e in events if e["event_type"] == "instruction"][0]
            self.assertEqual(instr["value"], "pick blue")

    def test_clean_run_streams_raw_equals_postprocessed(self):
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "clean.jsonl"
            with TraceRecorder(
                trace, run_id="rp_live3", target="bot", scenario="clean",
                seed=0, replay_level=ReplayLevel.TRACE_PLAYBACK,
            ) as rec:
                probe = PolicyProbe(_fixed_policy, rec)
                probe.run([_obs() for _ in range(3)])
                self.assertEqual(probe.hits, [])
            self.assertTrue(validate_trace_file(trace).valid)


if __name__ == "__main__":
    unittest.main()
