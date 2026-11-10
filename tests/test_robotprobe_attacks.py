import unittest

import numpy as np

from robotprobe.attacks import (
    ActionFaultAttack,
    InstructionOverrideAttack,
    Observation,
    StaleObservationReplayAttack,
)


class RobotProbeAttackTests(unittest.TestCase):
    def test_instruction_override_does_not_mutate_observation(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        obs = Observation(image=image, instruction="pick red")

        out = InstructionOverrideAttack(payload="pick blue").perturb(obs, step=0)

        self.assertEqual(out.instruction, "pick blue")
        self.assertEqual(obs.instruction, "pick red")
        self.assertIs(out.image, image)

    def test_action_hold_previous_freezes_active_dimension(self):
        attack = ActionFaultAttack(
            mode="hold_previous_dims",
            dims=(2,),
            start=1,
            end=3,
        )

        attack.perturb_action(np.array([[0.1, 0.2, 0.7]], dtype=np.float32), step=0)
        active = attack.perturb_action(np.array([[0.1, 0.2, 0.0]], dtype=np.float32), step=1)

        np.testing.assert_array_equal(active, np.array([[0.1, 0.2, 0.7]], dtype=np.float32))

    def test_stale_observation_replays_last_pre_window_image(self):
        image0 = np.zeros((2, 2, 3), dtype=np.uint8)
        image1 = np.ones((2, 2, 3), dtype=np.uint8)
        image2 = np.full((2, 2, 3), 2, dtype=np.uint8)
        attack = StaleObservationReplayAttack(start=2, end=3)

        attack.perturb(Observation(image=image0, instruction="task"), step=0)
        attack.perturb(Observation(image=image1, instruction="task"), step=1)
        out = attack.perturb(Observation(image=image2, instruction="task"), step=2)

        np.testing.assert_array_equal(out.image, image1)
        self.assertFalse(np.shares_memory(out.image, image1))


if __name__ == "__main__":
    unittest.main()
