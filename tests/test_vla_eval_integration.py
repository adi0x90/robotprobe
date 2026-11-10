import unittest

import numpy as np

from robotprobe.integrations.vla_eval.robotprobe_adapter import (
    perturb_vla_eval_action,
    perturb_vla_eval_observation,
)
from robotprobe.attacks import ActionFaultAttack, InstructionOverrideAttack


class VlaEvalIntegrationTests(unittest.TestCase):
    def test_perturb_observation_applies_instruction_attack_without_mutating_input(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        obs = {
            "images": {"agentview": image},
            "task_description": "pick up the red bowl",
            "state": np.zeros(7),
        }
        attack = InstructionOverrideAttack(name="instruction_swap", payload="pick up the blue bowl")

        out = perturb_vla_eval_observation(obs, attack, step=0, camera="agentview")

        self.assertEqual(out["task_description"], "pick up the blue bowl")
        self.assertEqual(obs["task_description"], "pick up the red bowl")
        self.assertIs(out["images"]["agentview"], image)

    def test_perturb_action_applies_action_fault_to_actions_field(self):
        payload = {"actions": np.array([[0.1, 0.2, 0.7]], dtype=np.float32)}
        attack = ActionFaultAttack(
            name="gripper_freeze",
            mode="hold_previous_dims",
            dims=(2,),
            start=1,
            end=3,
        )
        perturb_vla_eval_action(payload, attack, step=0)

        out = perturb_vla_eval_action(
            {"actions": np.array([[0.1, 0.2, 0.0]], dtype=np.float32)},
            attack,
            step=1,
        )

        np.testing.assert_array_equal(out["actions"], np.array([[0.1, 0.2, 0.7]], dtype=np.float32))
        np.testing.assert_array_equal(payload["actions"], np.array([[0.1, 0.2, 0.7]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
