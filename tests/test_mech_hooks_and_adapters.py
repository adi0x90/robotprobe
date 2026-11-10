import unittest

import numpy as np

from robotprobe.mech.activation_hooks import CapturePolicy, ForwardHookActivationCollector, ModuleSelector
from robotprobe.mech.adapters.base import WhiteBoxActivationAdapter


class MechHookAndAdapterTests(unittest.TestCase):
    def test_capture_policy_samples_every_n_steps_and_honors_max_steps(self):
        policy = CapturePolicy(every_n_steps=5, max_steps=12)

        self.assertTrue(policy.should_capture(0))
        self.assertTrue(policy.should_capture(5))
        self.assertTrue(policy.should_capture(10))
        self.assertFalse(policy.should_capture(11))
        self.assertFalse(policy.should_capture(15))

    def test_module_selector_matches_named_modules(self):
        model = FakeModel()
        selector = ModuleSelector(["backbone.layer18", "action_head"])

        selected = selector.select(model)

        self.assertEqual(sorted(selected), ["action_head", "backbone.layer18"])

    def test_forward_hook_collector_captures_selected_module_outputs(self):
        model = FakeModel()
        collector = ForwardHookActivationCollector(
            module_selector=ModuleSelector(["backbone.layer18"]),
            capture_policy=CapturePolicy(every_n_steps=1, max_steps=10),
        )

        with collector.attach(model):
            collector.set_step(3)
            model.forward(step=3)

        self.assertEqual(len(collector.captures), 1)
        capture = collector.captures[0]
        self.assertEqual(capture["step"], 3)
        self.assertEqual(capture["module"], "backbone.layer18")
        self.assertEqual(capture["activation"].tolist(), [3.0, 4.0, 5.0])

    def test_white_box_adapter_base_requires_capture_implementation(self):
        adapter = WhiteBoxActivationAdapter(target="fake")

        with self.assertRaises(NotImplementedError):
            adapter.capture_pair(None, None, None, None)


class FakeHandle:
    def __init__(self, module):
        self.module = module

    def remove(self):
        self.module.hooks.clear()


class FakeModule:
    def __init__(self, name):
        self.name = name
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle(self)

    def run(self, step):
        output = np.array([float(step), float(step + 1), float(step + 2)], dtype=np.float32)
        for hook in self.hooks:
            hook(self, (), output)
        return output


class FakeModel:
    def __init__(self):
        self.modules = {
            "backbone.layer18": FakeModule("backbone.layer18"),
            "action_head": FakeModule("action_head"),
        }

    def named_modules(self):
        return list(self.modules.items())

    def forward(self, *, step):
        for module in self.modules.values():
            module.run(step)


if __name__ == "__main__":
    unittest.main()
