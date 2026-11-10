import unittest

import numpy as np

try:
    import torch  # noqa: F401
    import nnsight  # noqa: F401
    HAVE_NNSIGHT = True
except ImportError:
    HAVE_NNSIGHT = False

from robotprobe.mech.adapters import NNsightActivationAdapter


@unittest.skipUnless(HAVE_NNSIGHT, "requires the optional nnsight extra (nnsight + torch)")
class NNsightAdapterTests(unittest.TestCase):
    def _model(self):
        import torch
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))

    def test_capture_pair_returns_real_activations_that_differ_for_differing_inputs(self):
        import torch
        adapter = NNsightActivationAdapter()
        clean = [torch.zeros(1, 4)]
        perturbed = [torch.ones(1, 4)]
        result = adapter.capture_pair(self._model(), clean, perturbed, {"layers": [{"name": "0"}, {"name": "2"}]})

        self.assertEqual(result["format"], "robotprobe.mech.nnsight.v0")
        self.assertEqual(result["modules"], ["0", "2"])
        self.assertEqual(len(result["clean"]), 2)  # two modules x one input
        self.assertEqual(len(result["perturbed"]), 2)

        clean_first = next(c for c in result["clean"] if c["module"] == "0")
        pert_first = next(c for c in result["perturbed"] if c["module"] == "0")
        self.assertEqual(clean_first["activation"].shape, (1, 8))
        # different inputs -> different captured activations (the diff signal MechProbe consumes)
        self.assertFalse(np.allclose(clean_first["activation"], pert_first["activation"]))

    def test_capture_indexes_steps(self):
        import torch
        adapter = NNsightActivationAdapter()
        caps = adapter.capture(self._model(), [torch.zeros(1, 4), torch.ones(1, 4)], ["2"])
        self.assertEqual([c["step"] for c in caps], [0, 1])
        self.assertEqual(caps[0]["activation"].shape, (1, 3))


    def test_capture_clones_before_downstream_inplace_mutation(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.ReLU(inplace=True))
        caps = NNsightActivationAdapter().capture(model, [torch.tensor([[0.25, -3.0]])], ["0"])
        np.testing.assert_array_equal(caps[0]["activation"], [[0.25, -3.0]])

    def test_bfloat16_is_converted_with_source_dtype_recorded(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Identity())
        caps = NNsightActivationAdapter().capture(model, [torch.ones(1, 2, dtype=torch.bfloat16)], ["0"])
        self.assertEqual(str(caps[0]["activation"].dtype), "float32")
        self.assertEqual(caps[0]["source_dtype"], "torch.bfloat16")

    def test_capture_pair_honors_sampling_and_dtype(self):
        import torch
        config = {"layers": [{"name": "0"}], "steps": {"mode": "every_n", "n": 2},
                  "storage": {"dtype": "float16", "max_bytes_per_run": 128}}
        result = NNsightActivationAdapter().capture_pair(
            torch.nn.Sequential(torch.nn.Identity()), [torch.ones(1, 2)] * 4,
            [torch.zeros(1, 2)] * 4, config,
        )
        self.assertEqual([c["step"] for c in result["clean"]], [0, 2])
        self.assertEqual(result["clean"][0]["activation"].dtype, np.dtype("float16"))

    def test_capture_pair_enforces_capture_memory_budget(self):
        import torch
        config = {"layers": [{"name": "0"}], "storage": {"max_bytes_per_run": 1}}
        with self.assertRaises(ValueError):
            NNsightActivationAdapter().capture_pair(
                torch.nn.Sequential(torch.nn.Identity()), [torch.ones(1, 2)], [torch.zeros(1, 2)], config
            )


if __name__ == "__main__":
    unittest.main()
