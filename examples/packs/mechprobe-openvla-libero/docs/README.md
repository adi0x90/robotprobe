# MechProbe OpenVLA LIBERO Pack

This pack is scaffolding for OpenVLA/LIBERO white-box activation capture. It is not a working
adapter.

It deliberately does not import OpenVLA or vla-eval code. Its value is the pack boundary:
model-specific activation hooks belong here, while RobotProbe core keeps only generic activation
events, activation stores, diffs, and reports.

What is actually here:

- a capture-config shape that `robotprobe pack validate` accepts;
- `adapters/openvla_activation_adapter.py`, which is importable without OpenVLA, Torch, or vla-eval
  and returns `{"status": "not_implemented"}`;
- release classification `sim_only`, with hardware execution disallowed;
- capability metadata with every `supports_*` flag false and `safe_default_mode: offline_only`;
- no SAE labels and no causal feature claims.

What is not here: any activation capture from a real model. Running this pack produces no
activations.

To turn it into a working adapter:

1. Implement `adapters/openvla_activation_adapter.py` against an instrumentable Python model
   instance, attaching forward hooks.
2. Use `robotprobe.mech.collect.persist_capture_pair` to write one `model_activation` event file
   per run and an `activation_index.json`, supplying real run/incident metadata and
   `synthetic=False`. See [the NNsight bridge example](../../../../docs/mechprobe.md#nnsight-bridge).
3. Validate against a paired LIBERO instruction-swap run and a post-policy negative control.
4. Enable only the capability flags demonstrated by the implemented adapter.
