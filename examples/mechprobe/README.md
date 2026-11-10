# MechProbe Example

This directory contains a synthetic capture config and a manual-direction dictionary. Run the
[complete CLI walkthrough](../../docs/mechprobe.md#commands) from the repository root to create
inputs, a summary, per-run activation-event traces, diffs, and JSON/Markdown reports in a temporary
directory.

`capture_config.yaml` selects every available fixture step and stores float16 arrays. The collector
honors the sampling/dtype settings and the byte limit for serialized activation files. These
activations are invented: all derived outputs remain Level 0 with `synthetic: true` and
`evidence_eligible: false`. They must fail an evidence gate requiring eligible real captures.

The dictionary labels are candidates, not validated feature meanings. `faithfulness` ranks
activation deltas and scales them by an external consequence metric; it does not test prediction or
run an intervention.

For actual CPU-model capture and persistence, use the [NNsight Python
example](../../docs/mechprobe.md#nnsight-bridge). Live VLA capture still requires a model-specific adapter.
