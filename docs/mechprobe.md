# MechProbe

The optional white-box layer: activation capture, storage, and clean-versus-perturbed comparison.

## Overview

MechProbe is RobotProbe's optional white-box mechanistic-forensics layer.

RobotProbe records what changed in the external robot stack. MechProbe compares selected internal
activations against the same clean-vs-perturbed incident. It provides generic capture primitives and
a CPU-tested NNsight adapter; a live VLA still needs a model-specific adapter and runtime integration.

The goal is not SAE visualization. The goal is incident-grounded mechanistic evidence: did an
internal representation change before the action changed, did it fail to change when an external
contract changed, and can any candidate feature survive prediction, intervention, and
counterfactual tests?

MechProbe is disabled by default. RobotProbe remains useful without it.

### What Exists Now

The current MechProbe slice includes:

- `model_activation` trace events with file references, shape, dtype, model metadata, and capture
  metadata;
- an activation store that writes step/module activation files under a RobotProbe incident or
  mech directory;
- `activation_index.json` linking trace steps, modules, run IDs, and activation files;
- a file-backed `ActivationCacheView` for querying modules, run labels, timesteps, paired
  activations, and cache summaries without importing live model runtimes;
- MechProbe adapter capability metadata for pack manifests, so claim cards can say whether a pack
  supports capture, temporal patching, action-head access, uncertainty, replay, or only offline
  analysis;
- activation diff over clean-vs-perturbed paired captures;
- offline consequence-faithfulness scoring (`consequence_aopc`) that ranks activation deltas and
  scales them by a consequence metric; this does not measure predictive or causal importance;
- synthetic CPU-only activation collection for local pipeline tests and demos;
- generic forward-hook collection primitives and a Python persistence bridge for adapter captures;
- manual-direction feature diff with candidate labels and evidence levels;
- a RobotSec mech-forensics challenge checker for evidence-level gates;
- negative-control handling for post-policy faults where model internals should not be blamed;
- JSON and Markdown MechProbe reports with evidence levels and claim boundaries;
- CLI commands for config validation, cache inspection, activation diffing, offline
  consequence-faithfulness, and report rendering.

It does not include live OpenVLA/Pi0.5/GR00T activation adapters yet. Those should ship as
pack-specific white-box adapters, not as assumptions inside RobotProbe core.

The first pack boundary is `examples/packs/mechprobe-openvla-libero`. It validates the
OpenVLA/LIBERO capture contract and claim boundary, but intentionally does not import live model
code yet.

### Commands

The following sequence runs from the repository root after `uv sync`. It uses the bundled
clean/instruction-swap trace pair and writes into a fresh temporary directory. Those external traces
are recorded inputs; `--target synthetic` invents the internal activations solely to exercise the
pipeline. The resulting report is Level 0 and is not evidence about the recorded model.

```bash
MECH_OUT=$(mktemp -d)
MECH_TRACES=robotsec_range/fixtures/robotprobe-v1-libero-realtrace
uv run robotprobe mech validate --capture-config examples/mechprobe/capture_config.yaml
uv run robotprobe summarize \
  --clean "$MECH_TRACES/clean.jsonl" \
  --perturbed "$MECH_TRACES/instruction_swap.jsonl" > "$MECH_OUT/summary.json"
uv run robotprobe mech collect \
  --target synthetic \
  --clean "$MECH_TRACES/clean.jsonl" \
  --perturbed "$MECH_TRACES/instruction_swap.jsonl" \
  --capture-config examples/mechprobe/capture_config.yaml \
  --out "$MECH_OUT/mech"
uv run robotprobe validate-trace "$MECH_OUT/mech/model_activation_events.clean.jsonl"
uv run robotprobe validate-trace "$MECH_OUT/mech/model_activation_events.perturbed.jsonl"
uv run robotprobe mech cache-info --mech-dir "$MECH_OUT/mech"
uv run robotprobe mech diff-activations \
  --mech-dir "$MECH_OUT/mech" --out "$MECH_OUT/mech/activation_diff.json"
uv run robotprobe mech diff-features \
  --mech-dir "$MECH_OUT/mech" \
  --dictionary examples/mechprobe/dictionary.json \
  --out "$MECH_OUT/mech/feature_diff.json"
uv run robotprobe mech faithfulness \
  --mech-dir "$MECH_OUT/mech" --summary "$MECH_OUT/summary.json" \
  --target-metric raw_action_delta --max-k 8 \
  --out "$MECH_OUT/mech/consequence_aopc.json"
uv run robotprobe mech report \
  --summary "$MECH_OUT/summary.json" \
  --activation-diff "$MECH_OUT/mech/activation_diff.json" \
  --feature-diff "$MECH_OUT/mech/feature_diff.json" \
  --out "$MECH_OUT/mech_report.json"
uv run robotprobe mech report \
  --summary "$MECH_OUT/summary.json" \
  --activation-diff "$MECH_OUT/mech/activation_diff.json" \
  --feature-diff "$MECH_OUT/mech/feature_diff.json" \
  --out "$MECH_OUT/mech_report.md"
```

Each activation-event file contains one run ID and validates as a RobotProbe trace. `collect`
returns their paths in `activation_events.clean` and `activation_events.perturbed`. JSON and Markdown
reports include provenance, comparison coverage, and candidate feature results.

`cache-info` inspects index metadata without validating tensor payloads or comparisons. It returns
`evidence_eligible: null`, `evidence_status: "not_assessed"`, and an `eligibility_reason`;
complete indexed coverage does not establish eligibility. Run `diff-activations` to validate the
stored tensors and assess eligibility.

Check the JSON report with:

```bash
uv run robotsec-range check-mech \
  --challenge robotprobe-mech-v0-instruction-internal-leadtime \
  --report "$MECH_OUT/mech_report.json"
```

This check **must fail with exit code 1** for the synthetic report. The built-in challenge requires
eligible real captures and an observed internal lead of at least four steps at Level 1. A fixture
cannot qualify. This is a temporal-correlation gate, not a validated prediction test. Custom
challenges requiring Level 2 cannot pass on current offline outputs. A successful command execution
is different from passing an evidence gate.

`faithfulness` writes `mode: offline_proxy`, `intervention_executed: false`, and
`promotion_allowed: false`. Its score is an area over ranked activation dimensions, scaled by the
selected consequence metric. It is not normalized across different layer widths or `--max-k`
values. A positive score does not establish that a dimension predicts an action or causes a failure.

The CLI `collect` targets are `synthetic` and its `toy` alias. Other targets, including `nnsight`,
return `not_implemented` with exit code 2. Use the [NNsight Python capture and persistence
example](mechprobe.md#nnsight-bridge) for a real CPU model; live VLA model loading is not exposed by the CLI.

### Capture configuration and adapter metadata

The capture config declares layer names, storage format and dtype, a positive byte budget per run,
and an optional `steps: {mode: every_n, n: ...}` sampling policy. Supported storage dtypes are
`float16`, `float32`, and `float64`. An optional `steps.max_steps` is an inclusive final step.
The persistence bridge requires a fresh or empty output directory. It applies these settings and
counts serialized activation-file bytes against the budget; JSON indexes and event files are not
part of that tensor budget. If sampling leaves no
captures, collection returns `no_captures` rather than successful evidence.

The generic forward-hook collector takes the episode step from the runtime or tracks model-forward
calls; activation values are never timestamps. Captures are snapshots. `bfloat16` tensors are
converted for NumPy storage while retaining their source dtype metadata.

Adapter capability flags describe implemented behavior. The OpenVLA example pack is scaffolding and
advertises all `supports_*` flags as false, with `safe_default_mode: offline_only`. Its import does
not load a model runtime. Only change those flags after implementing and validating the capability.
`robotprobe pack validate`, `robotprobe pack list`, and `robotprobe pack claims` surface this metadata
without importing the pack runtime code.

### Activation Event

Activation tensors are not embedded inline in JSONL traces. The trace records references:

```json
{
  "event_type": "model_activation",
  "layer": "model_activation",
  "run_id": "clean_001",
  "target": "pi05-libero",
  "scenario": "instruction_swap",
  "replay_level": "L0_TRACE_PLAYBACK",
  "step": 10,
  "module": "policy.backbone.layers.18",
  "module_kind": "residual_stream",
  "activation_ref": "activations/clean/policy.backbone.layers.18/step_000010.safetensors",
  "activation_shape": [1, 128, 4096],
  "activation_dtype": "float16",
  "model_ref": {
    "model_name": "pi0.5",
    "checkpoint_hash": "sha256:...",
    "runtime": "torch"
  },
  "capture": {
    "capture_mode": "forward_hook",
    "hook_version": "mechprobe.activation_hook.v0",
    "detached": true,
    "requires_grad": false
  }
}
```

### Evidence Levels

MechProbe feature or activation claims use this evidence ladder:

| Level | Meaning | Promotion |
| --- | --- | --- |
| 0 | Unvalidated label, fixture, unknown provenance, or insufficient data | Not eligible as model evidence |
| 1 | Correlates with incident | Research note only |
| 2 | Predicts action divergence | Predictive finding |
| 3 | Intervention changes action | Serious diagnostic candidate |
| 4 | Reproduces or prevents failure | Strong causal finding |
| 5 | Generalizes across settings | Reusable mechanistic contract |

Findings below Level 3 are candidate research findings, not root-cause claims.

Current offline analysis reaches at most Level 1 for complete, explicitly non-synthetic captures.
Temporal precedence and a nonzero consequence metric do not demonstrate prediction. Synthetic
captures, unknown provenance, and incomplete comparisons remain Level 0 with
`evidence_eligible: false`. These fields survive feature analysis and report generation;
`check-mech` rejects synthetic or ineligible reports.

There is no patch or steering command. A future intervention workflow would need an actual replay
that changes the expected action or outcome before supporting Level 3 or above.

### Negative Controls

A separate negative-control experiment is required research practice before attributing a failure
to model internals. The CLI does not create or verify that such an experiment was performed. When
an adapter marks a capture index `negative_control: true`, analysis checks its paired activation
comparisons. For a post-policy fault such as gripper freeze, z-scale mismatch, middleware velocity
bias, or an action postprocessor clamp, the expected observations are:

```text
raw_policy_action_changed: false
internal_activation_changed: false or irrelevant
postprocessed_action_changed: true
contract_violated: true
```

A stable negative control requires actual paired comparisons for every required captured module.
Missing pairs are insufficient evidence, and nonfinite values or incompatible tensor shapes are
invalid input. One unchanged layer cannot establish stability when another layer changes.

If valid paired activations remain stable while execution changes downstream, the report preserves
the external RobotProbe attribution. First external divergence may refer to an instruction or
observation change; first action divergence is a separate field. Internal lead time is measured
against action divergence, not against every external event.

### What This Does Not Prove

MechProbe does not prove that an SAE feature label is true by inspection.

MechProbe does not work on closed model-server-only systems unless activation probes are exposed.

MechProbe does not turn feature steering into a safety guarantee.

MechProbe does not replace RobotProbe's external trace analysis. It adds optional white-box
evidence when activations are available.

### Current Limits

- There is no live OpenVLA, Pi0.5, or GR00T activation adapter yet; adapters ship as packs.
- Capture can select multiple modules, but there is no temporal patching or activation steering.
- Only manual-direction feature dictionaries are implemented; SAE dictionaries and causal feature
  promotion are not implemented.
- `robotprobe bundle` and MCAP export do not include MechProbe indexes, tensors, or reports
  automatically. Share the entire MechProbe output directory separately, preserving relative
  activation references.

The next adapter should be pack-specific, built against a model you can actually instrument, and
validated with a negative control that shows internals should *not* be blamed for a post-policy
fault.

## NNsight bridge

`NNsightActivationAdapter` captures selected module outputs from an NNsight-supported PyTorch
model. `capture_pair` returns in-memory arrays. The separate `persist_capture_pair` function writes
the activation store, index, and per-run event traces consumed by the existing MechProbe CLI.
The caller supplies the incident, run, model, and replay identity; the bridge cannot infer them
from an activation tensor. Persistence requires a fresh or empty output directory.

### Install

From the repository root:

```bash
uv sync --extra nnsight
```

For an installed package, use `pip install 'robotprobe[nnsight]'` instead. NNsight and Torch are
optional and are not part of the default runtime.

### Capture, persist, and analyze a CPU model

Save the following Python example to a temporary script and run it from the repository root with
`uv run --extra nnsight python /path/to/script.py`. It creates a fresh output directory and prints
its path. The model is deterministic and in evaluation mode; the clean/perturbed traces record its
actual outputs. This demonstrates CPU model capture and file-backed integration, not VLA behavior.

```python
import json
import tempfile
from pathlib import Path

import torch

from robotprobe.core import ReplayLevel
from robotprobe.mech.adapters import NNsightActivationAdapter
from robotprobe.mech.collect import persist_capture_pair
from robotprobe.record import TraceRecorder
from robotprobe.summarize import summarize_trace_pair

out = Path(tempfile.mkdtemp(prefix="mechprobe-nnsight-"))
torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
).eval()
clean_inputs = [torch.zeros(1, 4) for _ in range(3)]
perturbed_inputs = [torch.zeros(1, 4), torch.ones(1, 4), torch.ones(1, 4)]
config = {
    "format": "robotprobe.mech.capture_config.v0",
    "id": "cpu-input-change",
    "target": "cpu-toy",
    "model_access": "white_box",
    "layers": [{"name": "0", "kind": "activation"}, {"name": "2", "kind": "activation"}],
    "steps": {"mode": "every_n", "n": 1},
    "storage": {"format": "safetensors", "dtype": "float32", "max_bytes_per_run": 1000000},
    "claim_boundary": {"no_feature_labels": True, "no_causal_claims": True},
}

for label, inputs in (("clean", clean_inputs), ("perturbed", perturbed_inputs)):
    with TraceRecorder(
        out / f"{label}.jsonl",
        run_id=f"cpu-{label}",
        target="cpu-toy",
        scenario="input-change",
        replay_level=ReplayLevel.POLICY_REPLAY,
    ) as recorder:
        with torch.no_grad():
            for step, x in enumerate(inputs):
                recorder.event(step, "raw_action", "policy_forward", value=model(x)[0].tolist())

pair = NNsightActivationAdapter().capture_pair(model, clean_inputs, perturbed_inputs, config)
persist_capture_pair(
    capture_pair=pair,
    capture_config=config,
    out_dir=out / "mech",
    incident_id="cpu-input-change",
    clean_run_id="cpu-clean",
    perturbed_run_id="cpu-perturbed",
    model={"name": "cpu-linear-demo", "runtime": "torch"},
    target="cpu-toy",
    scenario="input-change",
    replay_level=ReplayLevel.POLICY_REPLAY.value,
    external_first_divergence_step=1,
    synthetic=False,
)
summary = summarize_trace_pair(out / "clean.jsonl", out / "perturbed.jsonl")
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(out)
```

Set `MECH_OUT` to the printed directory, then run:

```bash
uv run robotprobe validate-trace "$MECH_OUT/mech/model_activation_events.clean.jsonl"
uv run robotprobe validate-trace "$MECH_OUT/mech/model_activation_events.perturbed.jsonl"
uv run robotprobe mech diff-activations \
  --mech-dir "$MECH_OUT/mech" --out "$MECH_OUT/mech/activation_diff.json"
uv run robotprobe mech faithfulness \
  --mech-dir "$MECH_OUT/mech" --summary "$MECH_OUT/summary.json" \
  --target-metric raw_action_delta --out "$MECH_OUT/mech/consequence_aopc.json"
uv run robotprobe mech report \
  --summary "$MECH_OUT/summary.json" \
  --activation-diff "$MECH_OUT/mech/activation_diff.json" \
  --out "$MECH_OUT/mech_report.md"
```

Module names follow `model.named_modules()` paths (`"0"`, `"encoder.layer.3"`, and so on). Episode
input index supplies the step, and all modules captured for that input share it. Persisted
activations are immutable snapshots. `bfloat16` capture converts to a NumPy-compatible dtype and
retains `source_dtype`; `storage.dtype` controls the stored representation. Sampling and the
per-run serialized tensor budget follow the [capture configuration](mechprobe.md#capture-configuration-and-adapter-metadata).

`capture_pair` can be used without persistence when only in-memory arrays are needed. Its result
alone is not a file-backed MechProbe directory. The CLI has no live model-loader: `mech collect
--target nnsight` returns `not_implemented` even when the optional dependencies are installed.

### Claim boundary

This adapter is tested on CPU toy models. Real OpenVLA, Pi0.5, or GR00T capture still requires
model-specific runtime integration, correct episode-step alignment, and validation against a
post-policy negative control. These tasks are not supplied by changing a configuration value.

The adapter reads activations only. There is no patching or steering API/CLI here. Complete real
offline comparisons reach at most evidence Level 1; temporal precedence and a consequence score
do not demonstrate prediction or causation. See [the evidence levels](mechprobe.md#evidence-levels)
and [claim boundaries](status-and-claims.md#claims).

Keep the complete `mech/` directory with its reports when sharing results. `robotprobe bundle` and
MCAP export do not automatically include these files.
