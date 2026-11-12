# Command Reference

Every command below exists in the CLI. Run `robotprobe --help` or
`robotprobe <command> --help` for the authoritative argument list.

## Demo and environment

```bash
robotprobe demo --out robotprobe-demo-out   # bundled incident, end to end
robotprobe doctor                           # environment and optional-dependency check
robotprobe doctor --pack examples/packs/libero-action-contracts
robotprobe doctor --integration vla-eval
```

## Traces

```bash
robotprobe validate-trace traces/run.jsonl
robotprobe conformance traces/run.jsonl                 # reports the achieved level
robotprobe conformance traces/run.jsonl --level L2_CONTRACT_CHECKABLE
robotprobe replay --level L0 traces/run.jsonl
robotprobe replay --level L3 --action-envelope safety/action_envelope.yaml traces/run.jsonl
```

`conformance` and `replay` both use `L0`–`L3` names for different taxonomies. See
[RobotProbe Glossary](concepts.md#glossary).

## Comparing runs

```bash
robotprobe summarize \
  --clean traces/clean.jsonl \
  --perturbed traces/gripper_freeze.jsonl \
  --manifest interventions/gripper_freeze_mid.yaml

robotprobe summarize ... --metrics action_energy,jerk,trajectory_deviation,recovery_latency,near_miss

robotprobe report \
  --summary results/summary.json \
  --check results/check.json \
  --title "Gripper freeze" \
  --out results/report.md

robotprobe bundle \
  --clean traces/clean.jsonl \
  --perturbed traces/gripper_freeze.jsonl \
  --manifest interventions/gripper_freeze_mid.yaml \
  --challenge robotprobe-v0-gripper-freeze \
  --out incidents/gripper_freeze
```

## Challenges

```bash
robotsec-range list-challenges
robotsec-range check \
  --challenge robotprobe-v0-gripper-freeze \
  --summary results/summary.json \
  --clean-trace traces/clean.jsonl \
  --perturbed-trace traces/gripper_freeze.jsonl \
  --manifest interventions/gripper_freeze_mid.yaml
robotsec-range check-mech --challenge robotprobe-mech-v0-instruction-internal-leadtime --report mech_report.json
```

`check` validates both traces and independently recomputes the summary and required metrics before
awarding a pass. The supplied summary must agree with that recomputation. Custom challenges require
both trace options; built-in challenges may resolve packaged fixture traces automatically. This
proves consistency with the supplied evidence, not the evidence's authenticity.

## Contracts

```bash
robotprobe contracts check \
  --contract examples/packs/keepout-safety/contracts/keepout_margin.yaml \
  --trace traces/stealth_velocity_bias_keepout.jsonl
```

## Packs

```bash
robotprobe pack init /tmp/my-pack --id my-pack --plugin-id example.contracts.demo --plugin-type contract
robotprobe pack init /tmp/ros2-pack --id ros2-pack --template ros2-command-authority
robotprobe pack validate examples/packs/keepout-safety
robotprobe pack list examples/packs/keepout-safety
robotprobe pack test examples/packs/keepout-safety
robotprobe pack claims examples/packs/keepout-safety
robotprobe pack registry build examples/packs --out /tmp/pack-registry.json
robotprobe pack registry list /tmp/pack-registry.json
```

`pack test` refuses to import pack entrypoints unless `--allow-code` is passed. See
[Pack Authoring Walkthrough](packs.md#authoring-a-pack).

## Export and import

```bash
robotprobe export sidecar --bundle incidents/gripper_freeze --out incidents/gripper_freeze/robotprobe.sidecar.json
robotprobe export mcap    --bundle incidents/gripper_freeze --out incidents/gripper_freeze/robotprobe.mcap
robotprobe import mcap    --mcap incidents/gripper_freeze/robotprobe.mcap --out /tmp/restored
```

See [Observability Bridge](integrations.md#mcap-and-foxglove).

## Ingest

```bash
robotprobe ingest ros2-mcap --input bag.mcap --out traces/run.jsonl \
  --run-id ros2_run_001 --target ros2 --scenario command_authority_replay

robotprobe ingest vla-eval --input episode.json --out traces/episode.jsonl \
  --run-id vla_run_001 --scenario libero_spatial

robotprobe ingest isaac-lab-arena --input isaac_trace.jsonl --out traces/isaac.jsonl \
  --run-id isaac_run_001 --scenario lift_cube
```

ROS 2 CDR payloads are decoded when `mcap-ros2-support` is installed
(`pip install 'robotprobe[ros2-decoders]'`); otherwise payloads fall back to a hash and byte count.

## Fault campaigns

```bash
robotprobe campaign run --campaign examples/campaigns/campaign.json --out out/ --fail-on-violation
```

See [Fault Campaigns](experimental.md#fault-campaigns).

## World-model faithfulness

```bash
robotprobe worldmodel faithfulness --trace traces/imagined_evidence.jsonl --fail-on-divergence
```

See [World-Model Faithfulness Verification](experimental.md#world-model-faithfulness).

## MechProbe (experimental)

Run from the repository root. These commands assume an existing clean/perturbed trace pair and
an output directory; [the full walkthrough](mechprobe.md#commands) uses bundled traces and creates
every prerequisite. Prefix commands with `uv run` when using the repository environment.

```bash
robotprobe mech validate --capture-config examples/mechprobe/capture_config.yaml
robotprobe summarize --clean traces/clean.jsonl --perturbed traces/instruction_swap.jsonl > summary.json
robotprobe mech collect --target synthetic \
  --trace-pair traces/clean.jsonl,traces/instruction_swap.jsonl \
  --capture-config examples/mechprobe/capture_config.yaml \
  --out incidents/instruction_swap/mech
robotprobe validate-trace incidents/instruction_swap/mech/model_activation_events.clean.jsonl
robotprobe validate-trace incidents/instruction_swap/mech/model_activation_events.perturbed.jsonl
robotprobe mech cache-info --mech-dir incidents/instruction_swap/mech
robotprobe mech diff-activations --mech-dir incidents/instruction_swap/mech --out activation_diff.json
robotprobe mech diff-features --mech-dir incidents/instruction_swap/mech \
  --dictionary examples/mechprobe/dictionary.json --out feature_diff.json
robotprobe mech faithfulness --mech-dir incidents/instruction_swap/mech --summary summary.json \
  --target-metric postprocessed_action_delta --out faithfulness.json
robotprobe mech report --summary summary.json --activation-diff activation_diff.json \
  --feature-diff feature_diff.json --out mech_report.md
robotprobe mech report --summary summary.json --activation-diff activation_diff.json \
  --feature-diff feature_diff.json --out mech_report.json
```

`--target synthetic` and its `toy` alias generate fixture activations. Their indexes, diffs, and
reports preserve `synthetic: true` and `evidence_eligible: false`; fixture evidence is Level 0.
`robotsec-range check-mech` requires a JSON report and rejects synthetic/ineligible results.
Real offline comparisons reach at most Level 1. The built-in lead-time challenge checks eligible
real temporal correlation; custom challenges requiring Level 2 cannot pass on current offline
outputs.

Other collection targets return `not_implemented`. Actual model capture uses the
[NNsight Python adapter and persistence bridge](mechprobe.md#nnsight-bridge), followed by the same
file-backed CLI analysis commands. There is no live model-loader, patch, or steering CLI command.

## Public incident datasets

```bash
robotprobe incident-pack download vifailback --out "$HOME/datasets/robotprobe-incidents/raw/vifailback" --task cut_sausage
robotprobe incident-pack convert vifailback \
  --annotation "$HOME/datasets/robotprobe-incidents/raw/vifailback/annotated_data/cut_sausage/cut_sausage_annotations.json" \
  --out "$HOME/datasets/robotprobe-incidents/packs/vifailback-cut-sausage" \
  --limit 3

robotprobe incident-pack download robofac --out "$HOME/datasets/robotprobe-incidents/raw/robofac" \
  --leaf simulation_data/LiftPegUpright-box/view0
robotprobe incident-pack convert robofac \
  --metadata "$HOME/datasets/robotprobe-incidents/raw/robofac/simulation_data/LiftPegUpright-box/view0/view0.json" \
  --h5 "$HOME/datasets/robotprobe-incidents/raw/robofac/simulation_data/LiftPegUpright-box/view0/view0.h5" \
  --out "$HOME/datasets/robotprobe-incidents/packs/robofac-liftpeg" \
  --limit 3 --max-steps 50
```

Requires the `incident-datasets` extra. See [Public Incident Packs](integrations.md#public-incident-packs).

## Simulator demos

```bash
uv run python examples/simulators/gymnasium_cartpole_demo.py --out examples/simulators/out/gymnasium-cartpole
uv run python examples/simulators/security_keepout_demo.py --out examples/simulators/out/security-keepout
```

See [Local Simulator Demos](getting-started.md#simulator-demos).
