# Integrations

Getting data in and out: MCAP and Foxglove, vla-eval, and public incident datasets.

## MCAP and Foxglove

RobotProbe does not replace MCAP, Foxglove, Rerun, ROS tracing, OpenTelemetry, or simulator-native
logs. Those tools are better at broad telemetry capture, visualization, and operations dashboards.

RobotProbe adds a narrower layer:

```text
existing telemetry
  -> RobotProbe trace / sidecar / MCAP topics
  -> clean-vs-perturbed causal diff
  -> intervention manifest
  -> contract check
  -> root-cause report
```

The bridge is useful when a team already has logs but needs a reproducible incident diagnosis:
where the run diverged, which boundary changed, what contract was violated, and what claim is safe
to make from the evidence.

### What This Mimics In A Real Stack

Imagine a robot task still reports success, but a safety envelope was silently violated:

- the model output looked plausible;
- the post-policy action stream drifted toward a keepout zone;
- the normal dashboard shows telemetry but not a diagnosis;
- the incident reviewer needs a compact report and machine-readable evidence.

RobotProbe keeps the original telemetry path intact and writes diagnosis artifacts beside it.

### Generate A Demo Bundle

```bash
uv run python examples/simulators/security_keepout_demo.py \
  --out /tmp/robotprobe-observability-bridge
```

Inspect the report:

```bash
cat /tmp/robotprobe-observability-bridge/report.md
cat /tmp/robotprobe-observability-bridge/check.json
```

Expected shape:

```text
task success remains true
changed layer is action_postprocess
contract is KeepoutZoneContract
fault is stealth_velocity_bias_keepout
```

### Export A RobotProbe Sidecar

```bash
uv run robotprobe export sidecar \
  --bundle /tmp/robotprobe-observability-bridge \
  --out /tmp/robotprobe-observability-bridge/robotprobe.sidecar.json

cat /tmp/robotprobe-observability-bridge/robotprobe.sidecar.json
```

The sidecar is plain JSON for tools that already own storage and visualization. It points to:

- bundle metadata;
- summary JSON;
- RobotSec check result;
- report path;
- trace refs;
- manifest ref.

Use this when an internal platform already has its own artifact store and only needs RobotProbe's
diagnostic layer as attached metadata.

### Export MCAP

```bash
uv run robotprobe export mcap \
  --bundle /tmp/robotprobe-observability-bridge \
  --out /tmp/robotprobe-observability-bridge/robotprobe.mcap
```

RobotProbe writes JSON payloads on RobotProbe-owned topics:

```text
/robotprobe/bundle
/robotprobe/summary
/robotprobe/check
/robotprobe/report
/robotprobe/manifest
/robotprobe/clean_trace
/robotprobe/perturbed_trace
```

This is intentionally conservative. It does not pretend to be a native ROS schema or Foxglove UI
schema. It preserves RobotProbe evidence in an MCAP container that MCAP-aware systems can move,
store, and inspect.

### Round-Trip The MCAP Bundle

```bash
uv run robotprobe import mcap \
  --mcap /tmp/robotprobe-observability-bridge/robotprobe.mcap \
  --out /tmp/robotprobe-observability-restored

find /tmp/robotprobe-observability-restored -maxdepth 2 -type f -print
cat /tmp/robotprobe-observability-restored/report.md
```

Round-trip import is a sanity check: the MCAP export contains enough RobotProbe-owned evidence to
recreate the bundle artifacts.

### How This Relates To Existing Tools

| Existing tool | What it is good at | What RobotProbe adds |
| --- | --- | --- |
| MCAP | Portable log container | Diagnosis topics and bundle round-trip |
| Foxglove | Robot telemetry visualization | Root-cause artifacts beside telemetry |
| Rerun | Multimodal data visualization | Contract and intervention semantics |
| ROS 2 bags | Runtime topic capture | Clean-vs-perturbed comparison and checks |
| vla-eval / Isaac | Evaluation and simulator execution | Failure semantics and evidence packets |

The adoption rule is simple: keep the team's observability stack, then add RobotProbe where
incident forensics starts.

### Current Limits

- RobotProbe's MCAP bridge uses JSON topics, not deep native Foxglove/Rerun schemas yet.
- ROS 2 CDR decoding is available behind the optional `ros2-decoders` extra
  (`python -m pip install -e '.[ros2-decoders]'`), which pulls in pure-Python `mcap-ros2-support`
  and needs no ROS installation. Without that extra, non-JSON payloads fall back to a
  `payload_sha256` + byte count rather than failing — including payloads that are malformed CDR.
- The sidecar/MCAP bridge does not prove live model behavior unless the source bundle came from a
  live model run.
- The hidden keepout demo is sim-only. It demonstrates diagnostic value, not hardware
  exploitability.

## vla-eval harness

AllenAI `vla-evaluation-harness` keeps model servers and benchmark containers decoupled.
RobotProbe should preserve that boundary: it does not fork the harness, replace model servers,
or change benchmark protocol semantics.

RobotProbe integrates at the episode boundary:

1. record the pre-inference observation and instruction;
2. optionally apply observation/instruction interventions;
3. record raw model action output;
4. optionally apply post-policy action interventions;
5. record postprocessed action and outcome;
6. export paired clean/perturbed traces, causal summary, RobotSec check, and report.

The current adapter exposes two integration layers:

- `RobotProbeVlaEvalRunner`: a dependency-light pair runner for local tests and harness-like
  episode callbacks.
- `RobotProbeSyncEpisodeRunner`: an instrumented sync runner that follows AllenAI
  `SyncEpisodeRunner` flow and records RobotProbe traces around `conn.act()` and
  `benchmark.apply_action()`.
- `scripts/vla_eval_real_model_smoke.py`: a real model-server smoke launcher that starts
  official `vla-eval serve --config ...` model-server YAMLs, runs a RobotProbe clean/perturbed
  synthetic episode pair against the websocket, and writes structured success/failure evidence.

Validation has covered both a RobotProbe-instrumented `vla_eval` connection/server smoke and a real
`vla-eval run --no-docker` CLI smoke, against a synthetic benchmark. A full official
benchmark-container run has not been completed; see [Evidence](status-and-claims.md#evidence).

### Alignment With Current vla-eval Source

As of the current AllenAI source checkout, `SyncEpisodeRunner` executes:

1. `benchmark.get_observation()`;
2. `conn.act(obs_dict)`;
3. `benchmark.apply_action(action)`;
4. repeat until `benchmark.is_done()`;
5. return `{"metrics": ..., "steps": ..., "elapsed_sec": ...}`.

The shared protocol types document the expected observation/action convention:

- observation: `images`, `task_description`, `state`;
- action: `actions`.

The built-in `EpisodeRecorder` captures per-step rows through `record_step(**fields)`.
RobotProbe's current adapter can be embedded either around the runner step loop or alongside
that recorder without changing model-server dependencies.

### Smoke Commands

Set `VLA_EVAL_HARNESS` to your checkout of the AllenAI harness before running these.

RobotProbe-instrumented smoke:

```bash
export VLA_EVAL_HARNESS=/path/to/vla-evaluation-harness

python scripts/vla_eval_robotprobe_smoke.py \
  --out runs/vla-eval-smoke \
  --manifest interventions/gripper_freeze_mid.yaml
```

This starts AllenAI's model-server protocol locally, runs clean and perturbed episodes through
the RobotProbe sync runner, writes paired traces, and writes a causal summary.

Real OpenVLA model-server smoke:

```bash
python scripts/vla_eval_real_model_smoke.py \
  --model-name openvla-libero-spatial \
  --server-config "$VLA_EVAL_HARNESS/configs/model_servers/openvla/libero_spatial.yaml" \
  --vla-eval-cwd "$VLA_EVAL_HARNESS" \
  --vla-eval-executable vla-eval \
  --out runs/openvla-libero-spatial \
  --manifest interventions/gripper_freeze_mid.yaml \
  --env HF_HOME="$HF_HOME" \
  --startup-timeout 1200 \
  --episode-timeout 180 \
  --allow-failure
```

Real GR00T/LIBERO model-server smoke:

```bash
python scripts/vla_eval_real_model_smoke.py \
  --model-name groot-libero \
  --server-config "$VLA_EVAL_HARNESS/configs/model_servers/groot/libero.yaml" \
  --vla-eval-cwd "$VLA_EVAL_HARNESS" \
  --vla-eval-executable vla-eval \
  --out runs/groot-libero \
  --manifest interventions/gripper_freeze_mid.yaml \
  --env HF_HOME="$HF_HOME" \
  --startup-timeout 1200 \
  --episode-timeout 180 \
  --allow-failure
```

Both commands preserve the upstream model-server boundary. RobotProbe does not import OpenVLA
or GR00T directly; it launches the official server config, connects through the `vla_eval`
websocket client, records the episode-level trace, and emits
`model-server-smoke-result.json`. If model download, license gating, CUDA setup, dependency
resolution, or server startup fails, the result JSON includes the exact command, config path,
failure stage, error, and model-server log tail.

## Public Incident Packs

RobotProbe can seed real incident packs from public robotics failure datasets without
mirroring entire datasets into this repository.

The goal is to produce small, reproducible, evidence-bearing traces from real robot failure
corpora. These are useful for demos, regression tests, pack authoring, and external review.

### Datasets

#### ViFailback

Source: `sii-rhos-ai/ViFailback-Dataset` on Hugging Face.

Useful because it is failure-centric: annotations include task text, chronological image refs,
failure detection, failure type, failure subtask, avoid/correct keyframes, and low-level
avoidance/correction commands.

RobotProbe conversion:

- emits one JSONL trace per selected annotation record;
- records task text as an `instruction` event;
- records image paths as `observation` events;
- records the dataset failure label as a `DatasetFailureAnnotation` `contract_check`;
- records failure type, subtask, keyframe, and correction guidance in the `outcome` event;
- writes a small golden diagnosis JSON next to each trace.

This is an annotated incident trace, not a clean-vs-perturbed causal pair.

#### RoboFAC

Source: `MINT-SJTU/RoboFAC-dataset` on Hugging Face.

Useful because each leaf directory has metadata JSON plus optional H5 trajectory data. The metadata
identifies episodes, success/failure labels, environments, seeds, and unique IDs. The H5 file can
include real action, reward, success, and simulator/robot-state arrays.

RobotProbe conversion:

- emits one JSONL trace per selected episode;
- records task/environment context as an `instruction` event;
- records action arrays as `raw_action` and `postprocessed_action` events when H5 is available;
- records robot state and reward streams when available;
- records unsuccessful episodes as a `RoboFACFailureOutcome` `contract_check`;
- writes a small golden diagnosis JSON next to each trace.

This is also an annotated incident trace. It becomes more trajectory-rich when the H5 file is
downloaded.

### Install Extras

The core package stays lightweight. Dataset download/H5 conversion uses optional dependencies:

```bash
python -m pip install -e '.[incident-datasets]'
```

If you are downloading from Hugging Face regularly, log in locally:

```bash
huggingface-cli login
```

### Storage Location

Keep raw public datasets outside the repository:

```bash
mkdir -p "$HOME/datasets/robotprobe-incidents"
```

Recommended layout:

```text
~/datasets/robotprobe-incidents/
  raw/
    vifailback/
    robofac/
  packs/
    vifailback-cut-sausage/
    robofac-liftpeg/
```

### ViFailback Quickstart

Download a small seed:

```bash
robotprobe incident-pack download vifailback \
  --out "$HOME/datasets/robotprobe-incidents/raw/vifailback" \
  --task cut_sausage
```

Convert the first three annotated incidents:

```bash
robotprobe incident-pack convert vifailback \
  --annotation "$HOME/datasets/robotprobe-incidents/raw/vifailback/annotated_data/cut_sausage/cut_sausage_annotations.json" \
  --out "$HOME/datasets/robotprobe-incidents/packs/vifailback-cut-sausage" \
  --limit 3
```

Validate one produced trace:

```bash
robotprobe validate-trace \
  "$HOME/datasets/robotprobe-incidents/packs/vifailback-cut-sausage/traces/cut_sausage_episode_23.jsonl"
```

### RoboFAC Quickstart

Download one small RoboFAC leaf with H5:

```bash
robotprobe incident-pack download robofac \
  --out "$HOME/datasets/robotprobe-incidents/raw/robofac" \
  --leaf simulation_data/LiftPegUpright-box/view0
```

Convert a small capped sample:

```bash
robotprobe incident-pack convert robofac \
  --metadata "$HOME/datasets/robotprobe-incidents/raw/robofac/simulation_data/LiftPegUpright-box/view0/view0.json" \
  --h5 "$HOME/datasets/robotprobe-incidents/raw/robofac/simulation_data/LiftPegUpright-box/view0/view0.h5" \
  --out "$HOME/datasets/robotprobe-incidents/packs/robofac-liftpeg" \
  --limit 3 \
  --max-steps 50
```

Validate one produced trace:

```bash
robotprobe validate-trace \
  "$HOME/datasets/robotprobe-incidents/packs/robofac-liftpeg/traces/liftpegupright_box_episode_0.jsonl"
```

### Pack Validation

The adapters are also exposed as example packs:

```bash
robotprobe pack validate examples/incident_packs/vifailback
robotprobe pack validate examples/incident_packs/robofac
robotprobe pack registry build examples/incident_packs --out /tmp/robotprobe-incident-pack-registry.json
```

### Claim Boundary

These packs prove that RobotProbe can ingest selected public robot failure datasets and represent
their incidents as typed traces with machine-readable diagnosis artifacts.

They do not prove:

- live robot exploitability;
- physical hardware safety;
- broad VLA robustness;
- clean-vs-perturbed causal attribution unless a dataset provides a true paired run;
- that image/video refs have been redistributed under this repository.

Raw dataset files remain external artifacts under their original licenses.
