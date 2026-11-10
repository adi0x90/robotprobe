# Experimental features

Features that work but have narrow claim boundaries. Read each section's boundary before relying on it.

## Live instrumentation (PolicyProbe)

`PolicyProbe` is an in-process debugging layer for Python policy callables: it streams typed events
from the inference boundary, injects input or output faults while the policy runs, and fires
watchpoints when a predicate matches. It is the form this capability takes in most VLA stacks, where
the policy is a Python function called inside an eval loop.

`PolicyProbe` wraps a `policy(observation) -> action` callable and, per step:

- **hooks** the inference boundary, streaming typed events (`instruction`, `observation`,
  `raw_action`, `postprocessed_action`) to a live `TraceRecorder` (the Frida onEnter/onLeave analogue);
- **intervenes** in-flight via RobotProbe `Attack` objects: `input_attack` perturbs the observation
  (instruction override, stale-frame replay) and `output_attack` perturbs the action (modify-return);
- fires **watchpoints** — predicates over streamed events that invoke a callback on match (the GDB
  watchpoint), recorded on `probe.hits`.

### Usage

```python
import numpy as np
from robotprobe import ReplayLevel, TraceRecorder
from robotprobe.attach import PolicyProbe, Watchpoint
from robotprobe.attacks import ActionFaultAttack, Observation

def policy(obs):                       # your in-process VLA/policy callable
    return model.predict(obs)          # returns an action vector

freeze = ActionFaultAttack(mode="set_dims", dims=(6,), value=0.0, start=20)  # live gripper freeze
gripper_wp = Watchpoint(
    name="gripper_frozen",
    event_type="postprocessed_action",
    predicate=lambda e: e["value"][6] == 0.0,
    on_hit=lambda e: print("watchpoint hit at step", e["step"]),
)

with TraceRecorder("live.jsonl", run_id="r1", target="my-vla", scenario="live_freeze",
                   seed=0, replay_level=ReplayLevel.TRACE_PLAYBACK) as rec:
    probe = PolicyProbe(policy, rec, output_attack=freeze, watchpoints=[gripper_wp])
    probe.run([Observation(image=frame, instruction="pick the red block") for frame in episode])

print(probe.hits)   # [WatchpointHit(watchpoint="gripper_frozen", step=20, ...), ...]
```

`probe.step(step, observation, instruction=None)` instruments a single call; `probe.run(episode)`
iterates an episode of `Observation` (or `(Observation, instruction)` tuples). The streamed trace is a
normal RobotProbe trace — feed it straight into `summarize`, `conformance`, `contracts`, or a
campaign.

### Claim boundary

This hooks an **in-process Python** policy. Native cross-process attach (Frida proper) and live
activation patching of a model runtime are out of scope (they need a running model server / GPU).
The instrumentation itself is CPU-light and model-agnostic. See [Claims](status-and-claims.md#claims).

## Fault Campaigns

A *campaign* sweeps a scheduled set of sim-only faults across a recorded baseline trace and verifies
each synthesized history against contracts, post-hoc. The model is borrowed from distributed-systems
fault injection (Jepsen), mapped onto RobotProbe like this:

| Jepsen | RobotProbe campaign |
|--------|---------------------|
| `nemesis` (scheduled fault process) | the `faults` list, each applied to the baseline over a step window; an intervention manifest is emitted per fault as provenance |
| history | the synthesized perturbed trace per fault (`out/traces/<name>.jsonl`) |
| `checker` (post-hoc, decoupled) | the contract DSL run over each history; consequence metrics + causal diagnosis |

### Claim boundary (read first)

Faults are applied to the **recorded baseline** action / observation / instruction stream.
Outcomes are **not re-simulated** (that needs a policy/simulator = out of scope, GPU). So a campaign
answers *"if this fault were applied to this trajectory, which contracts break and what is the
action-stream consequence?"* — a pre-deployment what-if sweep — not *"what would the robot have
done."* Contracts that depend on sim-derived state (e.g. `keepout_violated`) only fire if that state
is already present in the baseline. To make envelope contracts useful, the campaign attaches
CPU-derivable scalars (`action_l2`, `action_max_abs`) to synthesized action events.

### Run

```bash
robotprobe campaign run --campaign examples/campaigns/campaign.json --out out/ --fail-on-violation
cat out/campaign_report.md
```

`--fail-on-violation` exits non-zero if any fault violates a contract (CI gate).

### Campaign spec

```json
{
  "id": "action-envelope-sweep",
  "baseline": "baseline.jsonl",
  "metrics": ["trajectory_deviation", "action_energy"],
  "contracts": ["contracts/action_envelope.yaml"],
  "faults": [
    {"name": "benign_scale_down", "channel": "action", "mode": "scale_dims", "dims": [0,1,2], "scale": 0.5},
    {"name": "saturate_x", "channel": "action", "mode": "set_dims", "dims": [0], "value": 5.0, "start": 2, "end": 6}
  ]
}
```

Paths resolve relative to the campaign file. Channels and their fault parameters:

- `action` — applied via `ActionFaultAttack`: modes `zero_dims`, `scale_dims` (`scale`),
  `set_dims` (`value`), `offset_dims` (`offset`), `hold_previous_dims`, `random_zero_dim`; with
  `dims`, `start`, `end`, `seed`. Diagnosed as `action_postprocess`.
- `instruction` — replaces the instruction with `payload` in the window. Diagnosed as `instruction`.
- `observation` — replays the held previous frame hash in the window (stale observation). Diagnosed
  as `observation`.

### Output

`out/campaign_report.json` + `.md`, one row per fault with the diagnosed root layer, first-divergence
step, contract verdict, and consequence-metric deltas; plus `out/traces/<name>.jsonl` (history) and
`out/manifests/<name>.yaml` (the emitted nemesis manifest) per fault.

## World-model faithfulness

Checks whether *imagined* evidence matched reality before a policy acted on it.

Systems where a generated subgoal or imagined view conditions a downstream policy/reasoner emit
`world_model_output` (and `subgoal_request`) trace events. Closed-loop imagined rollouts are known to
suffer hallucination and long-horizon error accumulation that silently corrupt the signal the policy
optimizes, which "encourages policies to exploit model inaccuracies rather than genuine task
progress" (2026 world-model-for-VLA literature). This tool flags that divergence per imagined event,
with the lead time before the policy commits.

### Run

```bash
robotprobe worldmodel faithfulness --trace trace.jsonl --fail-on-divergence
```

`--fail-on-divergence` exits non-zero if any imagined event diverged (CI/runtime gate). Options:
`--state-tolerance` (L2-norm threshold for predicted-vs-realized state), `--divergence-threshold` (for an
explicit upstream `divergence` scalar), `--out report.json`.

### What it measures

For each `world_model_output` event it pairs the imagined evidence with the next realized
`observation` and the final `outcome`, and raises divergence signals:

- `success_mismatch` — `predicted_success` on the imagined event != realized `outcome.success`.
- `state_divergence` — L2 (Euclidean-norm) distance between `predicted_state` and the realized
  observation `state` exceeds `--state-tolerance`. This "L2" is the vector norm, not a level in
  either the conformance or replay taxonomy.
- `declared_divergence` — an explicit upstream `divergence` scalar exceeds `--divergence-threshold`.

It also reports `lead_time_steps` — steps between the imagined evidence and the next
`postprocessed_action`. That is the budget a runtime monitor would have to catch the divergence
**before** the policy commits. `faithfulness_rate` = faithful / total imagined events;
`min_divergent_lead_time` is the tightest catch window across divergent events.

### Trace fields

```python
recorder.event(
    5, "world_model_output", "imagined_evidence",
    generated_image_ref="gen/000005.png", request_ref="req/000005.json",
    predicted_state=[1.0, 2.0],   # optional: imagined state to compare to reality
    predicted_success=True,       # optional: imagined task outcome
    # divergence=0.8,             # optional: upstream-computed divergence scalar
)
recorder.event(6, "observation", "observation", image_sha256="...", state=[1.0, 2.0])
```

### Claim boundary

This compares **declared** `predicted_*` fields against realized observation/outcome fields already
in the recorded trace. It does **not** render or compare pixels and does not prove a generated image
is unfaithful by inspection. A divergence is evidence about the recorded declared fields, not a
universal hallucination verdict. See [Claims](status-and-claims.md#claims).
