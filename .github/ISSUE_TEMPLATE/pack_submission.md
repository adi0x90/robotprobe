---
name: Pack submission
about: Propose a new RobotProbe pack, adapter, contract, or golden diagnosis
title: "[pack] "
labels: pack
---

## Pack purpose

What boundary or incident does this pack cover?


## Pack type

- [ ] adapter
- [ ] intervention
- [ ] contract
- [ ] metric
- [ ] diagnoser
- [ ] exporter
- [ ] challenge_pack

## Safety classification

- [ ] trace_only
- [ ] sim_only
- [ ] hardware_shadow
- [ ] hardware_executable

Public alpha packs should not be `hardware_executable`.

## Verification run

```bash
uv run robotprobe pack validate path/to/pack
uv run robotprobe pack test path/to/pack
uv run robotprobe pack claims path/to/pack
```

Paste the relevant output:

```text

```

## Artifacts included

- [ ] `manifest.yaml`
- [ ] contracts/interventions/adapters as declared
- [ ] fixtures
- [ ] golden expected outputs
- [ ] docs/README.md
- [ ] source trace or source reference

## Claims and non-claims

What does this pack prove?


What does this pack not prove?
