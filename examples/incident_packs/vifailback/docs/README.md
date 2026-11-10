# ViFailback Incident Pack

This pack converts selected ViFailback annotation JSON records into RobotProbe incident traces.

It is trace-only. It records dataset-provided task text, image references, failure labels,
keyframe references, and correction/avoidance commands. It does not claim a live exploit,
hardware replay, or clean-vs-perturbed causal proof.

Example:

```bash
robotprobe incident-pack download vifailback \
  --out $HOME/datasets/robotprobe-incidents/raw/vifailback \
  --task cut_sausage

robotprobe incident-pack convert vifailback \
  --annotation $HOME/datasets/robotprobe-incidents/raw/vifailback/annotated_data/cut_sausage/cut_sausage_annotations.json \
  --out $HOME/datasets/robotprobe-incidents/packs/vifailback-cut-sausage \
  --limit 3
```
