# RoboFAC Incident Pack

This pack converts selected RoboFAC episode metadata and optional H5 action/state trajectories
into RobotProbe incident traces.

It is trace-only. With an H5 file, the converter records action, applied-action, robot-state,
reward, and outcome events. Without an H5 file, it still records episode metadata and the
dataset success/failure label.

Example:

```bash
robotprobe incident-pack download robofac \
  --out $HOME/datasets/robotprobe-incidents/raw/robofac \
  --leaf simulation_data/LiftPegUpright-box/view0

robotprobe incident-pack convert robofac \
  --metadata $HOME/datasets/robotprobe-incidents/raw/robofac/simulation_data/LiftPegUpright-box/view0/view0.json \
  --h5 $HOME/datasets/robotprobe-incidents/raw/robofac/simulation_data/LiftPegUpright-box/view0/view0.h5 \
  --out $HOME/datasets/robotprobe-incidents/packs/robofac-liftpeg \
  --limit 3 \
  --max-steps 50
```
