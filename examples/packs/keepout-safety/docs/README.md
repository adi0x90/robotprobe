# Keepout Safety Pack

This example pack demonstrates the RobotProbe v0.2 plugin/pack shape. It is intentionally
simulator-only.

It contains:

- a `contract` plugin declaration for a recorded keepout violation;
- an `intervention` plugin declaration for the hidden keepout velocity-bias scenario;
- a RobotSec challenge that expects an action-postprocess root cause;
- safety and claim boundaries in the pack manifest.

Run:

```bash
robotprobe pack validate examples/packs/keepout-safety
robotprobe pack list examples/packs/keepout-safety
```

After generating a security demo bundle, check the contract against the perturbed trace:

```bash
python examples/simulators/security_keepout_demo.py --out /tmp/robotprobe-keepout
robotprobe contracts check \
  --contract examples/packs/keepout-safety/contracts/keepout_margin.yaml \
  --trace /tmp/robotprobe-keepout/traces/stealth_velocity_bias_keepout.jsonl
```
