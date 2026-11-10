# Golden Root-Cause Report: Stale Observation Replay

The stale-observation incident localizes to the observation boundary. The policy reacts to a replayed
camera frame, so raw and postprocessed actions can drift downstream while the root fault remains
sensor freshness.

Expected diagnosis:

- Layer: `observation`
- Contract: `SensorFreshnessContract`
- Fault: `stale_observation_replay`
- Observation replayed: `true`

Non-claim: this is not an action postprocess fault unless the first evidence of divergence appears
after policy output.
