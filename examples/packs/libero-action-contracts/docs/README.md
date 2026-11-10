# LIBERO Action Contracts Pack

This pack is a golden root-cause corpus for action-contract and input-boundary incidents around
LIBERO/Pi0.5-style traces.

It is the first pack intended to teach RobotProbe's core diagnostic distinction:

```text
Do not blame the model when the raw policy action stayed unchanged and the first changed boundary
is action postprocess.
```

Included incident families:

- `gripper_freeze_mid`: post-policy gripper command freshness failure.
- `action_scale_mismatch`: post-policy unit/scale contract failure.
- `action_chunk_boundary_mismatch`: delayed or shifted action chunk boundary.
- `stale_observation_replay`: sensor freshness failure.
- `instruction_swap`: instruction integrity failure.

Run:

```bash
robotprobe pack validate examples/packs/libero-action-contracts
robotprobe pack test examples/packs/libero-action-contracts
robotprobe pack claims examples/packs/libero-action-contracts
```

The `golden/summaries` files are compared with their `fixtures/` counterparts by `pack test`, so the
pair catches drift after the diagnosis was reviewed rather than proving it. The `golden/reports`
files are human-readable root-cause reports for reviewers and are not checked by `pack test`.

Claim boundary:

- This pack demonstrates RobotProbe's evidence format and diagnosis checks for recorded incidents.
- It does not demonstrate hardware exploitability, hardware safety, broad model robustness, or model
  failure when the evidence localizes after policy output.
