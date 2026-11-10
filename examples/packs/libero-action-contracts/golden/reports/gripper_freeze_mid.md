# Golden Root-Cause Report: Gripper Freeze Mid-Episode

The recorded LIBERO/Pi0.5 incident localizes to `action_postprocess`, not to the policy forward pass.
The raw policy action remains unchanged while the postprocessed action diverges at step 35.

Normal success-rate evaluation can miss this kind of deployment fault because the binary success
label does not tell the reviewer that the command contract was violated or that the episode became
less efficient.

Expected diagnosis:

- Layer: `action_postprocess`
- Contract: `ActionSpaceContract`
- Fault: `gripper_freeze_mid`
- First divergence: `35`
- Raw policy action changed: `false`
- Postprocessed action changed: `true`

Non-claim: this report does not prove a model failure. The evidence says the first meaningful
boundary violation is after the policy action is emitted.
