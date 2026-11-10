# Golden Root-Cause Report: Action Scale Mismatch

The incident represents a deployment/action-contract fault where raw policy output is preserved but
the applied action is transformed under the wrong scale.

Expected diagnosis:

- Layer: `action_postprocess`
- Contract: `ActionSpaceContract`
- Fault: `action_scale_mismatch`
- Raw policy action changed: `false`
- Postprocessed action changed: `true`

Non-claim: this does not prove the policy chose a bad action. It proves the action contract after
policy output can be the first fault boundary.
