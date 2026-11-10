# Golden Root-Cause Report: Action Chunk Boundary Mismatch

This incident represents a chunking or deployment boundary error. The raw policy action is stable,
but the command stream applied downstream changes at the chunk boundary.

Expected diagnosis:

- Layer: `action_postprocess`
- Contract: `ActionChunkContract`
- Fault: `action_chunk_boundary_mismatch`
- Raw policy action changed: `false`
- Postprocessed action changed: `true`

Non-claim: this is not evidence of broad VLA weakness. It is evidence that action chunk handling
needs a contract and regression check.
